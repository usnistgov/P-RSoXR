"""
GPU-accelerated batch fitting for polarized X-ray reflectometry using JAX.

Mirrors the ModelsBatchBuilder / gpu_fit_models_cmaes workflow from
Refltools/gpu_reflect.py but handles the additional complexity of the
P-RSoXR 4×4 transfer matrix method:

  - Two arrays per energy (slabs + dielectric tensor) instead of one
  - Energy-dependent tensor from structure.tensor(energy=E)
  - Per-layer 3×3 complex dielectric tensors (not just SLD + iSLD)
  - Per-objective polarization state ('s', 'p', 'sp', 'ps')

Design: CPU evaluates the model (parameter→slab+tensor mapping) once per
generation; JAX evaluates reflectivity and chi2 for all candidates in
parallel on GPU. This is correct for arbitrary parameter models including
PXR_MaterialSLD with energy-dependent interpolated optical constants.

For parameters that map DIRECTLY to slab or tensor entries, the inner
generation loop can optionally be run entirely on GPU via jax.lax.fori_loop
(see use_jax_scatter=True — experimental, requires all free params to have
linear slab/tensor mappings).
"""

import copy
import time
import warnings

import numpy as np
import jax
import jax.numpy as jnp
from jax import jit

from pypxr._jax_uniaxial_reflect import (
    juniaxial_reflectivity_scan,
    uniaxial_reflectivity_jax,
    uniaxial_reflectivity_jax_f32,
)

jax.config.update("jax_enable_x64", True)

_FWHM = 2 * np.sqrt(2 * np.log(2.0))
_INTLIMIT = 3.5

# Polarization index mapping — matches PXR_ReflectModel.model() convention:
#   refl[nq, 0, 0] = |r_ss|², refl[nq, 1, 1] = |r_pp|²
#   pol='s' → refl[:, 1, 1], pol='p' → refl[:, 0, 0]  (model.py line 316-319)
_POL_IDX = {'ss': (0, 0), 'sp': (0, 1), 'ps': (1, 0), 'pp': (1, 1),
            's': (1, 1), 'p': (0, 0)}


# ---------------------------------------------------------------------------
# Utility: extract objective data and model state
# ---------------------------------------------------------------------------

def _objective_data(objective):
    """Return (q, y, y_err) float64 arrays from an objective's dataset."""
    d = objective.data
    return (
        np.asarray(d.x, dtype=np.float64),
        np.asarray(d.y, dtype=np.float64),
        np.asarray(d.y_err, dtype=np.float64),
    )


def _get_pxr_state(objective, energy):
    """
    Return (slabs, tensor, scale, bkg) reflecting current parameter values.

    slabs  : (nlayers, 4) float64
    tensor : (nlayers, 3, 3) complex128
    scale  : float
    bkg    : float
    """
    slabs = objective.model.structure.slabs()[:, :4].astype(np.float64)
    tensor = objective.model.structure.tensor(energy=energy).astype(np.complex128)
    scale = float(objective.model.scale.value)
    bkg = float(objective.model.bkg.value)
    return slabs, tensor, scale, bkg


def _extract_free_params(objective):
    """Return varying Parameter objects in flattened order."""
    return [p for p in objective.parameters.flattened() if p.vary]


def _get_bounds(free_params):
    """Return (n_free, 2) array of [lb, ub]."""
    bounds = []
    for p in free_params:
        lb = p.bounds.lb if hasattr(p.bounds, 'lb') else float(p.bounds[0])
        ub = p.bounds.ub if hasattr(p.bounds, 'ub') else float(p.bounds[1])
        bounds.append([lb, ub])
    return np.array(bounds, dtype=np.float64)


def _set_free_params(objective, values):
    """Set free parameters of objective to values in-place."""
    for p, v in zip(_extract_free_params(objective), values):
        p.value = float(v)


def _detect_transform(objective):
    """Return True if objective uses log10 (logY) transform."""
    t = getattr(objective, 'transform', None)
    return t is not None and getattr(t, 'form', None) == 'logY'


def _get_pol_indices(objective):
    """Return (pi, pj) polarization output indices for this objective."""
    pol = getattr(objective.model, 'pol', 's')
    return _POL_IDX.get(pol, (1, 1))


# ---------------------------------------------------------------------------
# GPU chi2 kernels
# ---------------------------------------------------------------------------

@jit
def _chi2_linear_batch(q_pad, y_pad, y_err_pad, mask,
                       slabs, tensors, energies, scale, bkg,
                       pol_i, pol_j, use_f32):
    """
    Compute chi2 for (n_e, popsize) candidates.

    q_pad    : (n_e, max_nq) float64
    y_pad    : (n_e, max_nq) float64
    y_err_pad: (n_e, max_nq) float64
    mask     : (n_e, max_nq) float64  (1 = valid point, 0 = padded)
    slabs    : (n_e, popsize, nlayers, 4) float64
    tensors  : (n_e, popsize, nlayers, 3, 3) complex128
    energies : (n_e,) float64
    scale    : (n_e, popsize) float64
    bkg      : (n_e, popsize) float64
    pol_i    : (n_e,) int32
    pol_j    : (n_e,) int32

    Returns
    -------
    chi2 : (n_e, popsize) float64
    """
    def one_energy(q_e, y_e, yerr_e, mask_e, slabs_e, tensors_e,
                   energy_e, scale_e, bkg_e, pi_e, pj_e):
        # slabs_e: (popsize, nlayers, 4); tensors_e: (popsize, nlayers, 3, 3)
        def one_candidate(s_c, t_c, sc_c, bk_c):
            refl, _ = juniaxial_reflectivity_scan(q_e, s_c, t_c, energy_e)
            r = refl[:, pi_e, pj_e] * sc_c + bk_c
            residuals = (y_e - r) / yerr_e
            # nan_to_num: padded q=0 positions give NaN from degenerate kz; mask keeps them out
            safe_sq = jnp.nan_to_num(residuals ** 2, nan=0.0, posinf=0.0, neginf=0.0)
            return jnp.sum(mask_e * safe_sq)
        return jax.vmap(one_candidate)(slabs_e, tensors_e, scale_e, bkg_e)

    return jax.vmap(one_energy)(
        q_pad, y_pad, y_err_pad, mask, slabs, tensors,
        energies, scale, bkg, pol_i, pol_j,
    )


@jit
def _chi2_log_batch(q_pad, y_log_pad, y_err_log_pad, mask,
                    slabs, tensors, energies, scale, bkg,
                    pol_i, pol_j):
    """Log10-transform chi2 for (n_e, popsize) candidates."""
    def one_energy(q_e, y_e, yerr_e, mask_e, slabs_e, tensors_e,
                   energy_e, scale_e, bkg_e, pi_e, pj_e):
        def one_candidate(s_c, t_c, sc_c, bk_c):
            refl, _ = juniaxial_reflectivity_scan(q_e, s_c, t_c, energy_e)
            r = refl[:, pi_e, pj_e] * sc_c + bk_c
            r_log = jnp.log10(jnp.maximum(r, jnp.float64(1e-100)))
            residuals = (y_e - r_log) / yerr_e
            # nan_to_num: padded q=0 positions give NaN from degenerate kz; mask keeps them out
            safe_sq = jnp.nan_to_num(residuals ** 2, nan=0.0, posinf=0.0, neginf=0.0)
            return jnp.sum(mask_e * safe_sq)
        return jax.vmap(one_candidate)(slabs_e, tensors_e, scale_e, bkg_e)

    return jax.vmap(one_energy)(
        q_pad, y_log_pad, y_err_log_pad, mask, slabs, tensors,
        energies, scale, bkg, pol_i, pol_j,
    )


# ---------------------------------------------------------------------------
# PXRBatchBuilder
# ---------------------------------------------------------------------------

class PXRBatchBuilder:
    """
    Converts a collection of refnx.Objective objects with PXR_ReflectModel
    into GPU-batched fitness evaluations for CMA-ES optimisation.

    Handles the full generality of P-RSoXR models:
      - Parameters may map to slab entries (thickness, roughness) or to
        dielectric tensor entries (optical constants), or be derived via
        an energy-dependent physical model (PXR_MaterialSLD).
      - The model is evaluated on CPU per generation; JAX evaluates the
        reflectivity calculation and chi2 in parallel on GPU.

    Each objective must use a single polarization state ('s' or 'p'), not
    a concatenated 'sp' objective. For simultaneous s+p fitting, pass both
    as separate objectives in the list/dict.

    Parameters
    ----------
    objectives : dict {energy_eV: Objective} or list of Objectives
        If a dict, energies are taken from the keys (sorted).
        If a list, the energy is read from obj.model.energy for each objective.
    energy_list : list of float, optional
        Overrides the energy list when objectives is a dict.
    """

    def __init__(self, objectives, energy_list=None):
        if isinstance(objectives, dict):
            objectives_dict = objectives
            if energy_list is None:
                energy_list = sorted(objectives_dict.keys())
            self.energies = list(energy_list)
            self.objectives = [objectives_dict[e] for e in self.energies]
        else:
            # List of objectives — energy read from each model
            self.objectives = list(objectives)
            self.energies = [float(obj.model.energy) for obj in self.objectives]
        self.n_energies = len(self.energies)

        # Data arrays per energy
        self.q_list, self.y_list, self.y_err_list = [], [], []
        for obj in self.objectives:
            q, y, ye = _objective_data(obj)
            self.q_list.append(q)
            self.y_list.append(y)
            self.y_err_list.append(ye)

        self.n_q_list = [len(q) for q in self.q_list]
        self.max_nq = max(self.n_q_list)

        # Pad Q/data to max_nq
        self.q_pad    = np.zeros((self.n_energies, self.max_nq), np.float64)
        self.y_pad    = np.zeros((self.n_energies, self.max_nq), np.float64)
        self.y_err_pad = np.ones((self.n_energies, self.max_nq), np.float64)
        self.mask      = np.zeros((self.n_energies, self.max_nq), np.float64)
        for i, (nq, q, y, ye) in enumerate(
            zip(self.n_q_list, self.q_list, self.y_list, self.y_err_list)
        ):
            self.q_pad[i, :nq] = q
            self.y_pad[i, :nq] = y
            self.y_err_pad[i, :nq] = ye
            self.mask[i, :nq] = 1.0

        # Transform detection (use first objective as representative)
        self.use_log = _detect_transform(self.objectives[0])
        if self.use_log:
            self.y_log_pad = np.log10(np.maximum(self.y_pad, 1e-100))
            y_safe = np.where(self.mask > 0, self.y_pad, 1.0)
            self.y_err_log_pad = np.abs(self.y_err_pad / (y_safe * np.log(10)))

        # Polarization indices per energy
        self.pol_i = np.array([_get_pol_indices(obj)[0] for obj in self.objectives], np.int32)
        self.pol_j = np.array([_get_pol_indices(obj)[1] for obj in self.objectives], np.int32)

        # Free parameters — assumed same structure across all energies
        self.free_params = _extract_free_params(self.objectives[0])
        self.n_free = len(self.free_params)
        self.bounds = _get_bounds(self.free_params)

        # Per-energy bounds (n_energies, n_free, 2) for correct CMA-ES clipping
        self.per_energy_bounds = np.stack(
            [_get_bounds(_extract_free_params(obj)) for obj in self.objectives]
        )

        # Current base parameter values per energy
        self.base_values = np.stack(
            [np.array([p.value for p in _extract_free_params(obj)], np.float64)
             for obj in self.objectives]
        )  # (n_energies, n_free)

        # Base model state per energy
        slabs_list, tensor_list, scale_list, bkg_list = [], [], [], []
        for obj, e in zip(self.objectives, self.energies):
            s, t, sc, bk = _get_pxr_state(obj, e)
            slabs_list.append(s)
            tensor_list.append(t)
            scale_list.append(sc)
            bkg_list.append(bk)

        self.base_slabs  = np.stack(slabs_list)   # (n_e, nlayers, 4)
        self.base_tensor = np.stack(tensor_list)  # (n_e, nlayers, 3, 3) complex128
        self.base_scale  = np.array(scale_list)   # (n_e,)
        self.base_bkg    = np.array(bkg_list)      # (n_e,)
        self.n_slab_rows    = self.base_slabs.shape[1]
        self.n_tensor_layers = self.base_tensor.shape[1]

        # Persistent GPU arrays (transferred once, reused every generation)
        self._q_pad_gpu    = jnp.array(self.q_pad)
        self._mask_gpu     = jnp.array(self.mask)
        self._energies_gpu = jnp.array(self.energies, dtype=jnp.float64)
        self._pol_i_gpu    = jnp.array(self.pol_i, dtype=jnp.int32)
        self._pol_j_gpu    = jnp.array(self.pol_j, dtype=jnp.int32)
        if self.use_log:
            self._y_log_gpu     = jnp.array(self.y_log_pad)
            self._y_err_log_gpu = jnp.array(self.y_err_log_pad)
        else:
            self._y_gpu     = jnp.array(self.y_pad)
            self._y_err_gpu = jnp.array(self.y_err_pad)

    # ------------------------------------------------------------------
    # CPU model evaluation: parameter values → slabs + tensors
    # ------------------------------------------------------------------

    def build_batch(self, pop_matrix):
        """
        Evaluate the model for every candidate at every energy.

        Parameters
        ----------
        pop_matrix : (n_energies, popsize, n_free) ndarray

        Returns
        -------
        slabs   : (n_e, popsize, nlayers, 4) float64
        tensors : (n_e, popsize, nlayers, 3, 3) complex128
        scale   : (n_e, popsize) float64
        bkg     : (n_e, popsize) float64
        """
        n_e, popsize, _ = pop_matrix.shape
        slabs_out   = np.empty((n_e, popsize, self.n_slab_rows, 4), np.float64)
        tensors_out = np.empty((n_e, popsize, self.n_tensor_layers, 3, 3), np.complex128)
        scale_out   = np.empty((n_e, popsize), np.float64)
        bkg_out     = np.empty((n_e, popsize), np.float64)

        for i, (obj, energy) in enumerate(zip(self.objectives, self.energies)):
            fp = _extract_free_params(obj)
            saved = np.array([p.value for p in fp])
            for j in range(popsize):
                _set_free_params(obj, pop_matrix[i, j])
                slabs_out[i, j]   = obj.model.structure.slabs()[:, :4]
                tensors_out[i, j] = obj.model.structure.tensor(energy=energy)
                scale_out[i, j]   = float(obj.model.scale.value)
                bkg_out[i, j]     = float(obj.model.bkg.value)
            # Restore base values
            _set_free_params(obj, saved)

        return slabs_out, tensors_out, scale_out, bkg_out

    # ------------------------------------------------------------------
    # GPU fitness evaluation
    # ------------------------------------------------------------------

    def fitness(self, pop_matrix):
        """
        Compute chi2 for all energies and candidates.

        Parameters
        ----------
        pop_matrix : (n_energies, popsize, n_free) ndarray or JAX array

        Returns
        -------
        chi2 : (n_energies, popsize) ndarray
        """
        pop_np = np.array(pop_matrix)
        slabs, tensors, scale, bkg = self.build_batch(pop_np)

        slabs_j   = jnp.array(slabs)
        tensors_j = jnp.array(tensors)
        scale_j   = jnp.array(scale)
        bkg_j     = jnp.array(bkg)

        if self.use_log:
            chi2 = _chi2_log_batch(
                self._q_pad_gpu, self._y_log_gpu, self._y_err_log_gpu,
                self._mask_gpu, slabs_j, tensors_j, self._energies_gpu,
                scale_j, bkg_j, self._pol_i_gpu, self._pol_j_gpu,
            )
        else:
            chi2 = _chi2_linear_batch(
                self._q_pad_gpu, self._y_gpu, self._y_err_gpu,
                self._mask_gpu, slabs_j, tensors_j, self._energies_gpu,
                scale_j, bkg_j, self._pol_i_gpu, self._pol_j_gpu,
                False,
            )

        return np.array(chi2)  # (n_e, popsize)


# ---------------------------------------------------------------------------
# pxr_gpu_fit_models_cmaes
# ---------------------------------------------------------------------------

def pxr_gpu_fit_models_cmaes(
    objectives,
    energy_list=None,
    popsize=20,
    n_generations=500,
    seed=0,
    verbose=False,
    tol=1e-4,
    patience=5,
    check_every=10,
    _algo_cls=None,
    _sigma_init=None,
):
    """
    Fit P-RSoXR models for multiple energies/polarizations simultaneously
    using Sep-CMA-ES.

    Mirrors gpu_fit_models_cmaes from Refltools/gpu_sweep_optimizer.py but
    uses the 4×4 TMM reflectivity calculation for polarised X-ray data and
    evaluates the full model (parameter→slab+tensor) on CPU per generation.

    The reflectivity calculation and chi2 summation run on GPU via JAX.
    Early stopping and convergence detection follow the same logic as the
    standard XRR workflow.

    Parameters
    ----------
    objectives : dict {energy_eV: Objective} or list of Objectives
        Objectives with PXR_ReflectModel (single polarization each).
        If a dict, energy_list defaults to sorted keys.
        If a list, energy is read from obj.model.energy for each objective.
    energy_list     : list of energies (only used when objectives is a dict)
    popsize         : population size per energy (default 20)
    n_generations   : maximum number of generations (default 500)
    seed            : random seed for CMA-ES initialisation
    verbose         : print progress every check_every generations
    tol             : relative improvement threshold for early stopping
    patience        : consecutive non-improving checks before stopping
    check_every     : generations between convergence checks
    _algo_cls       : evosax strategy class (default Sep_CMA_ES)
    _sigma_init     : initial step-size sigma (default: algorithm default)

    Returns
    -------
    dict with keys:
        'fitted_objectives' : {energy: Objective at best-fit params}
        'best_chisqr'       : {energy: float64}
        'energies'          : list of processed energies
        'elapsed_sec'       : wall time in seconds
        'generations_run'   : actual generations completed
        'converged'         : True if early stopping fired
    """
    try:
        from evosax.algorithms import Sep_CMA_ES
    except ImportError as exc:
        raise ImportError(
            "evosax is required for pxr_gpu_fit_models_cmaes. "
            "Install with: pip install evosax"
        ) from exc

    if _algo_cls is None:
        _algo_cls = Sep_CMA_ES

    builder = PXRBatchBuilder(objectives, energy_list)
    n_energies = builder.n_energies
    n_free = builder.n_free

    per_lb = jnp.array(builder.per_energy_bounds[:, :, 0])  # (n_e, n_free)
    per_ub = jnp.array(builder.per_energy_bounds[:, :, 1])

    # Initial mean: centre of shared bounds
    mean_init = jnp.array((builder.bounds[:, 0] + builder.bounds[:, 1]) / 2.0)

    solution_proto = np.zeros(n_free, dtype=np.float64)
    strategy = _algo_cls(population_size=popsize, solution=solution_proto)
    params = strategy.default_params
    if _sigma_init is not None:
        params = params.replace(std_init=_sigma_init)

    v_init = jax.jit(jax.vmap(strategy.init, in_axes=(0, None, None)))
    v_ask  = jax.jit(jax.vmap(strategy.ask,  in_axes=(0, 0, None)))
    v_tell = jax.jit(jax.vmap(strategy.tell, in_axes=(0, 0, 0, 0, None)))

    master_key = jax.random.PRNGKey(seed)
    master_key, init_key = jax.random.split(master_key)
    init_keys = jax.random.split(init_key, n_energies)
    states = v_init(init_keys, mean_init, params)

    # Warmup: trigger JIT compilation before timing
    if verbose:
        print("  Warming up JIT compilation...")
    master_key, gen_key = jax.random.split(master_key)
    keys = jax.random.split(gen_key, n_energies)
    populations, states = v_ask(keys, states, params)
    populations = jnp.clip(populations, per_lb[:, None, :], per_ub[:, None, :])
    chi2_warmup = jnp.array(builder.fitness(np.array(populations)), dtype=jnp.float64)
    states, _ = v_tell(keys, populations, chi2_warmup, states, params)
    jax.block_until_ready(states.best_fitness)
    generations_run = 1
    if verbose:
        print("  Warmup done.")

    prev_mean_best = np.inf
    no_improve_count = 0
    converged = False
    t_start = time.perf_counter()

    for gen in range(n_generations - 1):
        master_key, gen_key = jax.random.split(master_key)
        keys = jax.random.split(gen_key, n_energies)

        populations, states = v_ask(keys, states, params)
        populations = jnp.clip(populations, per_lb[:, None, :], per_ub[:, None, :])

        # CPU model eval + GPU reflectivity + chi2
        chi2 = jnp.array(builder.fitness(np.array(populations)), dtype=jnp.float64)

        states, _ = v_tell(keys, populations, chi2, states, params)
        generations_run += 1

        if (gen + 1) % check_every == 0:
            best = np.array(states.best_fitness)  # GPU→CPU sync
            mean_best = float(best.mean())

            if verbose:
                print(
                    f"  gen {generations_run:4d}/{n_generations}  "
                    f"mean_χ²={mean_best:.3f}  "
                    f"min_χ²={float(best.min()):.3f}  "
                    f"max_χ²={float(best.max()):.3f}"
                )

            if tol > 0 and np.isfinite(prev_mean_best):
                rel_improvement = (prev_mean_best - mean_best) / max(prev_mean_best, 1e-12)
                if rel_improvement < tol:
                    no_improve_count += 1
                    if no_improve_count >= patience:
                        if verbose:
                            print(
                                f"  Early stop at gen {generations_run}: "
                                f"improvement {rel_improvement:.2e} < tol {tol:.0e} "
                                f"for {patience} consecutive checks."
                            )
                        converged = True
                        break
                else:
                    no_improve_count = 0
            prev_mean_best = mean_best

    elapsed = time.perf_counter() - t_start

    best_solutions = np.clip(
        np.array(states.best_solution),
        builder.per_energy_bounds[:, :, 0],
        builder.per_energy_bounds[:, :, 1],
    )  # (n_e, n_free)

    # Re-evaluate best solutions for accurate final chi2
    best_slabs, best_tensors, best_scale, best_bkg = builder.build_batch(
        best_solutions[:, np.newaxis, :]  # (n_e, 1, n_free)
    )
    slabs_j   = jnp.array(best_slabs)
    tensors_j = jnp.array(best_tensors)
    scale_j   = jnp.array(best_scale)
    bkg_j     = jnp.array(best_bkg)

    if builder.use_log:
        best_chi2_arr = np.array(_chi2_log_batch(
            builder._q_pad_gpu, builder._y_log_gpu, builder._y_err_log_gpu,
            builder._mask_gpu, slabs_j, tensors_j, builder._energies_gpu,
            scale_j, bkg_j, builder._pol_i_gpu, builder._pol_j_gpu,
        ))[:, 0]
    else:
        best_chi2_arr = np.array(_chi2_linear_batch(
            builder._q_pad_gpu, builder._y_gpu, builder._y_err_gpu,
            builder._mask_gpu, slabs_j, tensors_j, builder._energies_gpu,
            scale_j, bkg_j, builder._pol_i_gpu, builder._pol_j_gpu,
            False,
        ))[:, 0]

    fitted_objectives = {}
    best_chisqr_dict = {}
    for i, (energy, obj) in enumerate(zip(builder.energies, builder.objectives)):
        obj_fitted = copy.deepcopy(obj)
        _set_free_params(obj_fitted, best_solutions[i])
        pol = getattr(getattr(obj_fitted, 'model', None), 'pol', 'unknown')
        key = (energy, pol)
        fitted_objectives[key] = obj_fitted
        best_chisqr_dict[key] = float(best_chi2_arr[i])

    return {
        'fitted_objectives': fitted_objectives,
        'best_chisqr':       best_chisqr_dict,
        'energies':          list(builder.energies),
        'elapsed_sec':       elapsed,
        'generations_run':   generations_run,
        'converged':         converged,
    }
