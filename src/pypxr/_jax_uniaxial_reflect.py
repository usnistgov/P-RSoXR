"""
JAX implementation of the uniaxial 4×4 transfer matrix method for
polarized X-ray reflectivity from anisotropic stratified media.

Mirrors the numpy implementation in _uniaxial_reflect.py but uses JAX
for JIT compilation and GPU-accelerated batch evaluation.

Usage
-----
from pypxr._jax_uniaxial_reflect import uniaxial_reflectivity_jax
refl, tran = uniaxial_reflectivity_jax(q_jnp, slabs_jnp, tensor_jnp, energy)
"""

import numpy as np
import jax
import jax.numpy as jnp
from jax import jit

jax.config.update("jax_enable_x64", True)

# Physical constants (match _uniaxial_reflect.py)
hc = 12398.4193  # eV·Å
_FWHM = 2 * np.sqrt(2 * np.log(2.0))
_INTLIMIT = 3.5
TINY = 1e-30

# ---------------------------------------------------------------------------
# Per-layer building blocks (operate on a single layer's epsilon)
# ---------------------------------------------------------------------------

def _jcalculate_kz_uni(epsilon, kx, ky, k0):
    """
    Analytic kz eigenvalues for uniaxial medium.

    Parameters
    ----------
    epsilon : (3, 3) complex128  — dielectric tensor of one layer
    kx, ky  : (nq,) float64
    k0      : scalar float64

    Returns
    -------
    kz_out : (nq, 4) complex128
        Columns: [+extraord, -extraord, +ord, -ord]
    """
    e_o = epsilon[0, 0]
    e_e = epsilon[2, 2]
    nu = (e_e - e_o) / e_o

    k_par2 = kx**2 + ky**2

    kz_ord = jnp.sqrt(e_o * k0**2 - k_par2 + 0j)

    # na=1, la=0 for optic axis along z (as in numpy version)
    kz_extra = (1.0 / (1.0 + nu)) * jnp.sqrt(
        e_o * k0**2 * (1.0 + nu) ** 2 - k_par2 * (1.0 + nu) + 0j
    )

    return jnp.stack([kz_extra, -kz_extra, kz_ord, -kz_ord], axis=-1)  # (nq, 4)


def _jcalculate_Dpol_uni(epsilon, kx, ky, kz, k0):
    """
    Displacement (D) and magnetic (H) polarization eigenvectors.

    Parameters
    ----------
    epsilon : (3, 3) complex128
    kx, ky  : (nq,) float64
    kz      : (nq, 4) complex128
    k0      : scalar

    Returns
    -------
    Dpol : (nq, 4, 3) complex128
    Hpol : (nq, 4, 3) complex128
    """
    opticaxis = jnp.array([0.0, 0.0, 1.0])

    e_o = epsilon[0, 0]
    e_e = epsilon[2, 2]
    nu = (e_e - e_o) / e_o

    nq = kx.shape[0]

    # k-vector: (nq, 4, 3)
    kvec = jnp.stack([
        jnp.broadcast_to(kx[:, None], (nq, 4)),
        jnp.broadcast_to(ky[:, None], (nq, 4)),
        kz,
    ], axis=-1)  # (nq, 4, 3)

    # Normalise k-vector (divide by its magnitude)
    kdiv = jnp.sqrt(jnp.einsum('ijk,ijk->ij', kvec, kvec))  # (nq, 4)
    knorm = kvec / kdiv[:, :, None]  # (nq, 4, 3)

    # Projection of knorm onto optic axis
    kpol = jnp.einsum('ijk,k->ij', knorm, opticaxis)  # (nq, 4)

    # Ordinary modes (indices 2, 3): D ∝ opticaxis × knorm
    dpol_ord_2 = jnp.cross(opticaxis[None, :], knorm[:, 2, :])  # (nq, 3)
    dpol_ord_3 = jnp.cross(opticaxis[None, :], knorm[:, 3, :])  # (nq, 3)

    # Extraordinary modes (indices 0, 1)
    factor_0 = (1.0 + nu) / (1.0 + nu * kpol[:, 0, None] ** 2) * kpol[:, 0, None]
    factor_1 = (1.0 + nu) / (1.0 + nu * kpol[:, 1, None] ** 2) * kpol[:, 1, None]
    dpol_extra_0 = opticaxis[None, :] - factor_0 * knorm[:, 0, :]  # (nq, 3)
    dpol_extra_1 = opticaxis[None, :] - factor_1 * knorm[:, 1, :]  # (nq, 3)

    dpol_temp = jnp.stack([dpol_extra_0, dpol_extra_1, dpol_ord_2, dpol_ord_3], axis=1)  # (nq,4,3)

    # Normalise D vectors
    dpol_norm = jnp.linalg.norm(dpol_temp, axis=-1, keepdims=True)  # (nq,4,1)
    dpol_temp = dpol_temp / dpol_norm

    # H = (k × D) / k0
    hpol_temp = jnp.cross(kvec, dpol_temp) / k0  # (nq, 4, 3)

    return dpol_temp, hpol_temp


def _jcalculate_D_single(Dpol, Hpol):
    """
    Assemble 4×4 dynamic matrix for a single (nq,4,3) eigenvector set.

    Returns D (nq,4,4) and Di = inv(D) (nq,4,4).
    """
    # Row 0: Dpol x-components
    # Row 1: Hpol y-components
    # Row 2: Dpol y-components
    # Row 3: Hpol x-components
    D = jnp.stack([
        jnp.stack([Dpol[:, 0, 0], Dpol[:, 1, 0], Dpol[:, 2, 0], Dpol[:, 3, 0]], axis=-1),
        jnp.stack([Hpol[:, 0, 1], Hpol[:, 1, 1], Hpol[:, 2, 1], Hpol[:, 3, 1]], axis=-1),
        jnp.stack([Dpol[:, 0, 1], Dpol[:, 1, 1], Dpol[:, 2, 1], Dpol[:, 3, 1]], axis=-1),
        jnp.stack([Hpol[:, 0, 0], Hpol[:, 1, 0], Hpol[:, 2, 0], Hpol[:, 3, 0]], axis=-1),
    ], axis=1)  # (nq, 4, 4)

    Di = jnp.linalg.inv(D)  # (nq, 4, 4) — JAX supports batched inversion
    return D, Di


def _jcalculate_P_single(kz, d):
    """
    Diagonal propagation matrix for one layer.

    kz : (nq, 4) complex128
    d  : scalar float — thickness in Å

    Returns P : (nq, 4, 4) complex128 diagonal
    """
    diag = jnp.exp(-1j * kz * d)  # (nq, 4)
    return jnp.einsum('ij,jk->ijk', diag, jnp.eye(4, dtype=jnp.complex128))  # (nq,4,4)


def _jcalculate_W_single(kz_curr, kz_prev, roughness):
    """
    Nevot–Croche roughness matrix for one interface.

    kz_curr, kz_prev : (nq, 4) complex128
    roughness        : scalar float — interface roughness in Å

    Returns W : (nq, 4, 4) complex128
    """
    r2 = roughness ** 2
    # eminus[i, j] = exp(-(kz_curr[:,i] - kz_prev[:,j])^2 * r^2 / 2)  → only diagonal used per numpy code
    # The numpy code uses a specific pattern matching indices 0,1,2,3 of the same layer rolled
    eplus  = jnp.exp(-(kz_curr + kz_prev) ** 2 * r2 / 2.0)  # (nq, 4)
    eminus = jnp.exp(-(kz_curr - kz_prev) ** 2 * r2 / 2.0)  # (nq, 4)

    # Assemble W matching the pattern in _uniaxial_reflect.py calculate_W:
    #   W[i,j] pattern with eminus on diagonal-like positions, eplus on off-diagonal
    W = jnp.stack([
        jnp.stack([eminus[:, 0], eplus[:, 1],  eminus[:, 2], eplus[:, 3]],  axis=-1),
        jnp.stack([eplus[:, 0],  eminus[:, 1], eplus[:, 2],  eminus[:, 3]], axis=-1),
        jnp.stack([eminus[:, 0], eplus[:, 1],  eminus[:, 2], eplus[:, 3]],  axis=-1),
        jnp.stack([eplus[:, 0],  eminus[:, 1], eplus[:, 2],  eminus[:, 3]], axis=-1),
    ], axis=1)  # (nq, 4, 4)
    return W


def _jcalculate_output(M):
    """
    Extract reflectance coefficients from 4×4 system transfer matrix.

    M : (nq, 4, 4) complex128

    Returns
    -------
    refl : (nq, 2, 2) float64  — |r|² for [ss, sp; ps, pp]
    tran : (nq, 2, 2) complex128
    """
    denom = M[:, 0, 0] * M[:, 2, 2] - M[:, 0, 2] * M[:, 2, 0]

    r_ss = (M[:, 1, 0] * M[:, 2, 2] - M[:, 1, 2] * M[:, 2, 0]) / denom
    r_sp = (M[:, 3, 0] * M[:, 2, 2] - M[:, 3, 2] * M[:, 2, 0]) / denom
    r_ps = (M[:, 0, 0] * M[:, 1, 2] - M[:, 1, 0] * M[:, 0, 2]) / denom
    r_pp = (M[:, 0, 0] * M[:, 3, 2] - M[:, 3, 0] * M[:, 0, 2]) / denom

    t_ss =  M[:, 2, 2] / denom
    t_sp = -M[:, 2, 0] / denom
    t_ps = -M[:, 0, 2] / denom
    t_pp =  M[:, 0, 0] / denom

    refl = jnp.stack([
        jnp.stack([jnp.abs(r_ss)**2, jnp.abs(r_sp)**2], axis=-1),
        jnp.stack([jnp.abs(r_ps)**2, jnp.abs(r_pp)**2], axis=-1),
    ], axis=1)  # (nq, 2, 2)

    tran = jnp.stack([
        jnp.stack([t_ss, t_sp], axis=-1),
        jnp.stack([t_ps, t_pp], axis=-1),
    ], axis=1)  # (nq, 2, 2)

    return jnp.real(refl), tran


# ---------------------------------------------------------------------------
# Core JAX reflectivity function
# ---------------------------------------------------------------------------

def juniaxial_reflectivity_scan(q, slabs, tensor, energy):
    """
    4×4 TMM polarized reflectivity using jax.lax.scan over layers.

    Parameters
    ----------
    q      : (nq,) float64  — momentum transfer in Å⁻¹
    slabs  : (nlayers, 4) float64  — [thickness, SLD_trace, iSLD_trace, roughness]
    tensor : (nlayers, 3, 3) complex128  — susceptibility tensor χ per layer
             (same convention as PXR_Structure.tensor(): ε = I - 2χ conjugated)
    energy : scalar float64  — photon energy in eV

    Returns
    -------
    refl : (nq, 2, 2) float64   — reflectance [[ss,sp],[ps,pp]]
    tran : (nq, 2, 2) complex128 — transmittance coefficients
    """
    wl = hc / energy
    k0 = 2.0 * jnp.pi / wl

    flatq = q.ravel().astype(jnp.float64)
    nq = flatq.shape[0]
    nlayers = slabs.shape[0]

    theta = jnp.pi / 2.0 - jnp.arcsin(jnp.clip(flatq / (2.0 * k0), -1.0, 1.0))
    kx = k0 * jnp.sin(theta)
    ky = jnp.zeros(nq, dtype=jnp.float64)

    # Convert susceptibility tensor → dielectric tensor: ε = conj(I - 2χ)
    eps = jnp.conj(jnp.eye(3, dtype=jnp.complex128)[None, :, :] - 2.0 * tensor.astype(jnp.complex128))
    # eps shape: (nlayers, 3, 3)

    # --- Precompute kz, D, Di for all layers via vmap ---
    # kz_all : (nlayers, nq, 4)
    kz_all = jax.vmap(lambda ep: _jcalculate_kz_uni(ep, kx, ky, k0))(eps)

    # Dpol_all, Hpol_all : (nlayers, nq, 4, 3)
    Dpol_all, Hpol_all = jax.vmap(
        lambda ep, kz: _jcalculate_Dpol_uni(ep, kx, ky, kz, k0)
    )(eps, kz_all)

    # D_all, Di_all : (nlayers, nq, 4, 4)
    D_all, Di_all = jax.vmap(_jcalculate_D_single)(Dpol_all, Hpol_all)

    # P_all : (nlayers, nq, 4, 4) — propagation matrices
    P_all = jax.vmap(_jcalculate_P_single)(kz_all, slabs[:, 0])

    # W_all : (nlayers, nq, 4, 4) — roughness matrices
    # kz_prev[j] = kz_all[j-1]; for j=0 we roll (fronting uses its own kz rolled)
    kz_prev = jnp.roll(kz_all, 1, axis=0)
    W_all = jax.vmap(_jcalculate_W_single)(kz_all, kz_prev, slabs[:, 3])

    # --- TMM product via lax.scan over inner layers ---
    # Starting M is identity: (nq, 4, 4)
    M0 = jnp.broadcast_to(jnp.eye(4, dtype=jnp.complex128), (nq, 4, 4)).copy()

    def scan_step(M, xs):
        Di_prev, D_j, W_j, P_j = xs
        # Interface matrix: (Di_{j-1} @ D_j) ⊙ W_j @ P_j
        A = jnp.einsum('...ij,...jk->...ik', Di_prev, D_j) * W_j
        C = jnp.einsum('...ij,...jk->...ik', A, P_j)
        return jnp.einsum('...ij,...jk->...ik', M, C), None

    # Inner layers: indices 1 … nlayers-2 (fronting=0, backing=nlayers-1)
    xs_inner = (Di_all[:-2], D_all[1:-1], W_all[1:-1], P_all[1:-1])
    M, _ = jax.lax.scan(scan_step, M0, xs_inner)

    # Final interface to backing (no propagation phase)
    last = jnp.einsum('...ij,...jk->...ik', Di_all[-2], D_all[-1]) * W_all[-1]
    M = jnp.einsum('...ij,...jk->...ik', M, last)

    return _jcalculate_output(M)


# JIT-compiled entry points
uniaxial_reflectivity_jax = jit(juniaxial_reflectivity_scan)


# ---------------------------------------------------------------------------
# Float32 fast path — ~4× speedup for CMA-ES ranking (~0.5% chi2 error)
# ---------------------------------------------------------------------------

def _juniaxial_reflectivity_f32_inner(q, slabs, tensor, energy):
    """Internal float32 variant — same logic but cast to complex64."""
    wl = jnp.float32(hc) / jnp.float32(energy)
    k0 = jnp.float32(2.0 * np.pi) / wl

    flatq = q.ravel().astype(jnp.float32)
    nq = flatq.shape[0]
    nlayers = slabs.shape[0]

    theta = jnp.float32(jnp.pi / 2.0) - jnp.arcsin(
        jnp.clip(flatq / (2.0 * k0), jnp.float32(-1.0), jnp.float32(1.0))
    )
    kx = k0 * jnp.sin(theta)
    ky = jnp.zeros(nq, dtype=jnp.float32)

    eps = jnp.conj(
        jnp.eye(3, dtype=jnp.complex64)[None] - jnp.float32(2.0) * tensor.astype(jnp.complex64)
    )

    def kz_uni_f32(epsilon, kx_, ky_, k0_):
        e_o = epsilon[0, 0]
        e_e = epsilon[2, 2]
        nu = (e_e - e_o) / e_o
        k_par2 = kx_**2 + ky_**2
        kz_ord = jnp.sqrt(e_o * k0_**2 - k_par2 + jnp.complex64(0j))
        kz_ex  = (jnp.float32(1.0) / (jnp.float32(1.0) + nu)) * jnp.sqrt(
            e_o * k0_**2 * (jnp.float32(1.0) + nu)**2 - k_par2 * (jnp.float32(1.0) + nu) + jnp.complex64(0j)
        )
        return jnp.stack([kz_ex, -kz_ex, kz_ord, -kz_ord], axis=-1)

    def dpol_f32(epsilon, kx_, ky_, kz, k0_):
        oa = jnp.array([0.0, 0.0, 1.0], dtype=jnp.float32)
        e_o = epsilon[0, 0]; e_e = epsilon[2, 2]
        nu = (e_e - e_o) / e_o
        nq_ = kx_.shape[0]
        kvec = jnp.stack([
            jnp.broadcast_to(kx_[:, None], (nq_, 4)),
            jnp.broadcast_to(ky_[:, None], (nq_, 4)),
            kz,
        ], axis=-1)
        kdiv = jnp.sqrt(jnp.einsum('ijk,ijk->ij', kvec, kvec))
        knorm = kvec / kdiv[:, :, None]
        kpol = jnp.einsum('ijk,k->ij', knorm, oa)
        dp2 = jnp.cross(oa[None, :], knorm[:, 2, :])
        dp3 = jnp.cross(oa[None, :], knorm[:, 3, :])
        f0 = (jnp.float32(1.0) + nu) / (jnp.float32(1.0) + nu * kpol[:, 0, None]**2) * kpol[:, 0, None]
        f1 = (jnp.float32(1.0) + nu) / (jnp.float32(1.0) + nu * kpol[:, 1, None]**2) * kpol[:, 1, None]
        dp0 = oa[None, :] - f0 * knorm[:, 0, :]
        dp1 = oa[None, :] - f1 * knorm[:, 1, :]
        dpol = jnp.stack([dp0, dp1, dp2, dp3], axis=1)
        dpol = dpol / jnp.linalg.norm(dpol, axis=-1, keepdims=True)
        hpol = jnp.cross(kvec, dpol) / k0_
        return dpol, hpol

    def D_f32(Dpol, Hpol):
        D = jnp.stack([
            jnp.stack([Dpol[:, 0, 0], Dpol[:, 1, 0], Dpol[:, 2, 0], Dpol[:, 3, 0]], axis=-1),
            jnp.stack([Hpol[:, 0, 1], Hpol[:, 1, 1], Hpol[:, 2, 1], Hpol[:, 3, 1]], axis=-1),
            jnp.stack([Dpol[:, 0, 1], Dpol[:, 1, 1], Dpol[:, 2, 1], Dpol[:, 3, 1]], axis=-1),
            jnp.stack([Hpol[:, 0, 0], Hpol[:, 1, 0], Hpol[:, 2, 0], Hpol[:, 3, 0]], axis=-1),
        ], axis=1)
        return D, jnp.linalg.inv(D)

    kz_all = jax.vmap(lambda ep: kz_uni_f32(ep, kx, ky, k0))(eps)
    Dp_all, Hp_all = jax.vmap(lambda ep, kz: dpol_f32(ep, kx, ky, kz, k0))(eps, kz_all)
    D_all, Di_all = jax.vmap(D_f32)(Dp_all, Hp_all)

    def P_f32(kz, d):
        diag = jnp.exp(-jnp.complex64(1j) * kz * d.astype(jnp.float32))
        return jnp.einsum('ij,jk->ijk', diag, jnp.eye(4, dtype=jnp.complex64))

    def W_f32(kz_c, kz_p, r):
        r2 = r.astype(jnp.float32) ** 2
        ep = jnp.exp(-(kz_c + kz_p)**2 * r2 / 2.0)
        em = jnp.exp(-(kz_c - kz_p)**2 * r2 / 2.0)
        return jnp.stack([
            jnp.stack([em[:, 0], ep[:, 1], em[:, 2], ep[:, 3]], axis=-1),
            jnp.stack([ep[:, 0], em[:, 1], ep[:, 2], em[:, 3]], axis=-1),
            jnp.stack([em[:, 0], ep[:, 1], em[:, 2], ep[:, 3]], axis=-1),
            jnp.stack([ep[:, 0], em[:, 1], ep[:, 2], em[:, 3]], axis=-1),
        ], axis=1)

    P_all = jax.vmap(P_f32)(kz_all, slabs[:, 0])
    kz_prev = jnp.roll(kz_all, 1, axis=0)
    W_all = jax.vmap(W_f32)(kz_all, kz_prev, slabs[:, 3])

    M0 = jnp.broadcast_to(jnp.eye(4, dtype=jnp.complex64), (nq, 4, 4)).copy()

    def scan_step(M, xs):
        Di_prev, D_j, W_j, P_j = xs
        A = jnp.einsum('...ij,...jk->...ik', Di_prev, D_j) * W_j
        C = jnp.einsum('...ij,...jk->...ik', A, P_j)
        return jnp.einsum('...ij,...jk->...ik', M, C), None

    M, _ = jax.lax.scan(scan_step, M0, (Di_all[:-2], D_all[1:-1], W_all[1:-1], P_all[1:-1]))
    last = jnp.einsum('...ij,...jk->...ik', Di_all[-2], D_all[-1]) * W_all[-1]
    M = jnp.einsum('...ij,...jk->...ik', M, last)

    denom = M[:, 0, 0] * M[:, 2, 2] - M[:, 0, 2] * M[:, 2, 0]
    r_ss = (M[:, 1, 0] * M[:, 2, 2] - M[:, 1, 2] * M[:, 2, 0]) / denom
    r_pp = (M[:, 0, 0] * M[:, 3, 2] - M[:, 3, 0] * M[:, 0, 2]) / denom
    r_sp = (M[:, 3, 0] * M[:, 2, 2] - M[:, 3, 2] * M[:, 2, 0]) / denom
    r_ps = (M[:, 0, 0] * M[:, 1, 2] - M[:, 1, 0] * M[:, 0, 2]) / denom

    refl = jnp.stack([
        jnp.stack([jnp.abs(r_ss)**2, jnp.abs(r_sp)**2], axis=-1),
        jnp.stack([jnp.abs(r_ps)**2, jnp.abs(r_pp)**2], axis=-1),
    ], axis=1)
    return jnp.real(refl).astype(jnp.float32)


uniaxial_reflectivity_jax_f32 = jit(_juniaxial_reflectivity_f32_inner)


# ---------------------------------------------------------------------------
# Gauss-Legendre smearing (replaces scipy-spline approach)
# ---------------------------------------------------------------------------

def _gauss_legendre(n):
    """Return (abscissa, weights) for n-point Gauss-Legendre on [-1, 1]."""
    return np.polynomial.legendre.leggauss(n)


def juniaxial_reflectivity_smeared(q, slabs, tensor, energy, pol='ss', dq_frac=0.0, quad_order=17):
    """
    Resolution-smeared polarized reflectivity using Gauss-Legendre quadrature.

    Parameters
    ----------
    q        : (nq,) float64  — momentum transfer in Å⁻¹
    slabs    : (nlayers, 4) float64
    tensor   : (nlayers, 3, 3) complex128
    energy   : float
    pol      : 'ss', 'pp', 'sp', or 'ps'  — output polarization
    dq_frac  : float — dQ/Q resolution as a fraction (e.g. 0.02 for 2%)
    quad_order : int — number of quadrature points

    Returns
    -------
    smeared_r : (nq,) float64
    """
    pol_map = {'ss': (0, 0), 'sp': (0, 1), 'ps': (1, 0), 'pp': (1, 1)}
    pi, pj = pol_map[pol]

    if dq_frac < 0.005:
        refl, _ = uniaxial_reflectivity_jax(
            jnp.array(q), jnp.array(slabs), jnp.array(tensor), energy
        )
        return np.array(refl[:, pi, pj])

    abscissa, weights = _gauss_legendre(quad_order)
    prefactor = 1.0 / np.sqrt(2 * np.pi)

    def gauss(x):
        return np.exp(-0.5 * x * x)

    gaussvals = prefactor * gauss(abscissa * _INTLIMIT)

    dqvals = dq_frac * q
    va = (q - _INTLIMIT * dqvals / _FWHM)[:, None]
    vb = (q + _INTLIMIT * dqvals / _FWHM)[:, None]

    q_grid = (abscissa[None, :] * (vb - va) + vb + va) / 2.0  # (nq, quad_order)
    q_flat = jnp.array(q_grid.ravel())

    refl_flat, _ = uniaxial_reflectivity_jax(q_flat, jnp.array(slabs), jnp.array(tensor), energy)
    r_flat = np.array(refl_flat[:, pi, pj]).reshape(len(q), quad_order)

    smeared = np.sum(r_flat * gaussvals[None, :] * weights[None, :], axis=1) * _INTLIMIT
    return smeared
