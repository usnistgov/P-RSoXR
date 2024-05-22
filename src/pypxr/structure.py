"""
The refnx code is distributed under the following license:

Copyright (c) 2015 A. R. J. Nelson, ANSTO

Permission to use and redistribute the source code or binary forms of this
software and its documentation, with or without modification is hereby
granted provided that the above notice of copyright, these terms of use,
and the disclaimer of warranty below appear in the source code and
documentation, and that none of the names of above institutions or
authors appear in advertising or endorsement of works derived from this
software without specific prior written permission from all parties.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL
THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
DEALINGS IN THIS SOFTWARE.

"""

import numbers
import operator
from collections import UserList

import numpy as np
from refnx.analysis import Parameter, Parameters, possibly_create_parameter
from refnx.reflect.interface import Erf, Step
from scipy.interpolate import interp1d
from scipy.stats import norm

try:  # Check for pip install
    from pypxr.reflectivity import PXR_reflectivity
except ImportError:
    from reflectivity import PXR_reflectivity
# from pypxr.reflectivity import PXR_reflectivity

speed_of_light = 299792458  # m/s
plank_constant = 4.135667697e-15  # ev*s
hc = (speed_of_light * plank_constant) * 1e10  # ev*A

tensor_index = ["xx", "yy", "zz"]  # Indexing for later definitions

"""
Class structure is closely related to the foundations built by refnx.
It was designed so prior knowledge of one software will interface with the other.
The prefix 'PXR' (Resonant X-ray Reflectivity) will designate the objects required for working with
polarized resonant soft X-ray reflectivity data. See class PXR_SLD for information on open tensor parameters.

Only Gaussian roughness is supported at this time.
"""


class PXR_Structure(UserList):
    r"""
    Represents the interfacial Structure of a reflectometry sample.
    Successive Components are added to the Structure to construct the interface.

    Parameters
    ----------

    components : sequence
        A sequence of PXR_Components to initialise the PXR_Structure.
    name : str
        Name of this structure
    reverse_structure : bool
        If `Structure.reverse_structure` is `True` then  slab representation produced by `Structure.slabs` is reversed.

    Example
    -------
    >>> from PyPXR import PXR_SLD, PXR_MaterialSLD
    >>> en = 284.4 #[eV]
    >>> # make the material with tensor index of refraction
    >>> vac = PXR_MaterialSLD('', density=1, energy=en, name='vacuum') #Superstrate
    >>> si = PXR_MaterialSLD('Si', density=2.33, energy=en, name='Si') #Substrate
    >>> sio2 = PXR_MaterialSLD('SiO2', density=2.4, energy=en, name='SiO2') #Substrate
    >>> n_xx = complex(-0.0035, 0.0004) # [unitless] #Ordinary Axis
    >>> n_zz = complex(-0.0045, 0.0009) # [unitless] #Extraordinary Axis
    >>> molecule = PXR_SLD(np.array([n_xx, n_zz]), name='material') #molecule
    >>> #Make the structure
    >>> #See 'PXR_Slab' for details on building layers
    >>> structure = vac(0,0) | molecule(100, 2) | sio2(15, 1.5) | si(1, 1.5)

    """

    def __init__(
        self, components=(), name="", reverse_structure=False
    ):  # Removed solvent parameter
        super(PXR_Structure, self).__init__()
        self._name = name

        self._reverse_structure = bool(reverse_structure)

        # if you provide a list of components to start with, then initialise
        # the structure from that
        self.data = [c for c in components if isinstance(c, Component)]

    def __copy__(self):
        s = PXR_Structure(name=self.name)
        s.data = self.data.copy()
        return s

    def __setitem__(self, i, v):
        self.data[i] = v

    def __str__(self):
        s = list()
        s.append("{:_>80}".format(""))
        s.append("Structure: {0: ^15}".format(str(self.name)))
        s.append("reverse structure: {0}".format(str(self.reverse_structure)))

        for component in self:
            s.append(str(component))

        return "\n".join(s)

    def __repr__(self):
        return (
            "Structure(components={data!r},"
            " name={_name!r},"
            " reverse_structure={_reverse_structure},".format(**self.__dict__)
        )

    def append(self, item):
        """
        Append a :class:`PXR_Component` to the Structure.

        Parameters
        ----------
        item: refnx.reflect.Component
            The component to be added.
        """
        if isinstance(item, PXR_Scatterer):
            self.append(item())
            return

        if not isinstance(item, PXR_Component):
            raise ValueError("You can only add PXR_Component objects to a" " structure")
        super(PXR_Structure, self).append(item)

    @property
    def name(self):
        return self._name

    @name.setter
    def name(self, name):
        self._name = name

    @property
    def reverse_structure(self):
        """
        **bool**  if `True` then the slab representation produced by
        :meth:`PXR_Structure.slabs` is reversed. The sld profile and calculated
        reflectivity will correspond to this reversed structure.
        """
        return bool(self._reverse_structure)

    @reverse_structure.setter
    def reverse_structure(self, reverse_structure):
        self._reverse_structure = reverse_structure

    def slabs(self):
        r"""

        Returns
        -------
        slabs : :class:`np.ndarray`
            Slab representation of this structure.
            Has shape (N, 3).
            N - number of slabs

            - slab[N, 0]
               thickness of layer N
            - slab[N, 1]
               sld.delta of layer N
            - slab[N, 2]
               sld.beta of layer N
            - slab[N, 3]
               roughness between layer N and N-1

        Notes
        -----
        If `PXR_Structure.reversed is True` then the slab representation order is
        reversed.
        """
        if not len(self):
            return None

        if not (
            isinstance(self.data[-1], PXR_Slab) and isinstance(self.data[0], PXR_Slab)
        ):
            raise ValueError(
                "The first and last PXR_Components in a PXR_Structure"
                " need to be PXR_slabs"
            )
        # PRSoXR only supports Gaussian interfaces as of 07/2021
        # Potentially be added in the future, please contact developer if interested.

        sl = [
            c.slabs(structure=self) for c in self.components
        ]  # concatenate PXR_Slab objects
        try:
            slabs = np.concatenate(sl)
        except ValueError:
            # some of slabs may be None. np can't concatenate arr and None
            slabs = np.concatenate([s for s in sl if s is not None])

        # if the slab representation needs to be reversed.
        reverse = self.reverse_structure
        if reverse:
            roughnesses = slabs[1:, 3]
            slabs = np.flipud(slabs)
            slabs[1:, 3] = roughnesses[::-1]
            slabs[0, 3] = 0.0

        return slabs

    def tensor(self, energy=None):
        """

        Parameters:
        -----------
        energy: float
            Photon energy used to calculate the tensor index of refraction.
            This only applies for objects that require a specific energy (see PXR_MaterialSLD).
            Common for substrates/superstrates

        Returns
        -------
        tensors : :class:`np.ndarray`
            Supplementary object to self.slabs that contains dielectric tensor for each layer.
            Has shape (N, 3,3).
            N - number of slabs

            - tensor[N, 1, 1]
               dielectric component xx of layer N
            - tensor[N, 2, 2]
               dielectric component yy of layer N
            - tensor[N, 3, 3]
               dielectric component zz of layer N

        Notes
        -----
        Output as a (3, 3) np.ndarray.
        Used for broadcasting in later calculations. All off-diagonal elements are zero.

        If `Structure.reversed is True` then the representation order is
        reversed. Energy is required for energy-dependent slabs

        """
        d1 = [c.tensor(energy=energy) for c in self.components]
        try:
            _tensor = np.concatenate(d1, axis=0)
        except ValueError:
            # some of slabs may be None. np can't concatenate arr and None
            _tensor = np.concatenate([s for s in d1 if s is not None], axis=0)

        reverse = self.reverse_structure
        if reverse:
            _tensor = np.flip(_tensor, axis=0)
        return _tensor

    def reflectivity(self, q, energy=250.0, backend="uni"):
        """
        Calculate theoretical polarized reflectivity of this structure

        Parameters
        ----------
        q : array-like
            Q values (Angstrom**-1) for evaluation
        energy : float
            Photon energy (eV) for evaluation
        backend : 'uni' or 'biaxial'
            Specifies if you want to run a uniaxial calculation or a full biaxial calculation.
            Biaxial has NOT been verified through outside means (07/2021 Biaxial currently does not work)

        """

        refl, tran, *components = PXR_reflectivity(
            q, self.slabs(), self.tensor(energy=energy), backend=backend
        )
        return refl[:, 1, 1], refl[:, 0, 0], components

    def sld_profile(self, z=None, align=0):
        """
        Calculates an index of refraction depth profile as a function of distance from the superstrate.

        Parameters
        -----------
        z : float
            Interfacial distance (Angstrom) measured from interface between the fronting medium and first layer.
        align : int, optional
            Places a specified interface in the slab representation of a PXR_Structure at z =0.
            Python indexing is allowed to select interface.

        Returns
        -------

        zed : np.ndarray
            Interfacial distance measured from superstrate offset by 'align'.
            Has shape (N, )
        prof : np.ndarray (complex)
            Real and imaginary tensor components of index of refraction [unitless]
            Has shape (N, 3)

            -prof[N, 0]
                dielectric component n_xx at depth N
            -prof[N, 1]
                dielectric component n_yy at depth N
            -prof[N, 3]
                dielectric component n_xx at depth N

        Notes
        -----
        >>> #To calculate the isotropic components
        >>> n_iso = prof.sum(axis=1)/3 #(nxx + nyy + nzz)/3
        >>> #To calculate the birefringence and dichroism
        >>> diff = prof[:,0] - prof[:,2] #nxx-nzz

        """
        slabs = self.slabs()
        tensor = self.tensor()
        if (
            (slabs is None)
            or (len(slabs) < 2)
            or (not isinstance(self.data[0], PXR_Slab))
            or (not isinstance(self.data[-1], PXR_Slab))
        ):
            raise ValueError(
                "Structure requires fronting and backing"
                " Slabs in order to calculate."
            )

        zed, prof = birefringence_profile(slabs, tensor, z)

        offset = 0
        if align != 0:
            align = int(align)
            if align >= len(slabs) - 1 or align < -1 * len(slabs):
                raise RuntimeError("abs(align) has to be less than " "len(slabs) - 1")
            # to figure out the offset you need to know the cumulative distance
            # to the interface
            slabs[0, 0] = slabs[-1, 0] = 0.0  # Set the thickness of each end to zero
            if align >= 0:
                offset = np.sum(slabs[: align + 1, 0])
            else:
                offset = np.sum(slabs[:align, 0])
        return zed - offset, prof

    def __ior__(self, other):
        """
        Build a structure by `IOR`'ing Structures/Components/SLDs.

        Parameters
        ----------
        other: :class:`PXR_Structure`, :class:`PXR_Component`, :class:`PXR_SLD`
            The object to add to the structure.

        Examples
        --------

        >>> air = SLD(0, name='air')
        >>> sio2 = SLD(3.47, name='SiO2')
        >>> si = SLD(2.07, name='Si')
        >>> structure = air | sio2(20, 3)
        >>> structure |= si(0, 4)

        """
        # self |= other
        if isinstance(other, PXR_Component):
            self.append(other)
        elif isinstance(other, PXR_Structure):
            self.extend(other.data)
        elif isinstance(other, PXR_Scatterer):
            slab = other(0, 0)
            self.append(slab)
        else:
            raise ValueError()

        return self

    def __or__(self, other):
        """
        Build a structure by `OR`'ing Structures/Components/SLDs.

        Parameters
        ----------
        other: :class:`PXR_Structure`, :class:`PXR_Component`, :class:`PXR_SLD`
            The object to add to the structure.

        Examples
        --------

        >>> vac = PXR_MaterialSLD('', density=1, energy=en, name='vacuum') #Superstrate
        >>> sio2 = PXR_MaterialSLD('SiO2', density=2.4, energy=en, name='SiO2') #Substrate
        >>> si = PXR_MaterialSLD('Si', density=2.33, energy=en, name='Si') #Substrate
        >>> structure = vac | sio2(10,5) | si(0, 1.5)

        """
        # c = self | other
        p = PXR_Structure()
        p |= self
        p |= other
        return p

    @property
    def components(self):
        """
        The list of components in the sample.
        """
        return self.data

    @property
    def parameters(self):
        r"""
        :class:`refnx.analysis.Parameters`, all the parameters associated with
        this structure.

        """
        p = Parameters(name="Structure - {0}".format(self.name))
        p.extend([component.parameters for component in self.components])
        return p

    def logp(self):
        """
        log-probability for the interfacial structure. Note that if a given
        component is present more than once in a Structure then it's log-prob
        will be counted twice.

        Returns
        -------
        logp : float
            log-prior for the Structure.
        """
        logp = 0
        for component in self.components:
            logp += component.logp()

        return logp

    def plot(self, pvals=None, samples=0, fig=None, difference=False, align=0):
        """
        Plot the structure.

        Requires matplotlib be installed.

        Parameters
        ----------
        pvals : np.ndarray, optional
            Numeric values for the Parameter's that are varying
        samples: number
            If this structures constituent parameters have been sampled, how
            many samples you wish to plot on the graph.
        fig: Figure instance, optional
            If `fig` is not supplied then a new figure is created. Otherwise
            the graph is created on the current axes on the supplied figure.
        difference: boolean, optional
            If True, plot the birefringence / dichroism on a separate graph.
        align: int, optional
            Aligns the plotted structures around a specified interface in the
            slab representation of a Structure. This interface will appear at
            z = 0 in the sld plot.

        Returns
        -------
        fig, ax : :class:`matplotlib.Figure`, :class:`matplotlib.Axes`
          `matplotlib` figure and axes objects.

        """
        import matplotlib.pyplot as plt

        params = self.parameters

        if pvals is not None:
            params.pvals = pvals

        if fig is None:
            fig = plt.figure()
            ax = fig.add_subplot(111)
        else:
            ax = fig.gca()

        if samples > 0:
            saved_params = np.array(params)
            # Get a number of chains, chosen randomly, and plot the model.
            for pvec in self.parameters.pgen(ngen=samples):
                params.pvals = pvec

                temp_zed, temp_prof = self.sld_profile(align=align)
                temp_iso = temp_prof.sum(axis=1) / 3  # (nxx + nyy + nzz)/3
                ax.plot(temp_zed, temp_iso, color="k", alpha=0.01)

            # put back saved_params
            params.pvals = saved_params

        # parameters to plot
        zed, prof = self.sld_profile(align=align)
        iso = prof.sum(axis=1) / 3
        ax.plot(zed, np.real(iso), color="red", zorder=20, label="delta")
        ax.plot(
            zed,
            np.real(prof[:, 0]),
            color="orange",
            zorder=10,
            label="dxx",
            linestyle="dashed",
        )
        ax.plot(
            zed,
            np.real(prof[:, 2]),
            color="orange",
            zorder=10,
            label="dzz",
            linestyle="dashed",
        )
        ax.plot(zed, np.imag(iso), color="blue", zorder=20, label="beta")
        ax.plot(
            zed,
            np.imag(prof[:, 0]),
            color="teal",
            zorder=10,
            label="bxx",
            linestyle="dashed",
        )
        ax.plot(
            zed,
            np.imag(prof[:, 2]),
            color="teal",
            zorder=10,
            label="bzz",
            linestyle="dashed",
        )
        # ax.plot(*self.sld_profile(align=align), color='red', zorder=20)
        ax.set_ylabel("Index of refraction")
        ax.set_xlabel("zed / $\\AA$")
        plt.legend()

        if difference:
            fig_diff = plt.figure()
            ax_diff = fig_diff.add_subplot(111)

            diff = prof[:, 0] - prof[:, 2]
            ax_diff.plot(
                zed, np.real(diff), color="red", zorder=20, label="birefringence"
            )
            ax_diff.plot(zed, np.imag(diff), color="blue", zorder=20, label="dichroism")
            plt.legend()

        return fig, ax


class PXR_Scatterer(object):
    """
    Abstract base class for a material with a complex tensor index of refraction
    """

    def __init__(self, name=""):
        self.name = name

    def __str__(self):
        sld = 1 - complex(self)  # Returns optical constant
        return "n = {0}".format(sld)

    def __complex__(self):
        raise NotImplementedError

    @property
    def parameters(self):
        raise NotImplementedError

    def __call__(self, thick=0, rough=0):
        """
        Create a :class:`PXR_Slab`.

        Parameters
        ----------
        thick: refnx.analysis.Parameter or float
            Thickness of slab in Angstrom
        rough: refnx.analysis.Parameter or float
            Roughness of slab in Angstrom

        Returns
        -------
        slab : refnx.PXR_reflect.PXR_Slab
            The newly made Slab with a dielectric tensor.

        Example
        --------
        >>> n_xx = complex(-0.0035, 0.0004) # [unitless] #Ordinary Axis
        >>> n_zz = complex(-0.0045, 0.0009) # [unitless] #Extraordinary Axis
        >>> molecule = PXR_SLD(np.array([n_xx, n_zz]), name='material') #molecule
        >>> #Crete a slab with 10 A in thickness and 3 A roughness
        >>> slab = molecule(10, 3)

        """
        return PXR_Slab(thick, self, rough, name=self.name)

    def __or__(self, other):
        # c = self | other
        slab = self()
        return slab | other


class PXR_SLD(PXR_Scatterer):
    """
    Object representing freely varying complex tensor index of refraction of a material

    Parameters
    ----------
    value : float, complex, 'np.array'
        Valid np.ndarray.shape: (2,), (3,), (3,3) ('xx', 'yy', 'zz')
        tensor index of refraction.
        Units (N/A)
    symmetry : ('iso', 'uni', 'bi')
        Tensor symmetry. Automatically applies inter-parameter constraints.
    name : str, optional
        Name of object for later reference.

    Notes
    -----
    Components correspond to individual tensor components defined as ('xx', 'yy', 'zz').
    In a uniaxial approximation the following inputs are equivalent.

    >>> n_xx = complex(-0.0035, 0.0004) # [unitless] #Ordinary Axis
    >>> n_zz = complex(-0.0045, 0.0009) # [unitless] #Extraordinary Axis
    >>> molecule = PXR_SLD(np.array([n_xx, n_zz]), name='molecule')
    >>> molecule = PXR_SLD(np.array([n_xx, n_xx, n_zz], name='molecule')
    >>> molecule = PXR_SLD(np.array([n_xx, n_xx, n_zz])*np.eye(3), name='molecule)

    An PXR_SLD object can be used to create a PXR_Slab:

    >>> n_xx = complex(-0.0035, 0.0004) # [unitless] #Ordinary Axis
    >>> n_zz = complex(-0.0045, 0.0009) # [unitless] #Extraordinary Axis
    >>> molecule = PXR_SLD(np.array([n_xx, n_zz]), name='material') #molecule
    >>> #Crete a slab with 10 A in thickness and 3 A roughness
    >>> slab = molecule(10, 3)

    Tensor symmetry can be applied using `symmetry`.

    >>> #'uni' will constrain n_xx = n_yy.
    >>> self.yy.setp(self.xx, vary=None, constraint=self.xx)
    >>> self.iyy.setp(self.ixx, vary=None, constraint=self.ixx)

    >>> #'iso' will constrain n_xx = n_yy = n_zz
    >>> self.yy.setp(self.xx, vary=None, constraint=self.xx)
    >>> self.iyy.setp(self.ixx, vary=None, constraint=self.ixx)
    >>> self.zz.setp(self.xx, vary=None, constraint=self.xx)
    >>> self.izz.setp(self.ixx, vary=None, constraint=self.ixx)
    """

    def __init__(self, value, symmetry="uni", name="", en_offset=0):
        super(PXR_SLD, self).__init__(name=name)
        self.imag = Parameter(0, name="%s_isld" % name)
        self._tensor = None

        # Figure out if the input is valid
        if isinstance(
            value, np.ndarray
        ):  # Make sure the input is an array with 3 elements
            if value.shape == (
                3,
            ):  # 3 element array, assume structure ['xx', 'yy', 'zz']
                pass
                # Great choice
            elif value.shape == (
                2,
            ):  # 2 element array, assume structure ['xx', 'zz'] (uniaxial)
                temp_val = (
                    np.ones(3) * value[0]
                )  # Make a 3-element array and fill it with 'xx'
                temp_val[2] = value[1]  # Append the last element as 'zz'
                value = temp_val  # Reset value
            elif value.shape == (
                3,
                3,
            ):  # 3x3 element array, assume diagonal is ['xx', 'yy', 'zz']
                value = value.diagonal()  # Just take the inner 3 elements for generating the index of refraction
        elif isinstance(
            value, (int, float, complex)
        ):  # If the value is a scalar, convert it into an array for later use.
            value = value * np.ones(3)
        else:
            # No input was given
            print("Please input valid index of refraction")
            print("Suggested format: np.ndarray shape: (3, )")

        # Build parameters from given tensor
        self._parameters = Parameters(
            name=name
        )  # Generate the parameters for the tensor object
        self.delta = Parameter(
            np.average(value).real, name="%s_diso" % name
        )  # create parameter for the 'isotropic' version of the given delta
        self.beta = Parameter(
            np.average(value).imag, name="%s_biso" % name
        )  # create parameter for the 'isotropic' version of the given beta
        # Create parameters for individual tensor components.
        # Each element of the tensor becomes its own fit parameter within the PXR machinary
        # All tensors are assumed diagonal in the substrate frame
        # See documentation for recommended parameter constraints
        self.xx = Parameter(value[0].real, name="%s_%s" % (name, tensor_index[0]))
        self.ixx = Parameter(value[0].imag, name="%s_i%s" % (name, tensor_index[0]))
        self.yy = Parameter(value[1].real, name="%s_%s" % (name, tensor_index[1]))
        self.iyy = Parameter(value[1].imag, name="%s_i%s" % (name, tensor_index[1]))
        self.zz = Parameter(value[2].real, name="%s_%s" % (name, tensor_index[2]))
        self.izz = Parameter(value[2].imag, name="%s_i%s" % (name, tensor_index[2]))

        self.birefringence = Parameter(
            (self.xx.value - self.zz.value), name="%s_bire" % name
        )  # Useful parameters to use as constraints
        self.dichroism = Parameter(
            (self.ixx.value - self.izz.value), name="%s_dichro" % name
        )  # Defined in terms of xx and zz

        self.en_offset = Parameter((en_offset), name="%s_enOffset" % name)

        self._parameters.extend(
            [
                self.delta,
                self.beta,
                self.en_offset,
                self.xx,
                self.ixx,
                self.yy,
                self.iyy,
                self.zz,
                self.izz,
                self.birefringence,
                self.dichroism,
            ]
        )

        self.symmetry = symmetry

    def __repr__(self):
        return (
            "Isotropic Index of Refraction = ([{delta!r}, {beta!r}],"
            " name={name!r})".format(**self.__dict__)
        )

    def __complex__(self):
        sldc = complex(self.delta.value, self.beta.value)
        return sldc

    @property
    def parameters(self):
        """
        :class:`refnx.analysis.Parameters` associated with this component

        """
        self._parameters.name = self.name
        return self._parameters
        # p = Parameters(name=self.name)
        # p.extend([self.real, self.imag])
        # return p

    @property
    def symmetry(self):
        """
        Specify `symmetry` to automatically constrain the components. Default is 'uni'
        """
        return self._symmetry

    @symmetry.setter
    def symmetry(self, symmetry):
        self._symmetry = symmetry
        if self._symmetry == "iso":
            self.yy.setp(self.xx, vary=None, constraint=self.xx)
            self.iyy.setp(self.ixx, vary=None, constraint=self.ixx)
            self.zz.setp(self.xx, vary=None, constraint=self.xx)
            self.izz.setp(self.ixx, vary=None, constraint=self.ixx)
        elif self._symmetry == "uni":
            self.yy.setp(self.xx, vary=None, constraint=self.xx)
            self.iyy.setp(self.ixx, vary=None, constraint=self.ixx)
        elif self._symmetry == "bi":
            self.xx.setp(self.xx, vary=None, constraint=None)
            self.ixx.setp(self.ixx, vary=None, constraint=None)
            self.yy.setp(self.yy, vary=None, constraint=None)
            self.iyy.setp(self.iyy, vary=None, constraint=None)
            self.zz.setp(self.zz, vary=None, constraint=None)
            self.izz.setp(self.izz, vary=None, constraint=None)

    @property
    def tensor(self):  #
        """
        A full 3x3 matrix composed of the individual parameter values.

        Returns
        -------
            out : np.ndarray (3x3)
                complex tensor index of refraction
        """
        self._tensor = np.array(
            [
                [self.xx.value + 1j * self.ixx.value, 0, 0],
                [0, self.yy.value + 1j * self.iyy.value, 0],
                [0, 0, self.zz.value + 1j * self.izz.value],
            ],
            dtype=complex,
        )
        return self._tensor


class PXR_MaterialSLD(PXR_Scatterer):
    """
    Object representing complex index of refraction of a chemical formula.
    Only works for an isotropic material, convenient for substrate and superstrate materials
    You can fit the mass density of the material.
    Takes advantage of the PeriodicTable python package for calculations

    Parameters
    ----------
    formula : str
        Chemical formula
    density : float or Parameter
        mass density of compound in g / cm**3
    energy : float, optional
        energy of radiation (ev) ~ Converted to Angstrom in function
    name : str, optional
        Name of material

    Notes
    -----
    You need to have the `periodictable` package installed to use this object.
    A PXR_MaterialSLD object can be used to create a PXR_Slab:

    >>> # A PXR_MaterialSLD object for a common substrate
    >>> en = 284.4 #[eV] Evaluate PeriodicTable at this energy
    >>> sio2 = PXR_MaterialSLD('SiO2', density=2.4, energy=en, name='SiO2') #Substrate
    >>> si = PXR_MaterialSLD('Si', density=2.33, energy=en, name='SiO2') #Substrate

    """

    def __init__(self, formula, density, energy=250.0, name=""):
        import periodictable as pt

        super(PXR_MaterialSLD, self).__init__(name=name)

        self.__formula = pt.formula(
            formula
        )  # Build the PeriodicTable object for storage
        self._compound = formula  # Keep a reference of the str object
        self.density = possibly_create_parameter(density, name="rho")

        self._energy = energy  # Store in eV for user interface
        self._wavelength = (
            hc / self._energy
        )  # Convert to Angstroms for later calculations
        self._tensor = None  # Build this when its called based in parameter values

        self._parameters = Parameters(name=name)
        self._parameters.extend([self.density])

    def __repr__(self):
        d = {
            "compound": self._compound,
            "density": self.density,
            "energy": self.energy,
            "wavelength": self.wavelength,
            "name": self.name,
        }
        return (
            "MaterialSLD({compound!r}, {density!r},"
            "energy={energy!r}, wavelength={wavelength!r}, name={name!r})".format(**d)
        )

    @property
    def formula(self):
        """
        Chemical formula used to calculate the index of refraction.

        Returns
        -------
            formula : str
                Full chemical formula used to calculate index of refraction.

        """
        return self._compound

    @formula.setter
    def formula(self, formula):
        import periodictable as pt

        self.__formula = pt.formula(formula)
        self._compound = formula

    @property
    def energy(self):
        """
        Photon energy to evaluate index of refraction in eV. Automatically updates wavelength when assigned.

        Returns
        -------
            energy : float
                Photon energy of X-ray probe.
        """
        return self._energy

    @energy.setter
    def energy(self, energy):
        self._energy = energy
        self._wavelength = (
            hc / self._energy
        )  # Update the wavelength if the energy changes

    @property
    def wavelength(self):
        """
        Wavelength to evaluate index of refraction in Angstroms. Automatically updates energy when assigned.

        Returns
        -------
            wavelength : float
                Wavelength of X-ray probe.
        """
        return self._wavelength

    @wavelength.setter
    def wavelength(self, wavelength):
        self._wavelength = wavelength
        self._energy = (
            hc / self._wavelength
        )  # Update the energy if the wavelength changes

    def __complex__(self):
        import periodictable as pt
        from periodictable import xsf

        sldc = pt.xsf.index_of_refraction(
            self.__formula, density=self.density.value, wavelength=self.wavelength
        )
        if (
            type(sldc).__module__ == np.__name__
        ):  # check if the type is accidentally cast into numpy.
            sldc = sldc.item()
        return 1 - sldc  # pt.xsf makes the type numpy affiliated...
        # __complex__ does not play nice so we reconvert with .item()

    @property
    def parameters(self):
        """
        :class:`refnx.analysis.Parameters` associated with this component

        """
        return self._parameters

    @property
    def tensor(self):
        """
        An isotropic 3x3 tensor composed of `complex(self.delta, self.beta)` along the diagonal.

        Returns
        -------
            tensor : np.ndarray
                complex tensor index of refraction
        """
        self._tensor = np.eye(3) * complex(self)
        return self._tensor


class PXR_Component(object):
    """
    A base class for describing the structure of a subset of an interface.

    Parameters
    ----------
    name : str, optional
        The name associated with the Component

    Notes
    -----
    Currently limited to Gaussian interfaces.
    """

    def __init__(self, name=""):
        self.name = name

    def __or__(self, other):
        """
        OR'ing components can create a :class:`Structure`.

        Parameters
        ----------
        other: refnx.reflect.Structure, refnx.reflect.Component
            Combines with this component to make a Structure

        Returns
        -------
        s: refnx.reflect.Structure
            The created Structure

        Examples
        --------

        >>> air = SLD(0, name='air')
        >>> sio2 = SLD(3.47, name='SiO2')
        >>> si = SLD(2.07, name='Si')
        >>> structure = air | sio2(20, 3) | si(0, 3)

        """
        # c = self | other
        p = PXR_Structure()
        p |= self
        p |= other
        return p

    def __mul__(self, n):
        """
        MUL'ing components makes them repeat.

        Parameters
        ----------
        n: int
            How many times you want to repeat the Component

        Returns
        -------
        s: refnx.reflect.Structure
            The created Structure
        """
        # convert to integer, should raise an error if there's a problem
        n = operator.index(n)
        if n < 1:
            return PXR_Structure()
        elif n == 1:
            return self
        else:
            s = PXR_Structure()
            s.extend([self] * n)
            return s

    def __str__(self):
        return str(self.parameters)

    @property
    def parameters(self):
        """
        :class:`refnx.analysis.Parameters` associated with this component
        """
        raise NotImplementedError(
            "A component should override the parameters " "property"
        )

    def slabs(self, structure=None):
        """
        The slab representation of this component

        Parameters
        ----------
        structure : PyPXR.anisotropic_reflect.PXR_Structure
            Summary of the structure that houses the component.

        Returns
        -------
        slabs : np.ndarray
            Slab representation of this Component.
            Has shape (N, 5).

            - slab[N, 0]
               thickness of layer N
            - slab[N, 1]
               SLD.real of layer N (not including solvent)
            - slab[N, 2]
               *overall* SLD.imag of layer N (not including solvent)
            - slab[N, 3]
               roughness between layer N and N-1
            - slab[N, 4]
               volume fraction of solvent in layer N.

        If a Component returns None, then it doesn't have any slabs.
        """

        raise NotImplementedError("A component should override the slabs " "property")

    def logp(self):
        """
        The log-probability that this Component adds to the total log-prior
        term. Do not include log-probability terms for the actual parameters,
        these are automatically included elsewhere.

        Returns
        -------
        logp : float
            Log-probability
        """
        return 0


class PXR_Slab(PXR_Component):
    """
    A slab component has uniform tensor index of refraction associated over its thickness.

    Parameters
    ----------
    thick : refnx.analysis.Parameter or float
        thickness of slab (Angstrom)
    sld : :class:`PyPXR.anisotropic_structure.PXR_Scatterer`, complex, or float
        (complex) tensor index of refraction of film
    rough : refnx.analysis.Parameter or float
        roughness on top of this slab (Angstrom)
    name : str
        Name of this slab
    """

    def __init__(self, thick, sld, rough, name=""):
        super(PXR_Slab, self).__init__(name=name)
        self.thick = possibly_create_parameter(thick, name=f"{name}_thick")
        if isinstance(sld, PXR_Scatterer):
            self.sld = sld
        else:
            self.sld = PXR_SLD(sld)

        self.rough = possibly_create_parameter(rough, name=f"{name}_rough")

        p = Parameters(name=self.name)
        p.extend([self.thick])
        p.extend(self.sld.parameters)
        p.extend([self.rough])

        self._parameters = p

    def __repr__(self):
        return (
            f"Slab({self.thick!r}, {self.sld!r}, {self.rough!r},"
            f" name={self.name!r},"
        )

    def __str__(self):
        # sld = repr(self.sld)
        #
        # s = 'Slab: {0}\n    thick = {1} Å, {2}, rough = {3} Å,
        #      \u03D5_solv = {4}'
        # t = s.format(self.name, self.thick.value, sld, self.rough.value,
        #              self.vfsolv.value)
        return str(self.parameters)

    @property
    def parameters(self):
        """
        :class:`refnx.analysis.Parameters` associated with this component

        """
        self._parameters.name = self.name
        return self._parameters

    def slabs(self, structure=None):
        """
        Slab representation of this component. See :class:`Component.slabs`
        """
        sldc = complex(self.sld)
        return np.array([[self.thick.value, sldc.real, sldc.imag, self.rough.value]])

    def tensor(self, energy=None):
        """
        Stored information pertaining to the tensor dielectric properties of the slab.

        Parameters
        -----------
        energy : float
            Updates PXR_SLD energy component associated with slab. Only required for PXR_MaterialSLD objects

        Returns
        --------
        tensor : np.ndarray
            Complex tensor index of refraction associated with slab.
        """
        if energy is not None and hasattr(self.sld, "energy"):
            self.sld.energy = energy
        return np.array([self.sld.tensor])


class PXR_MixedMaterialSlab(PXR_Component):
    """
    A slab component made of several components

    Parameters
    ----------
    thick : refnx.analysis.Parameter or float
        thickness of slab (Angstrom)
    sld_list : sequence of {anisotropic_reflect.PXR_Scatterer, complex, float}
        Sequence of materials that are contained in the slab.
    vf_list : sequence of refnx.analysis.Parameter or float
        relative volume fractions of each of the materials contained in the
        film.
    rough : refnx.analysis.Parameter or float
        roughness on top of this slab (Angstrom)
    name : str
        Name of this slab

    Notes
    -----
    The index of refraction for this slab is calculated using the normalised volume fractions of
    each of the constituent components:

    >>> np.sum([complex(sld) * vf / np.sum(vf_list) for sld, vf in
    ...         zip(sld_list, vf_list)]).

    """

    def __init__(
        self,
        thick,
        sld_list,
        vf_list,
        rough,
        name="",
    ):
        super(PXR_MixedMaterialSlab, self).__init__(name=name)

        self.thick = possibly_create_parameter(thick, name="%s - thick" % name)
        self.sld = []
        self.vf = []
        self._sld_parameters = Parameters(name=f"{name} - slds")
        self._vf_parameters = Parameters(name=f"{name} - volfracs")

        i = 0
        for s, v in zip(sld_list, vf_list):
            if isinstance(s, PXR_Scatterer):
                self.sld.append(s)
            else:
                self.sld.append(PXR_SLD(s))

            self._sld_parameters.append(self.sld[-1].parameters)

            vf = possibly_create_parameter(v, name=f"vf{i} - {name}", bounds=(0.0, 1.0))
            self.vf.append(vf)
            self._vf_parameters.append(vf)
            i += 1

        self.rough = possibly_create_parameter(rough, name=f"{name} - rough")

        p = Parameters(name=self.name)
        p.append(self.thick)
        p.extend(self._sld_parameters)
        p.extend(self._vf_parameters)
        p.extend([self.rough])

        self._parameters = p

    def __repr__(self):
        return (
            f"PXR_MixedMaterialSlab({self.thick!r}, {self.sld!r}, {self.vf!r},"
            f" {self.rough!r}, name={self.name!r},"
        )

    def __str__(self):
        return str(self.parameters)

    @property
    def parameters(self):
        """
        :class:`refnx.analysis.Parameters` associated with this component

        """
        self._parameters.name = self.name
        return self._parameters

    def slabs(self, structure=None):
        """
        Slab representation of this component. See :class:`Component.slabs`
        """
        vfs = np.array(self._vf_parameters)
        sum_vfs = np.sum(vfs)

        sldc = np.sum([complex(sld) * vf / sum_vfs for sld, vf in zip(self.sld, vfs)])

        return np.array(
            [
                [
                    self.thick.value,
                    sldc.real,
                    sldc.imag,
                    self.rough.value,
                ]
            ]
        )

    def tensor(self, energy=None):
        """
        Stored information pertaining to the tensor dielectric properties of the slab.

        Parameters
        -----------
        energy : float
            Updates PXR_SLD energy component associated with slab. Only required for PXR_MaterialSLD objects

        Returns
        --------
        tensor : np.ndarray
            Complex tensor index of refraction associated with slab.
        """
        vfs = np.array(self._vf_parameters)
        sum_vfs = np.sum(vfs)

        if energy is not None and hasattr(self.sld, "energy"):
            self.sld.energy = energy

        combinetensor = np.sum(
            [sld.tensor * vf / sum_vfs for sld, vf in zip(self.sld, vfs)], axis=0
        )

        return combinetensor  # self.sld.tensor


class PXR_Stack(PXR_Component, UserList):
    r"""
    A series of PXR_Components that are considered as a single item. When
    incorporated into a PXR_Structure the PXR_Stack will be repeated as a multilayer

    Parameters
    ------------
    components : sequence
        A series of PXR_Components to repeat in a structure
    name: str
        Human readable name for the stack
    repeats: number, Parameter
        Number of times to repeat the stack within a structure to make a multilayer

    """

    def __init__(self, components=(), name="", repeats=1):
        PXR_Component.__init__(self, name=name)
        UserList.__init__(self)

        self.repeats = possibly_create_parameter(repeats, "repeat")
        self.repeats.bounds.lb = 1

        # Construct the list of components
        for c in components:
            if isinstance(c, PXR_Component):
                self.data.append(c)
            else:
                raise ValueError(
                    "You can only initialise a PXR_Stack with PXR_Components"
                )

    def __setitem__(self, i, v):
        self.data[i] = vac

    def __str__(self):
        s = list()
        s.append("{:=>80}".format(""))

        s.append(f"Stack start: {int(round(abs(self.repeats.value)))} repeats")
        for component in self:
            s.append(str(component))
        s.append("Stack finish")
        s.append("{:=>80}".format(""))

        return "/n".join(s)

    def __repr__(self):
        return (
            "Stack(name={name!r},"
            " components={data!r},"
            " repeats={repeats!r}".format(**self.__dict__)
        )

    def append(self, item):
        """
        Append a PXR_Component to the Stack.

        Parameters
        -----------
        item: PXR_Compponent
            PXR_Component to be added to the PXR_Stack

        """

        if isinstance(item, PXR_Scatterer):
            self.append(item())
            return

        if not isinstance(item, PXR_Component):
            raise ValueError("You can only add PXR_Components")
        self.data.append(item)

    def slabs(self, structure=None):
        """
        Slab representation of this component.

        Notes
        -----
        Returns a list of each slab included within this Stack.

        """
        if not len(self):
            return None

        repeats = int(round(abs(self.repeats.value)))

        slabs = np.concatenate([c.slabs(structure=self) for c in self.components])

        if repeats > 1:
            slabs = np.concatenate([slabs] * repeats)

        if hasattr(self, "solvent"):
            delattr(self, "solvent")

        return slabs

    def tensor(self, energy=None):
        """
        Tensor representation of this component. Builds list of all components
        """

        if not len(self):
            return None

        repeats = int(round(abs(self.repeats.value)))

        tensor = np.concatenate(
            [c.tensor(energy=energy) for c in self.components], axis=0
        )

        if repeats > 1:
            tensor = np.concatenate([tensor] * repeats)

        return tensor

    @property
    def components(self):
        """
        List of components
        """
        return self.data

    @property
    def parameters(self):
        r"""
        All Parameters associated with this Stack

        """

        p = Parameters(name="Stack - {0}".format(self.name))
        p.append(self.repeats)
        p.extend([component.parameters for component in self.components])
        return p


def birefringence_profile(slabs, tensor, z=None, step=False):
    """
    Calculates a series of depth profiles corresponding to the slab model used to calculated p-RSoXR

    Parameters
    ----------
    slabs : Information regarding the layer stack, see PXR_Structure class
    tensor : List of dielectric tensors from each layer stack, see PXR_Structure class
    z : float
        Interfacial distance (Angstrom) measured from interface between the
        fronting medium and the first layer.
    step : Boolean
        Set 'True' for slab model without interfacial widths


    Returns
    -------
    zed : float / np.ndarray
        Depth into the film / Angstrom

    index_tensor : complex / np.ndarray
        Real and imaginary tensor components of index of refraction / unitless
        Array elements: [nxx, nyy, nzz]

    Optional:

    index_step : complex / np.ndarray
        Real and imaginary tensor components of index of refraction / unitless
        Calculated WITHOUT interfacial roughness

    Notes
    -----
    This can be called in vectorised fashion.

    To calculate the isotropic components:
        index_iso = index_tensor.sum(axis=1)/3 #(nxx + nyy + nzz)/3
    To calculate the birefringence/dichroism:
        diff = index_tensor[:,0] - index_tensor[:,2] #nxx - nzz

    """
    nlayers = (
        np.size(slabs, 0) - 2
    )  # Calculate total number of layers (not including fronting/backing)

    # work on a copy of the input array
    layers = np.copy(slabs)
    layers[:, 0] = np.fabs(slabs[:, 0])  # Ensure the thickness is positive
    layers[:, 3] = np.fabs(slabs[:, 3])  # Ensure the roughness is positive
    # bounding layers should have zero thickness
    layers[0, 0] = layers[-1, 0] = 0

    # distance of each interface from the fronting interface
    dist = np.cumsum(layers[:-1, 0])
    total_film_thickness = int(
        np.round(dist[-1])
    )  # Total film thickness for point density
    # workout how much space the SLD profile should encompass
    # (if z array not provided)
    if z is None:
        zstart = -5 - 4 * np.fabs(slabs[1, 3])
        zend = 5 + dist[-1] + 4 * layers[-1, 3]
        zed = np.linspace(
            zstart, zend, num=total_film_thickness * 2
        )  # 0.5 Angstrom resolution default
    else:
        zed = np.asfarray(z)

    # Reduce the dimensionality of the tensor for ease of use
    reduced_tensor = tensor.diagonal(
        0, 1, 2
    )  # 0 - no offset, 1 - first axis of the tensor, 2 - second axis of the tensor

    tensor_erf = (
        np.ones((len(zed), 3), dtype=float) * reduced_tensor[0]
    )  # Full wave of initial conditions
    tensor_step = np.copy(tensor_erf)  # Full wave without interfacial roughness
    delta_n = reduced_tensor[1:] - reduced_tensor[:-1]  # Change in n at each interface

    # use erf for roughness function, but step if the roughness is zero
    step_f = Step()  # Step function (see refnx documentation)
    erf_f = Erf()  # Error function (see refnx documentation)
    sigma = layers[1:, 3]  # Interfacial width parameter

    # accumulate the SLD of each step.
    for i in range(nlayers + 1):
        f = erf_f
        g = step_f
        if sigma[i] == 0:
            f = step_f
        tensor_erf += (
            delta_n[None, i, :] * f(zed, scale=sigma[i], loc=dist[i])[:, None]
        )  # Broadcast into a single item
        tensor_step += (
            delta_n[None, i, :] * g(zed, scale=0, loc=dist[i])[:, None]
        )  # Broadcast into a single item

    return zed, tensor_erf if step is False else tensor_step
