"""
refnx is distributed under the following license:

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

from collections import UserList
import numbers
import operator

import numpy as np
from scipy.stats import norm
from scipy.interpolate import interp1d

from refnx._lib import flatten
from refnx.analysis import Parameters, Parameter, possibly_create_parameter
from refnx.reflect.interface import Interface, Erf, Step
from ani_reflect.ani_reflect_model import ani_reflectivity

speed_of_light = 299792458 #m/s
plank_constant = 4.135667697e-15 #ev*s
hc = (speed_of_light * plank_constant)*1e10# ev*A

tensor_index = ['xx', 'yy', 'zz']

"""
Class structure is closely related to the foundations built by refnx. It was designed so prior knowledge of one software will interface with the other.
The prefix 'RXR' (Resonant X-ray Reflectivity) will designate the objects required for working with polarized resonant soft X-ray reflectivity data. See class RXR_SLD for information on open tensor parameters.

Only Gaussian roughness is supported at this time.
"""



class RXR_Structure(UserList):
    """
    Represents the interfacial Structure of a reflectometry sample.
    Successive Components are added to the Structure to construct the
    interface.


    Parameters
    ----------
    components : sequence
        A sequence of Components to initialise the Structure.
    name : str
        Name of this structure
    reverse_structure : bool
        If `Structure.reverse_structure` is `True` then the slab
        representation produced by `Structure.slabs` is reversed. The sld
        profile and calculated reflectivity will correspond to this
        reversed structure.

    Notes
    -----
    If `Structure.reverse_structure is True` then the slab representation
    order is reversed.
    

    Example
    -------
    
    >>> from PRSoXR import RXR_SLD, RXR_MaterialSLD
    >>> # make the material with tensor index of refraction
    >>> vac = RXR_MaterialSLD('', density=1, name='vac')
    >>> polymer = RXR_SLD(np.array([complex(0.001, 0.002), complex(0.001, 0.002), complex(0.002, 0.001)]), name='Polymer')
    >>> si = RXR_MaterialSLD('Si', density=2.33, name='Si')
    >>> #Make the structure
    >>> #See 'RXR_Slab' for details on building layers
    >>> structure = vac(0,0) | polymer(100, 2) | si(0, 1.5)

    """
    def __init__(self, components=(), name='', reverse_structure=False): #Removed solvent parameter
        super(RXR_Structure, self).__init__()
        self._name = name
        
        self._reverse_structure = bool(reverse_structure)

        # if you provide a list of components to start with, then initialise
        # the structure from that
        self.data = [c for c in components if isinstance(c, Component)]
        
    def __copy__(self):
        s = RXR_Structure(name=self.name)
        s.data = self.data.copy()
        return s

    def __setitem__(self, i, v):
        self.data[i] = v

    def __str__(self):
        s = list()
        s.append('{:_>80}'.format(''))
        s.append('Structure: {0: ^15}'.format(str(self.name)))
        s.append('reverse structure: {0}'.format(str(self.reverse_structure)))

        for component in self:
            s.append(str(component))

        return '\n'.join(s)

    def __repr__(self):
        return ("Structure(components={data!r},"
                " name={_name!r},"
                " reverse_structure={_reverse_structure},".format(**self.__dict__))

    def append(self, item):
        """
        Append a :class:`RXR_Component` to the Structure.

        Parameters
        ----------
        item: refnx.reflect.Component
            The component to be added.
        """
        if isinstance(item, RXR_Scatterer):
            self.append(item())
            return

        if not isinstance(item, RXR_Component):
            raise ValueError("You can only add RXR_Component objects to a"
                             " structure")
        super(RXR_Structure, self).append(item)

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
        :meth:`Structure.slabs` is reversed. The sld profile and calculated
        reflectivity will correspond to this reversed structure.
        """
        return bool(self._reverse_structure)

    @reverse_structure.setter
    def reverse_structure(self, reverse_structure):
        self._reverse_structure = reverse_structure

    def slabs(self, **kwds):
        r"""

        Returns
        -------
        slabs : :class:`np.ndarray`
            Slab representation of this structure.
            Has shape (N, 4).

            - slab[N, 0]
               thickness of layer N
            - slab[N, 1]
               SLD.real of layer N
            - slab[N, 2]
               SLD.imag of layer N
            - slab[N, 3]
               roughness between layer N and N-1

        Notes
        -----
        If `RXR_Structure.reversed is True` then the slab representation order is
        reversed. The slab order is reversed before the solvation calculation
        is done. I.e. if `RXR_Structure.solvent == 'backing'` and
        `RXR_Structure.reversed is True` then the material that solvates the system
        is the component in `Structure[0]`, which corresponds to
        `RXR_Structure.slab[-1]`.

        """
        if not len(self):
            return None

        if not (isinstance(self.data[-1], RXR_Slab) and
                isinstance(self.data[0], RXR_Slab)):
            raise ValueError("The first and last RXR_Components in a RXR_Structure"
                             " need to be RXR_slabs")
        #PRSoXR only supports Gaussian interfaces as of 07/2021
        #Potentially be added in the future, please contact developer if interested.

        sl = [c.slabs(structure=self) for c in self.components] #concatenate PXR_Slab objects
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
            slabs[0, 3] = 0.

        return slabs
    
    def tensor(self, energy=None):
        """
        Parameters:
        -------
        energy: float
            Photon energy used to calculate the tensor index of refraction.
            This only applies for objects that require a specific energy (see PXR_MaterialSLD).
            Common for substrates/superstrates

        Returns
        -------
        tensors : :class:`np.ndarray`
            Complimentary representation to Slabs that contains dielectric tensor components for each layer.
            Has shape (N, 3,3).
            N - number of slabs

            - tensors[N, 1, 1]
               dielectric component xx of layer N
            - tensors[N, 2, 2]
               dielectric component yy of layer N
            - tensors[N, 3, 3]
               dielectric component zz of layer N
        Notes
        -----
        If `Structure.reversed is True` then the representation order is
        reversed. Has not functionality with an added solvent. May not be
        desireable since n(E) is difficult to calculate at resonance and 
        a database currently does not exist.
        Energy is required for energy-dependent slabs

        """
        d1 = [c.tensor(energy=energy) for c in self.components]
        try:
            _tensor = np.stack(d1,axis=0)
        except ValueError:
            # some of slabs may be None. np can't concatenate arr and None
            _tensor = np.stack([s for s in d1 if s is not None], axis=0)
            
        reverse = self.reverse_structure 
        if reverse:
            _tensor = np.flip(_tensor,axis=0)   
        return _tensor
        
    def reflectivity(self, q, energy=250.0, backend='uni'):
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
            Biaxial has NOT been verified through outside means
        threads : int, optional
            Specifies the number of threads for parallel calculation. This
            option is only applicable if you are using the ``_creflect``
            module. The option is ignored if using the pure python calculator,
            ``_reflect``. If `threads == 0` then all available processors are
            used.
        output : 's', 'p', or 'sp'
            's' - returns s-polarization
            'p' - returns p-polarization
            'sp' - returns a concatenated wave of s- and p-polarization.

        """        
        refl, tran, *components = ani_reflectivity(q, self.slabs(), self.tensor(energy=energy), backend=backend)
        #Organize output for what you want:
        
        return refl[:,1,1], refl[:,0,0], components
        
        
    def sld_profile(self, z=None, align=0):
        """
        Calculates an SLD profile, as a function of distance through the
        interface.

        Parameters
        ----------
        z : float
            Interfacial distance (Angstrom) measured from interface between the
            fronting medium and the first layer.
        align: int, optional
            Places a specified interface in the slab representation of a
            Structure at z = 0. Python indexing is allowed, e.g. supplying -1
            will place the backing medium at z = 0.

        Returns
        -------
        sld : float
            Scattering length density / 1e-6 Angstrom**-2

        Notes
        -----
        This can be called in vectorised fashion.
        """
        slabs = self.slabs()
        tensor = self.tensor()
        if ((slabs is None) or
                (len(slabs) < 2) or
                (not isinstance(self.data[0], RXR_Slab)) or
                (not isinstance(self.data[-1], RXR_Slab))):
            raise ValueError("Structure requires fronting and backing"
                             " Slabs in order to calculate.")

        zed, prof  = birefringence_profile(slabs, tensor, z)

        offset = 0
        if align != 0:
            align = int(align)
            if align >= len(slabs) - 1 or align < -1 * len(slabs):
                raise RuntimeError('abs(align) has to be less than '
                                   'len(slabs) - 1')
            # to figure out the offset you need to know the cumulative distance
            # to the interface
            slabs[0, 0] = slabs[-1, 0] = 0. #Set the thickness of each end to zero
            if align >= 0:
                offset = np.sum(slabs[:align + 1, 0])
            else:
                offset = np.sum(slabs[:align, 0])
        return zed - offset, prof

    def __ior__(self, other):
        """
        Build a structure by `IOR`'ing Structures/Components/SLDs.

        Parameters
        ----------
        other: :class:`Structure`, :class:`Component`, :class:`SLD`
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
        if isinstance(other, RXR_Component):
            self.append(other)
        elif isinstance(other, RXR_Structure):
            self.extend(other.data)
        elif isinstance(other, RXR_Scatterer):
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
        other: :class:`RXR_Structure`, :class:`RXR_Component`, :class:`RXR_SLD`
            The object to add to the structure.

        Examples
        --------

        >>> air = SLD(0, name='air')
        >>> sio2 = SLD(3.47, name='SiO2')
        >>> si = SLD(2.07, name='Si')
        >>> structure = RXR_Structure()
        >>> structure = air | sio2(20, 3) | si(0, 3)

        """
        # c = self | other
        p = RXR_Structure()
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
        p = Parameters(name='Structure - {0}'.format(self.name))
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
        difference: Boolean, optional
            If True, plot the birefringence / dichroism on a separate graph.
        align: int, optional
            Aligns the plotted structures around a specified interface in the
            slab representation of a Structure. This interface will appear at
            z = 0 in the sld plot. Note that Components can consist of more
            than a single slab, so some thought is required if the interface to
            be aligned around lies in the middle of a Component. Python
            indexing is allowed, e.g. supplying -1 will align at the backing
            medium.

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
                temp_iso = temp_prof.sum(axis=1)/3 #(nxx + nyy + nzz)/3
                ax.plot(temp_zed, temp_iso,
                        color="k", alpha=0.01)

            # put back saved_params
            params.pvals = saved_params

        #parameters to plot
        zed, prof = self.sld_profile(align=align)
        iso = prof.sum(axis=1)/3
        ax.plot(zed, np.real(iso), color='red', zorder=20, label='delta')
        ax.plot(zed, np.real(prof[:,0]), color='orange', zorder=10, label='dxx', linestyle='dashed')
        ax.plot(zed, np.real(prof[:,2]), color='orange', zorder=10, label='dzz', linestyle='dashed')
        ax.plot(zed, np.imag(iso), color='blue', zorder=20, label='beta')
        ax.plot(zed, np.imag(prof[:,0]), color='teal', zorder=10, label='bxx', linestyle='dashed')
        ax.plot(zed, np.imag(prof[:,2]), color='teal', zorder=10, label='bzz', linestyle='dashed')
        #ax.plot(*self.sld_profile(align=align), color='red', zorder=20)
        ax.set_ylabel('Index of refraction')
        ax.set_xlabel("zed / $\\AA$")
        plt.legend()
        
        if difference:
            fig_diff = plt.figure()
            ax_diff = fig_diff.add_subplot(111)
            
            diff = prof[:,0] - prof[:,2]
            ax_diff.plot(zed, np.real(diff), color='red', zorder=20, label='birefringence')
            ax_diff.plot(zed, np.imag(diff), color='blue', zorder=20, label='dichroism')
            plt.legend()
        
        return fig, ax

class RXR_Scatterer(object):
    """
    Abstract base class for something that contains a complex tensor index of refraction
    """
    def __init__(self, name=''):
        self.name = name

    def __str__(self):
        sld = complex(self) #Returns optical constant
        return 'n = {0}'.format(sld)

    def __complex__(self):
        raise NotImplementedError

    @property
    def parameters(self):
        raise NotImplementedError

    def __call__(self, thick=0, rough=0):
        """
        Create a :class:`RXR_Slab`.

        Parameters
        ----------
        thick: refnx.analysis.Parameter or float
            Thickness of slab in Angstrom
        rough: refnx.analysis.Parameter or float
            Roughness of slab in Angstrom

        Returns
        -------
        slab : refnx.RXR_reflect.RXR_Slab
            The newly made Slab with a dielectric tensor.

        Example
        --------

        >>> # an SLD object representing Silicon Dioxide
        >>> sio2 = SLD(3.47, name='SiO2')
        >>> # create a Slab of SiO2 20 A in thickness, with a 3 A roughness
        >>> sio2_layer = sio2(20, 3)

        """
        return RXR_Slab(thick, self, rough, name=self.name)

    def __or__(self, other):
        # c = self | other
        slab = self()
        return slab | other

class RXR_SLD(RXR_Scatterer):
    """
    Object representing freely varying complex tensor index of refraction of a material

    Parameters
    ----------
    value : float, complex, 'np.array'
        Potential Array shapes: (2,), (3,), (3,3) ('xx', 'yy', 'zz')
        tensor index of refraction.
        Units (N/A)
    name : str, optional
        Name of object for later reference.

    Notes
    -----
    An RXR_SLD object can be used to create a RXR_Slab:

    >>> # an SLD object representing Silicon Dioxide
    >>> sio2 = SLD(3.47, name='SiO2')
    >>> # create a Slab of SiO2 20 A in thickness, with a 3 A roughness
    >>> sio2_layer = sio2(20, 3)

    The SLD object can also be made from a complex number, or from Parameters

    >>> sio2 = SLD(3.47+0.01j)
    >>> re = Parameter(3.47)
    >>> im = Parameter(0.01)
    >>> sio2 = SLD(re)
    >>> sio2 = SLD([re, im])
    """
    def __init__(self, value, name=''):
        super(RXR_SLD, self).__init__(name=name)
        self.imag = Parameter(0, name='%s_isld' % name)
        
        #Figure out if the input is valid
        if (isinstance(value, np.ndarray)): #Make sure the input is an array with 3 elements
            if (value.shape == (3, )): #3 element array, assume structure ['xx', 'yy', 'zz']
                pass
                #Great choice
            elif (value.shape == (2, )): #2 element array, assume structure ['xx', 'zz'] (uniaxial)
                temp_val = np.ones(3) * value[0] #Make a 3-element array and fill it with 'xx'
                temp_val[2] = value[1] #Append the last element as 'zz'
                value = temp_val #Reset value
            elif (value.shape == (3, 3)): #3x3 element array, assume diagonal is ['xx', 'yy', 'zz']
                value = value.diagonal() #Just take the inner 3 elements for generating the index of refraction
                
        elif isinstance(value, (int, float, complex)): #If the value is a scalar, convert it into an array for later use.
            value = value * np.ones(3)
        
        else:
            #No input was given
            print("Please input valid index of refraction")
            print("Suggested format: np.ndarray shape: (3, )")
            return 0
        
        #Build parameters from given tensor
        self._parameters = Parameters(name=name) #Generate the parameters for the tensor object
        self.delta = Parameter( np.average(value).real, name='%s_diso' % name ) #create parameter for the 'isotropic' version of the given delta
        self.beta = Parameter ( np.average(value).imag, name='%s_biso' %name ) #create parameter for the 'isotropic' version of the given beta
        #Create parameters for individual tensor components.
        #Each element of the tensor becomes its own fit parameter within the PXRR machinary
        #All tensors are assumed diagonal in the substrate frame
        #See documentation for recommended parameter constraints
        self.xx = Parameter(value[0].real, name='%s_d%s'%(name, tensor_index[0]))
        self.ixx = Parameter(value[0].imag, name='%s_b%s'%(name, tensor_index[0]))
        self.yy = Parameter(value[1].real, name='%s_d%s'%(name, tensor_index[1]))
        self.iyy = Parameter(value[1].imag, name='%s_b%s'%(name, tensor_index[1]))
        self.zz = Parameter(value[2].real, name='%s_d%s'%(name, tensor_index[2]))
        self.izz = Parameter(value[2].imag, name='%s_b%s'%(name, tensor_index[2]))
        
        self.birefringence = Parameter((self.xx.value - self.zz.value), name='%s_bire' %name) #Useful parameters to use as constraints
        self.dichroism = Parameter((self.ixx.value - self.izz.value),name='%s_dichro' % name) #Defined in terms of xx and zz
        
        self._parameters.extend([self.delta,self.beta,self.birefringence,self.dichroism,self.xx,self.ixx,self.yy,self.iyy,self.zz,self.izz])
        
    def __repr__(self):
        return ("Isotropic Index of Refraction = ([{delta!r}, {beta!r}],"
                " name={name!r})".format(**self.__dict__))

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
    def tensor(self, *kwargs): #
        self._tensor = np.array([[self.xx.value + 1j*self.ixx.value, 0, 0],
                                [0, self.yy.value + 1j*self.iyy.value, 0],
                                [0, 0, self.zz.value + 1j*self.izz.value]],dtype=complex)
        return self._tensor

class RXR_MaterialSLD(RXR_Scatterer):
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
    Energy : float, optional
        Energy of radiation (ev) ~ Converted to Angstrom in function
    name : str, optional
        Name of material

    Notes
    -----
    You need to have the `periodictable` package installed to use this object.
    An SLD object can be used to create a Slab:

    >>> # a MaterialSLD object representing Silicon Dioxide
    >>> sio2 = MaterialSLD('SiO2', 2.2, name='SiO2')
    >>> # create a silica slab of SiO2 20 A in thickness, with a 3 A roughness
    >>> sio2_layer = sio2(20, 3)
    >>> # allow the mass density of the silica to vary between 2.1 and 2.3
    >>> # g/cm**3
    >>> sio2.density.setp(vary=True, bounds=(2.1, 2.3))
    """
    def __init__(self, formula, density, energy=250.0, name=''):
        import periodictable as pt        
        super(RXR_MaterialSLD, self).__init__(name=name)

        self.__formula = pt.formula(formula) #Build the PeriodicTable object for storage
        self._compound = formula #Keep a reference of the str object 
        self.density = possibly_create_parameter(density, name='rho')

        self._energy = energy ## Store in eV for user interface
        self._wavelength = hc/self._energy ## Convert to Angstroms for later calculations
        self._tensor = None #Build this when its called based in parameter values

        self._parameters = Parameters(name=name)
        self._parameters.extend([self.density])

    def __repr__(self):
        d = {'compound': self._compound,
             'density': self.density,
             'energy': self.energy,
             'wavelength': self.wavelength,
             'name': self.name}
        return ("MaterialSLD({compound!r}, {density!r},"
                "energy={energy!r}, wavelength={wavelength!r}, name={name!r})".format(**d))

    @property
    def formula(self):
        return self._compound

    @formula.setter
    def formula(self, formula):
        import periodictable as pt
        self.__formula = pt.formula(formula)
        self._compound = formula
        
    @property
    def energy(self):
        return self._energy
    
    @energy.setter
    def energy(self, energy):
        self._energy = energy
        self._wavelength = hc/self._energy  # Update the wavelength if the energy changes
        
    @property
    def wavelength(self):
        return self._wavelength
        
    @wavelength.setter
    def wavelength(self, wavelength):
        self._wavelength = wavelength 
        self._energy = hc/self._wavelength # Update the energy if the wavelength changes
        
    def __complex__(self):
        import periodictable as pt
        from periodictable import xsf
        sldc = pt.xsf.index_of_refraction(self.__formula, density=self.density.value,
                               wavelength=self.wavelength)
        if type(sldc).__module__ == np.__name__: #check if the type is accidently cast into numpy.
            sldc = sldc.item()
        return 1 - sldc ##pt.xsf makes the type numpy affiliated...__complex__ does not play nice so we reconvert with .item()
        
    @property
    def parameters(self):
        return self._parameters
        
    @property
    def tensor(self):
        self._tensor = np.eye(3)*complex(self)#(1 - pt.xsf.index_of_refraction(self.__formula, density=self.density.value, wavelength=self.wavelength))
        return self._tensor

class RXR_Component(object):
    """
    A base class for describing the structure of a subset of an interface.

    Parameters
    ----------
    name : str, optional
        The name associated with the Component

    Notes
    -----
    #Currently limited to Gaussian interfaces.
    """
    def __init__(self, name=''):
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
        p = RXR_Structure()
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
            return RXR_Structure()
        elif n == 1:
            return self
        else:
            s = RXR_Structure()
            s.extend([self] * n)
            return s

    def __str__(self):
        return str(self.parameters)

    @property
    def parameters(self):
        """
        :class:`refnx.analysis.Parameters` associated with this component
        """
        raise NotImplementedError("A component should override the parameters "
                                  "property")

    def slabs(self, structure=None):
        """
        The slab representation of this component

        Parameters
        ----------
        structure : refnx.reflect.Structure
            The Structure hosting the Component.

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

        raise NotImplementedError("A component should override the slabs "
                                  "property")

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

class RXR_Slab(RXR_Component):
    """
    A slab component has uniform tensor index of refraction associated over its thickness.

    Parameters
    ----------
    thick : refnx.analysis.Parameter or float
        thickness of slab (Angstrom)
    RXR_sld : :class:`refnx.reflect.RXR_Scatterer`, complex, or float
        (complex) tensor index of refraction of film
    rough : refnx.analysis.Parameter or float
        roughness on top of this slab (Angstrom)
    name : str
        Name of this slab
    """

    def __init__(self, thick, sld, rough, name='', vfsolv=0, interface=None):
        super(RXR_Slab, self).__init__(name=name)
        self.thick = possibly_create_parameter(thick,
                                               name=f'{name}_thick')
        if isinstance(sld, RXR_Scatterer):
            self.sld = sld 
        else:
            self.sld = RXR_SLD(sld)

        self.rough = possibly_create_parameter(rough,
                                               name=f'{name}_rough')

        p = Parameters(name=self.name)
        p.extend([self.thick])
        p.extend(self.sld.parameters)
        p.extend([self.rough])

        self._parameters = p

    def __repr__(self):
        return (f"Slab({self.thick!r}, {self.sld!r}, {self.rough!r},"
                f" name={self.name!r},")

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
        return np.array([[self.thick.value,
                          sldc.real,
                          sldc.imag,
                          self.rough.value]]) 
                          
    def tensor(self, energy=None):
        """
        Stored information pertaining to the tensor dielectric properties of the slab.
        Value of the tensor can be updated by changing the energy encoded into the SLD object.
        self.sld.energy = ()
        The trace (isotropic) of the layer is stored in the .slabs() attribute as the real and imaginary component of the SLD
        """
        if (energy is not None and hasattr(self.sld, 'energy')):
            self.sld.energy = energy
        return self.sld.tensor

class RXR_MixedMaterialSlab(RXR_Component):
    """
    A slab component made of several components

    Parameters
    ----------
    thick : refnx.analysis.Parameter or float
        thickness of slab (Angstrom)
    sld_list : sequence of {refnx.reflect.Scatterer, complex, float}
        Sequence of (complex) SLDs that are contained in film
        (/1e-6 Angstrom**2)
    vf_list : sequence of refnx.analysis.Parameter or float
        relative volume fractions of each of the materials contained in the
        film.
    rough : refnx.analysis.Parameter or float
        roughness on top of this slab (Angstrom)
    name : str
        Name of this slab

    Notes
    -----
    The SLD of this Slab is calculated using the normalised volume fractions of
    each of the constituent Scatterers:

    >>> np.sum([complex(sld) * vf / np.sum(vf_list) for sld, vf in
    ...         zip(sld_list, vf_list)]).

    The overall SLD then takes into account the volume fraction of solvent,
    `vfsolv`.
    """

    def __init__(
        self,
        thick,
        sld_list,
        vf_list,
        rough,
        name="",
    ):
        super(RXR_MixedMaterialSlab, self).__init__(name=name)
        
        self.thick = possibly_create_parameter(thick, name="%s - thick" % name)
        self.sld = []
        self.vf = []
        self._sld_parameters = Parameters(name=f"{name} - slds")
        self._vf_parameters = Parameters(name=f"{name} - volfracs")

        i = 0
        for s, v in zip(sld_list, vf_list):
            if isinstance(s, RXR_Scatterer):
                self.sld.append(s)
            else:
                self.sld.append(RXR_SLD(s))

            self._sld_parameters.append(self.sld[-1].parameters)

            vf = possibly_create_parameter(
                v, name=f"vf{i} - {name}", bounds=(0.0, 1.0)
            )
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
            f"RXR_MixedMaterialSlab({self.thick!r}, {self.sld!r}, {self.vf!r},"
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

        sldc = np.sum(
            [complex(sld) * vf / sum_vfs for sld, vf in zip(self.sld, vfs)]
        )

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
        
    def tensor(self, structure=None,energy=None):
        """
        Stored information pertaining to the tensor dielectric properties of the slab.
        Value of the tensor can be updated by changing the energy encoded into the SLD object.
        self.sld.energy = ()
        The trace of the layer is stored in the .slabs() attribute as the real and imaginary component of the SLD
        """
        vfs = np.array(self._vf_parameters)
        sum_vfs = np.sum(vfs)
        
        if (energy is not None and hasattr(self.sld, 'energy')):
            self.sld.energy = energy
        
        combinetensor = np.sum(
            [sld.tensor * vf / sum_vfs for sld, vf in zip(self.sld, vfs)], axis=0        
        )
        
        return combinetensor#self.sld.tensor
        
class RXR_MGMixedSlab(RXR_Component):
    """
    A slab component made of several components
    **Not confirmed by developer. Still in active testing/development.

    Parameters
    ----------
    thick : refnx.analysis.Parameter or float
        thickness of slab (Angstrom)
    sld_list : sequence of {refnx.reflect.Scatterer, complex, float}
        Sequence of (complex) SLDs that are contained in film
        (/1e-6 Angstrom**2) **The first in the list is the 'host' matrix for Maxwell-Garnett approximation
    vf : float
        Volume fraction of the dopant material in the 2-component slab
    shapefunc : List
        Three floats representing the x, y, z shape parameters for anisotropic model
        **No guarentee by the developer that one will be more 'correct' than another
        Spherical inclusion: [1/3, 1/3, 1/3] ***Default
        Rod-like inclusion: [1/2, 1/2, 0]
        Disk-like inclusion: []
        
    rough : refnx.analysis.Parameter or float
        roughness on top of this slab (Angstrom)
    name : str
        Name of this slab

    Notes
    -----
    The SLD of this Slab is calculated using the Maxwell-Garnett Effective Media model of
    each of the constituent Scatterers:

    >>> np.sum([complex(sld) * vf / np.sum(vf_list) for sld, vf in
    ...         zip(sld_list, vf_list)]).

    See Reference:
    Vadim A. Markel, "Introduction to the Maxwell Garnett
    approximation: tutorial," J. Opt. Soc. Am. A 33, 1244-1256 (2016)
    """

    def __init__(
        self,
        thick,
        rough,
        sld_list,
        vf,
        shapefunc = [1/3, 1/3, 1/3],
        name="",
    ):
        super(RXR_MGMixedSlab, self).__init__(name=name)
        
        self.thick = possibly_create_parameter(thick, name="%s - thick" % name)
        self.vf = possibly_create_parameter((vf/100 if vf>1 else vf), name=f"vf - {name}", bounds=(0.0, 1.0))
        self.rough = possibly_create_parameter(rough, name=f"{name} - rough")

        self.sld = []
        self._sld_parameters = Parameters(name=f"{name} - slds")
        self._vf_parameters = Parameters(name=f"{name} - volfracs")

        i = 0
        for s in sld_list:
            if isinstance(s, RXR_Scatterer):
                self.sld.append(s)
            else:
                self.sld.append(RXR_SLD(s))
            self._sld_parameters.append(self.sld[-1].parameters)
            i += 1

        p = Parameters(name=self.name)
        p.append([self.thick, self.vf])
        p.extend(self._sld_parameters)

        self._parameters = p
        self.shapefunc = np.array(shapefunc)

    def __repr__(self):
        return (
            f"RXR_MGMixedSlab({self.thick!r}, {self.sld!r}, {self.vf!r}, {self.shapefunc!r},"
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

        MGdelta = np.trace(self.tensor().real)/3
        MGbeta = np.trace(self.tensor().imag)/3
        sldc = complex(MGdelta, MGbeta)
        
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
        
    def tensor(self, structure=None,energy=None):
        """
        Stored information pertaining to the tensor dielectric properties of the slab.
        Value of the tensor can be updated by changing the energy encoded into the SLD object.
        self.sld.energy = ()
        The trace of the layer is stored in the .slabs() attribute as the real and imaginary component of the SLD
        """
        vf = self.vf.value
        host = np.diag(self.sld[0].tensor) #Just grab the diagonal elements for easy calculation
        dopant = np.diag(self.sld[1].tensor) #Just grab the diagonal elements for easy calculation
        
        if (energy is not None and hasattr(self.sld, 'energy')):
            self.sld.energy = energy
        
        numerator = host + (self.shapefunc * (1-vf) + (vf))*(dopant - host)
        denom = host + self.shapefunc * (1-vf) * (dopant - host)
        
        combinetensor = host * (numerator/denom)
        
        return combinetensor * np.eye(3)#self.sld.tensor 
        
#Support Functions

def birefringence_profile(slabs, tensor, z=None, step=False):
    """
    Calculates a series of depth profiles corresponding to the slab model used to calculated p-RSoXR

    Parameters
    ----------
    slabs : Information regarding the layer stack, see RXR_Structure class
    tensor : List of dielectric tensors from each layer stack, see RXR_Structure class
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
    nlayers = np.size(slabs, 0) - 2 #Calculate total number of layers (not including fronting/backing)

    # work on a copy of the input array
    layers = np.copy(slabs)
    layers[:, 0] = np.fabs(slabs[:, 0]) #Ensure the thickness is positive
    layers[:, 3] = np.fabs(slabs[:, 3]) #Ensure the roughness is positive
    # bounding layers should have zero thickness
    layers[0, 0] = layers[-1, 0] = 0

    # distance of each interface from the fronting interface
    dist = np.cumsum(layers[:-1, 0])
    total_film_thickness = int(np.round(dist[-1])) #Total film thickness for point density
    # workout how much space the SLD profile should encompass
    # (if z array not provided)
    if z is None:
        zstart = -5 - 4 * np.fabs(slabs[1, 3])
        zend = 5 + dist[-1] + 4 * layers[-1, 3]
        zed = np.linspace(zstart, zend, num=total_film_thickness*2) #0.5 Angstrom resolution default
    else:
        zed = np.asfarray(z)

    #Reduce the dimensionality of the tensor for ease of use
    reduced_tensor = tensor.diagonal(0,1,2) #0 - no offset, 1 - first axis of the tensor, 2 - second axis of the tensor 
    
    tensor_erf = np.ones((len(zed),3), dtype=float) * reduced_tensor[0] #Full wave of initial conditions  
    tensor_step = np.copy(tensor_erf) #Full wave without interfacial roughness
    delta_n = reduced_tensor[1:]  - reduced_tensor[:-1] #Change in n at each interface
    
    # use erf for roughness function, but step if the roughness is zero
    step_f = Step() #Step function (see refnx documentation)
    erf_f = Erf() #Error function (see refnx documentation)
    sigma = layers[1:, 3] #Interfacial width parameter

    # accumulate the SLD of each step.
    for i in range(nlayers + 1):
        f = erf_f
        g = step_f
        if sigma[i] == 0:
            f = step_f
        tensor_erf += delta_n[None,i,:]*f(zed, scale=sigma[i], loc=dist[i])[:,None] #Broadcast into a single item
        tensor_step += delta_n[None,i,:]*g(zed, scale=0, loc=dist[i])[:,None] #Broadcast into a single item
    
    return zed, tensor_erf if(step == False) else tensor_step
