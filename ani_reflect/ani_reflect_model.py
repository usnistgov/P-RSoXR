"""
*Calculates the specular (Neutron or X-ray) reflectivity from a stratified
series of layers.

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
import abc
import math
import numbers
import warnings

import numpy as np
import scipy
from scipy.interpolate import splrep, splev


from refnx.analysis import (Parameters, Parameter, possibly_create_parameter,
                            Transform)
from refnx.ani_reflect._ani_reflect import * ##TFerron Edits 05/20/2020 *Include model for anisotropic calculation
from refnx.ani_reflect._uniaxial_reflect import *


# some definitions for resolution smearing
_FWHM = 2 * np.sqrt(2 * np.log(2.0))
_INTLIMIT = 3.5


class ani_ReflectModel(object):
    r"""
    Parameters
    ----------
    structure : refnx.reflect.Structure
        The interfacial structure.
    scale : float or refnx.analysis.Parameter, optional
        scale factor. All model values are multiplied by this value before
        the background is added. This is turned into a Parameter during the
        construction of this object.
    bkg : float or refnx.analysis.Parameter, optional
        Q-independent constant background added to all model values. This is
        turned into a Parameter during the construction of this object.
    name : str, optional
        Name of the Model
    dq : float or refnx.analysis.Parameter, optional

        - `dq == 0` then no resolution smearing is employed.
        - `dq` is a float or refnx.analysis.Parameter
           a constant dQ/Q resolution smearing is employed.  For 5% resolution
           smearing supply 5.

        However, if `x_err` is supplied to the `model` method, then that
        overrides any setting given here. This value is turned into
        a Parameter during the construction of this object.
    threads: int, optional
        Specifies the number of threads for parallel calculation. This
        option is only applicable if you are using the ``_creflect``
        module. The option is ignored if using the pure python calculator,
        ``_reflect``. If `threads == -1` then all available processors are
        used.
    quad_order: int, optional
        the order of the Gaussian quadrature polynomial for doing the
        resolution smearing. default = 17. Don't choose less than 13. If
        quad_order == 'ultimate' then adaptive quadrature is used. Adaptive
        quadrature will always work, but takes a _long_ time (2 or 3 orders
        of magnitude longer). Fixed quadrature will always take a lot less
        time. BUT it won't necessarily work across all samples. For
        example, 13 points may be fine for a thin layer, but will be
        atrocious at describing a multilayer with bragg peaks.

    """
    def __init__(self, structure, scale=1, bkg=1e-7, name='', dq=0.,
                 threads=-1, quad_order=17, energy = None, qval = None, phi = 0, pol='s', backend = 'uni'): ##Tferron Edits 05/28/2020 Added a energy property to the reflectivity model to carry through to Anisotropic reflectivity (can be an array)
                                                                      ##Tferron Edits 05/28/2020 Added an angle phi representing the azimuthal angle of incidence with respect to the surface normal (for biaxial tensor properties)
                                                                      
        self.name = name
        self._parameters = None
        self.threads = threads
        self.quad_order = quad_order
        self.backend = backend
        ##Tferron Edits 05/28/2020 Added the energy property to carry through to anisotropic calculations /// And the angle phi for the angle of incidence 
        if type(energy) is list: energy = np.array(energy)
        self._energy = energy if type(energy) is np.ndarray else np.array([energy])## In eV
        self._qval = qval #Set to any value if you want to run an energy scan
        self.phi = phi
        self.pol = pol
        # to make it more like a refnx.analysis.Model
        self.fitfunc = None

        # all reflectometry models need a scale factor and background
        self._scale = possibly_create_parameter(scale, name='scale')
        self._bkg = possibly_create_parameter(bkg, name='bkg')

        # we can optimize the resolution (but this is always overridden by
        # x_err if supplied. There is therefore possibly no dependence on it.
        self._dq = possibly_create_parameter(dq, name='dq - resolution')

        self._structure = None
        self.structure = structure
        ##Generate information from the 4x4 reflectivity calculation...These will be stored if requested
        
    def __call__(self, x, p=None, x_err=None):
        r"""
        Calculate the generative model

        Parameters
        ----------
        x : float or np.ndarray
            q values for the calculation.
        p : refnx.analysis.Parameters, optional
            parameters required to calculate the model
        x_err : np.ndarray
            dq resolution smearing values for the dataset being considered.
        pol : string
            's' returns spol, and 'p' returns ppol

        Returns
        -------
        reflectivity : np.ndarray
            Calculated reflectivity
        """
        return self.model(x, p=p, x_err=x_err)

    def __repr__(self):
        return ("ReflectModel({_structure!r}, name={name!r},"
                " scale={_scale!r}, bkg={_bkg!r},"
                " dq={_dq!r}, threads={threads},"
                " quad_order={quad_order})".format(**self.__dict__))

    @property
    def dq(self):
        r"""
        :class:`refnx.analysis.Parameter`

            - `dq.value == 0`
               no resolution smearing is employed.
            - `dq.value > 0`
               a constant dQ/Q resolution smearing is employed.  For 5%
               resolution smearing supply 5. However, if `x_err` is supplied to
               the `model` method, then that overrides any setting reported
               here.

        """
        return self._dq

    @dq.setter
    def dq(self, value):
        self._dq.value = value

    @property
    def scale(self):
        r"""
        :class:`refnx.analysis.Parameter` - all model values are multiplied by
        this value before the background is added.

        """
        return self._scale

    @scale.setter
    def scale(self, value):
        self._scale.value = value

    @property
    def bkg(self):
        r"""
        :class:`refnx.analysis.Parameter` - linear background added to all
        model values.

        """
        return self._bkg

    @bkg.setter
    def bkg(self, value):
        self._bkg.value = value
        
    @property
    def energy(self):
        """
        Series of energies to model in this particular model, must match the dimension of the dataset if used in an objective
        
        """
        return self._energy

    @energy.setter
    def energy(self,energy):
        if type(energy) is list: energy = np.array(energy)
        self._energy = energy if type(energy) is np.ndarray else np.array([energy])
        
    @property
    def qval(self):
        return self._qval
    
    @qval.setter
    def qval(self, qval):
        self._qval = qval


    def model(self, x, p=None, x_err=None,save_components=None):
        r"""
        Calculate the reflectivity of this model

        Parameters
        ----------
        x : float or np.ndarray
            q or E values for the calculation.
            specifiy self.qval to be any value to fit energy-space
        p : refnx.analysis.Parameters, optional
            parameters required to calculate the model
        x_err : np.ndarray
            dq resolution smearing values for the dataset being considered.
        pol : sring
            's' for spol and 'p' for ppol

        Returns
        -------
        reflectivity : np.ndarray
            Calculated reflectivity

        """
        if p is not None:
            self.parameters.pvals = np.array(p)
        if x_err is None:
            # fallback to what this object was constructed with
            x_err = float(self.dq)
            
        #Determine if its an energy scan or qscan
        if self.qval is not None: #A single qval has been specified 
            if self.pol == 'fit':
                self.energy = np.linspace(np.min(x), np.max(x), 300)   
            else:
                self.energy = x
            qvals = self.qval
        else:
            if self.pol == 'fit': ##Data was concatenated so we need to make an input thats half the length
                qvals = np.linspace(np.min(x),np.max(x),400) ##400 is an arbitrary number for now.
            else:  
                qvals = x     
           
        ##loop over energy here ~~~ ?
        refl = np.zeros((len(self.energy),len(qvals),2,2),dtype=float)
        tran = np.zeros((len(self.energy),len(qvals),2,2),dtype=float)
        for i, energy in enumerate(self.energy):
            refl[i,:,:,:], tran[i,:,:,:] =  ani_reflectivity(qvals, self.structure.slabs(),
                                        self.structure.tensor(energy=energy),
                                        energy,
                                        self.phi,
                                        scale=self.scale.value,
                                        bkg=self.bkg.value,
                                        dq=x_err,
                                        threads=self.threads,
                                        quad_order=self.quad_order,
                                        ani_backend=self.backend)
        ## Check what the output is looking for (required to specify polarization for fitting)                            
        if self.pol == 's':
            return refl[0,:,1,1]#,0]
        elif self.pol == 'p':
            return refl[0,:,0,0]#,0]
        elif self.pol == 'fit':
            #Find the location that the spol and ppol data are split
            pol_swap_loc = np.argmax(np.abs(np.diff(x))) ##Where does it swap from the maximum Q of spol to the minimum Q at ppol
            spol_qvals = x[:pol_swap_loc+1]
            ppol_qvals = x[pol_swap_loc+1:]
            spol_fit = np.interp(spol_qvals,qvals,refl[0,:,1,1])
            ppol_fit = np.interp(ppol_qvals,qvals,refl[0,:,0,0])
                  
            return np.concatenate([spol_fit,ppol_fit])
        else:
            return refl
            
            
    def EFI(self, x, dz = 1, POI=[0.,1.,0.]):
        """
        Calculate the internal electric field of this model

        Parameters
        ----------
        x : float or np.ndarray
            q values for the calculation.
        dz : step size along the depth dimension


        Returns
        -------
        EFI : [ film depth , q ] np.ndarray
            Calculated electric field as a function of depth

        """
        ### --- Calculate the Reflectivity based on the input model --- ###
        ##DImensionality of outputs for later use ~~
        ##((4,4), layer, q, wavelength)

            
        kx, ky, kz, Dpol, Hpol, D, Di, P, W, Refl, Tran = yeh_4x4_reflectivity(x, self.structure.slabs(),
                                    self.structure.tensor(energy=self.energy),
                                    self.energy,
                                    self.phi,
                                    scale=self.scale.value,
                                    bkg=self.bkg.value,
                                    threads=self.threads,
                                    save_components=True)
        #print(kz.shape)
        #print(Dpol.shape)
        numpnts = Refl.shape[0]

        #Number of layers to consider
        numLayers = len(self.structure.slabs()[:,0])-2
        zInterface = np.zeros(numLayers + 2)
        zInterface[0:1] = 0
        for i in range(2,numLayers+1):
            zInterface[i] = zInterface[i-1] - self.structure.slabs()[-i,0]
        zInterface[-1] = zInterface[-2] - self.structure.slabs()[1,0]

        # E-Field Amplitude functions for each layer
        Amp_EField = np.zeros((numpnts, len(zInterface),4), dtype=complex)
        T = np.zeros((4,4),dtype=complex) ##Temporary matrix for generating the amplitude waves
        
        #Amplitude scale factors for Ex and Ey based on the 
        ##Currently these don't do anything. May need them if we start mixing polarization states.
        pol=np.array([1,0])
        A_spol = pol[0]/np.sqrt(pol[0]**2 + pol[1]**2) 
        A_ppol = pol[1]/np.sqrt(pol[0]**2 + pol[1]**2)
        
        ##Generate the amplitude functions for E
        # start with substrate
        #zInterface[0] = 0.0  # self.substrate.d   # set the substrate layer z to zero to get the correct amplitudes
        Amp_EField[:,0,0] = Tran[:,1,1] * A_ppol + Tran[:,0,1] * A_spol# ppol transmittance
        Amp_EField[:,0,1] = 0
        Amp_EField[:,0,2] = Tran[:,1,0] * A_ppol + Tran[:,0,0] * A_spol # spol transmittance
        Amp_EField[:,0,3] = 0
        
        for i in range(numpnts):
            T[:,:] = np.dot(Di[i,-2,:,:],D[i,-1,:,:]) * W[i,-1,:,:]
            Amp_EField[i,1,:] = np.dot(T[:,:],Amp_EField[i,0,:])
                
            for j in range(2,numLayers+1):
                T[:,:] = np.dot(np.dot(Di[i,-j-1,:,:], D[i,-j,:,:]) * W[i,-j,:,:] , P[i,-j,:,:])
                Amp_EField[i,j,:] = np.dot(T[:,:],Amp_EField[i,j-1,:])
                
            T[:,:] = np.dot(np.dot(Di[i,0,:,:], D[i,1,:,:])*W[i,1,:,:], P[i,1,:,:])
            Amp_EField[i,-1,:] = np.dot(T[:,:],Amp_EField[i,-2,:])
        #return [D, Di, P, W, Refl, Tran]
        zInterface = zInterface - zInterface[-1]
        ##Calculate the E-Field
        #print(zInterface)
        
        zpos = np.arange(-100,zInterface[0] + 100,dz)
        EField = np.zeros((numpnts,len(zpos),3), dtype=np.complex128)
        #current_layer = numLayers + 1
        for i in range(numpnts):
            current_layer = numLayers + 1
            #print(j)
            for j, z in enumerate(zpos):
                #Check current layer
                #print('z', z)
                #print('z change', zInterface[current_layer])
                if z > zInterface[current_layer] and current_layer > 0:
                    current_layer -= 1
                    #print('change layer', current_layer+1, 'to', current_layer)
                    #if current_layer == 0:
                    #    print('zero')
                    #else:
                    #    print('not zero')
                #Calculate Field 
                EField[i,j,:] = np.dot(Amp_EField[i,current_layer,:]* np.exp(1j*(kz[i,-current_layer-1,:] * (z - zInterface[current_layer]))),Dpol[i,-current_layer-1,:])

        
        return zpos, EField, zInterface[1:]
        """
        # calculate first layer manually
        zInterface[1] = 0.0
        T = np.dot(self.layers[0].Di, self.substrate.D)
        # ignore the substrate P-matrix as the amplitudes are for the last interface
        An[1] = np.dot(T, An[0])

        # calculate intermediate layers in loop
        for i in range(2, N + 1):
            zn[i] = zn[i - 1] - self.layers[i - 2].d
            T = np.dot(self.layers[i - 1].Di, np.dot(self.layers[i - 2].D, self.layers[i - 2].P))
            An[i] = np.dot(T, An[i - 1])

        # calculate last layer / superstrate manually again
        zInterface[-1] = zInterface[-2] - self.layers[-1].d

        T = np.dot(self.superstrate.Di, np.dot(self.layers[-1].D, self.layers[-1].P))
        An[-1] = np.dot(T, An[-2])

        """    
    def logp(self):
        r"""
        Additional log-probability terms for the reflectivity model. Do not
        include log-probability terms for model parameters, these are
        automatically included elsewhere.

        Returns
        -------
        logp : float
            log-probability of structure.

        """
        return self.structure.logp()

    @property
    def structure(self):
        r"""
        :class:`refnx.reflect.Structure` - object describing the interface of
        a reflectometry sample.

        """
        return self._structure

    @structure.setter
    def structure(self, structure):
        self._structure = structure
        p = Parameters(name='instrument parameters')
        p.extend([self.scale, self.bkg, self.dq])

        self._parameters = Parameters(name=self.name)
        self._parameters.extend([p, structure.parameters])

    @property
    def parameters(self):
        r"""
        :class:`refnx.analysis.Parameters` - parameters associated with this
        model.

        """
        self.structure = self._structure
        return self._parameters


def ani_reflectivity(q, slabs, tensor, energy = np.array([250]), phi = np.array([0]), scale=1., bkg=0., dq=5., quad_order=17,
                 threads=-1,ani_backend='uni'):
    r"""
    Full biaxial tensor calculation for calculating reflectivity from a stratified medium.

    Parameters
    ----------
    q : np.ndarray
        The qvalues required for the calculation.
        :math:`Q=\frac{4Pi}{\lambda}\sin(\Omega)`.
        Units = Angstrom**-1
    slabs : np.ndarray
        coefficients required for the calculation, has shape (2 + N, 5),
        where N is the number of layers

        - slabs[0, 0]
           ignored
        - slabs[N, 0]
           thickness of layer N
        - slabs[N+1, 0]
           ignored

        - slabs[0, 1]
           trace of biaxial SLD_real of fronting (/1e-6 Angstrom**-2)
        - slabs[N, 1]
           trace of biaxial SLD_real of layer N (/1e-6 Angstrom**-2)
        - slabs[-1, 1]
           trace of biaxial SLD_real of backing (/1e-6 Angstrom**-2)

        - slabs[0, 2]
           trace of biaxial iSLD_imag of fronting (/1e-6 Angstrom**-2)
        - slabs[N, 2]
           trace of biaxial iSLD_real of layer N (/1e-6 Angstrom**-2)
        - slabs[-1, 2]
           trace of biaxial iSLD_real of backing (/1e-6 Angstrom**-2)

        - slabs[0, 3]
           ignored
        - slabs[N, 3]
           roughness between layer N-1/N
        - slabs[-1, 3]
           roughness between backing and layer N
        
        -slabs[0, 4]
            full biaxial tensor of fronting media
        -slabs[N, 4]
            full biaxial tensor of layer N
        -slabs[-1, 4]
            full biaxial tensor of backing media
        
    tensor : 3x3 numpy array
        The full dielectric tensor required for the anisotropic calculation.
        Each component (real and imaginary) is a fit parameter 
        units - ???
        
    Energy : numpy array
        A list of energies that you want to calculate the reflectivity profile for. 
        When fitting it may be adventageous to fit one at a time, although simultaneous fitting may be possible
        NOT IMPLEMETED YET FOR MORE THAN 1 ENERGY
        
    phi : float
        Azimuthal angle of incidence for calculating k-vectors.
        This is only required if dealing with a biaxial tensor and interested in the rotational variation in reflection
        defaults to phi = 0 ~
        
    scale : float
        scale factor. All model values are multiplied by this value before
        the background is added
    bkg : float
        Q-independent constant background added to all model values.
    dq : float or np.ndarray, optional
        - `dq == 0`
           no resolution smearing is employed.
        - `dq` is a float
           a constant dQ/Q resolution smearing is employed.  For 5% resolution
           smearing supply 5.
        - `dq` is the same shape as q
           the array contains the FWHM of a Gaussian approximated resolution
           kernel. Point by point resolution smearing is employed.  Use this
           option if dQ/Q varies across your dataset.
        - `dq.ndim == q.ndim + 2` and `q.shape == dq[..., -3].shape`
           an individual resolution kernel is applied to each measurement
           point. This resolution kernel is a probability distribution function
           (PDF). `dqvals` will have the shape (qvals.shape, 2, M).  There are
           `M` points in the kernel. `dq[:, 0, :]` holds the q values for the
           kernel, `dq[:, 1, :]` gives the corresponding probability.
    quad_order: int, optional
        the order of the Gaussian quadrature polynomial for doing the
        resolution smearing. default = 17. Don't choose less than 13. If
        quad_order == 'ultimate' then adaptive quadrature is used. Adaptive
        quadrature will always work, but takes a _long_ time (2 or 3 orders
        of magnitude longer). Fixed quadrature will always take a lot less
        time. BUT it won't necessarily work across all samples. For
        example, 13 points may be fine for a thin layer, but will be
        atrocious at describing a multilayer with bragg peaks.
    threads: int, optionalv ##UNSURE IF USABLE FOR CURRENT CALCULATION
        Specifies the number of threads for parallel calculation. This
        option is only applicable if you are using the ``_creflect``
        module. The option is ignored if using the pure python calculator,
        ``_reflect``. If `threads == -1` then all available processors are
        used.

    Example
    -------

    >>> from refnx.reflect import reflectivity
    >>> q = np.linspace(0.01, 0.5, 1000)
    >>> slabs = np.array([[0, 2.07, 0, 0],
    ...                   [100, 3.47, 0, 3],
    ...                   [500, -0.5, 0.00001, 3],
    ...                   [0, 6.36, 0, 3]])
    >>> print(reflectivity(q, slabs))
    """
    
    #Generate Output datasets
    #Refl = np.zeros((2,2,len(q),len(Energy)),dtype=complex)
    #Tran = np.zeros((2,2,len(q),len(Energy)),dtype=complex)
    
    
    
    # constant dq/q smearing
    if isinstance(dq, numbers.Real) and float(dq) == 0:
        if ani_backend == 'uni':
            Refl, Tran = uniaxial_reflectivity(q, slabs, tensor, energy, phi, scale=scale, bkg=bkg, threads=threads, save_components=None)
        else:
            Refl, Tran = yeh_4x4_reflectivity(q, slabs, tensor, energy, phi, scale=scale, bkg=bkg, threads=threads, save_components=None)
        return [Refl, Tran]
            
    elif isinstance(dq, numbers.Real):
        dq = float(dq)
        smear_refl, smear_tran = _smeared_yeh_4x4_reflectivity(q,
                                         slabs,
                                         tensor, energy, phi,
                                         dq,
                                         threads=threads,save_components=None)
        return [(scale*smear_tefl + bkg), smear_tran]
    """ ##None of this functionality exists currently for anisotropic calculation
    # point by point resolution smearing (each q point has different dq/q)
    if isinstance(dq, np.ndarray) and dq.size == q.size:
        dqvals_flat = dq.flatten()
        qvals_flat = q.flatten()

        # adaptive quadrature
        if quad_order == 'ultimate':
            smeared_rvals = (scale *
                             _smeared_abeles_adaptive(qvals_flat,
                                                      slabs,
                                                      dqvals_flat,
                                                      threads=threads) +
                             bkg)
            return smeared_rvals.reshape(q.shape)
        # fixed order quadrature
        else:
            smeared_rvals = (scale *
                             _smeared_abeles_fixed(qvals_flat,
                                                   slabs,
                                                   dqvals_flat,
                                                   quad_order=quad_order,
                                                   threads=threads) +
                             bkg)
            return np.reshape(smeared_rvals, q.shape)

    # resolution kernel smearing
    elif (isinstance(dq, np.ndarray) and
          dq.ndim == q.ndim + 2 and
          dq.shape[0: q.ndim] == q.shape):

        qvals_for_res = dq[:, 0, :]
        # work out the reflectivity at the kernel evaluation points
        smeared_rvals = abeles(qvals_for_res,
                               slabs,
                               threads=threads)

        # multiply by probability
        smeared_rvals *= dq[:, 1, :]

        # now do simpson integration
        rvals = scipy.integrate.simps(smeared_rvals, x=dq[:, 0, :])

        return scale * rvals + bkg

    return None
    """

def _memoize_gl(f):
    """
    Cache the gaussian quadrature abscissae, so they don't have to be
    calculated all the time.
    """
    cache = {}

    def inner(n):
        if n in cache:
            return cache[n]
        else:
            result = cache[n] = f(n)
            return result
    return inner


@_memoize_gl
def gauss_legendre(n):
    """
    Calculate gaussian quadrature abscissae and weights
    Parameters
    ----------
    n : int
        Gaussian quadrature order.
    Returns
    -------
    (x, w) : tuple
        The abscissae and weights for Gauss Legendre integration.
    """
    return scipy.special.p_roots(n)


def _smearkernel(x, w, q, dq, threads):
    """
    Adaptive Gaussian quadrature integration

    Parameters
    ----------
    x : float
        Independent variable for integration.
    w : array-like
        The uniform slab model parameters in 'layer' form.
    q : float
        Nominal mean Q of normal distribution
    dq : float
        FWHM of a normal distribution.
    threads : int
        number of threads for parallel calculation
    Returns
    -------
    reflectivity : float
        Model reflectivity multiplied by the probability density function
        evaluated at a given distance, x, away from the mean Q value.
    """
    prefactor = 1 / np.sqrt(2 * np.pi)
    gauss = prefactor * np.exp(-0.5 * x * x)
    localq = q + x * dq / _FWHM
    return abeles(localq, w, threads=threads) * gauss


def _smeared_yeh_4x4_reflectivity(q, w, tensor, Energy, phi, resolution, threads=-1,save_components=None):
    """
    Fast resolution smearing for constant dQ/Q.

    Parameters
    ----------
    q: np.ndarray
        Q values to evaluate the reflectivity at
    w: np.ndarray
        Parameters for the reflectivity model
    resolution: float
        Percentage dq/q resolution. dq specified as FWHM of a resolution
        kernel.
    threads: int, optional
        Do you want to calculate in parallel? This option is only applicable if
        you are using the ``_creflect`` module. The option is ignored if using
        the pure python calculator, ``_reflect``.
    Returns
    -------
    reflectivity: np.ndarray
        The resolution smeared reflectivity
    """

    if resolution < 0.5:
        return yeh_4x4_reflectivity(q, w, tensor, Energy, phi, threads=threads,save_components=None)

    resolution /= 100
    gaussnum = 51
    gaussgpoint = (gaussnum - 1) / 2

    def gauss(x, s):
        return 1. / s / np.sqrt(2 * np.pi) * np.exp(-0.5 * x**2 / s / s)

    lowq = np.min(q)
    highq = np.max(q)
    if lowq <= 0:
        lowq = 1e-6

    start = np.log10(lowq) - 6 * resolution / _FWHM
    finish = np.log10(highq * (1 + 6 * resolution / _FWHM))
    interpnum = np.round(np.abs(1 * (np.abs(start - finish)) /
                         (1.7 * resolution / _FWHM / gaussgpoint)))
    xtemp = np.linspace(start, finish, int(interpnum))
    xlin = np.power(10., xtemp)

    # resolution smear over [-4 sigma, 4 sigma]
    gauss_x = np.linspace(-1.7 * resolution, 1.7 * resolution, gaussnum)
    gauss_y = gauss(gauss_x, resolution / _FWHM)

    Refl, Tran = yeh_4x4_reflectivity(xlin, w, tensor, Energy, phi, threads=threads,save_components=None)
    ##Convolve each solution independently
    smeared_ss = np.convolve(Refl[:,0,0], gauss_y, mode='same') * (gauss_x[1] - gauss_x[0])
    smeared_pp = np.convolve(Refl[:,1,1], gauss_y, mode='same') * (gauss_x[1] - gauss_x[0])
    smeared_sp = np.convolve(Refl[:,0,1], gauss_y, mode='same') * (gauss_x[1] - gauss_x[0])
    smeared_ps = np.convolve(Refl[:,1,0], gauss_y, mode='same') * (gauss_x[1] - gauss_x[0])

    #smeared_rvals *= gauss_x[1] - gauss_x[0]

    # interpolator = InterpolatedUnivariateSpline(xlin, smeared_rvals)
    #
    # smeared_output = interpolator(q)
    ##Re-interpolate and organize the results wave following spline interpolation
    tck_ss = splrep(xlin, smeared_ss)
    smeared_output_ss = splev(q, tck_ss)
    
    tck_sp = splrep(xlin, smeared_sp)
    smeared_output_sp = splev(q, tck_sp)
    
    tck_ps = splrep(xlin, smeared_ps)
    smeared_output_ps = splev(q, tck_ps)
    
    tck_pp = splrep(xlin, smeared_pp)
    smeared_output_pp = splev(q, tck_pp)
    
    ##Organize the output wave with the appropriate outputs
    smeared_output = np.rollaxis(np.array([[smeared_output_ss,smeared_output_sp],[smeared_output_ps,smeared_output_pp]]),2,0)

    return smeared_output, Tran
    
