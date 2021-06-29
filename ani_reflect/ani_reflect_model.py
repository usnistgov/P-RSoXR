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
#from ani_reflect._biaxial_reflect import * ##TFerron Edits 05/20/2020 *Include model for anisotropic calculation
from ani_reflect._uniaxial_reflect import *


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
    def __init__(self, structure, scale=1, bkg=0, name='', dq=0.,
                 threads=-1, quad_order=17, energy = None, phi = 0, pol='s', backend = 'uni'): ##Tferron Edits 05/28/2020 Added a energy property to the reflectivity model to carry through to Anisotropic reflectivity
                                                                      ##Tferron Edits 05/28/2020 Added an angle phi representing the azimuthal angle of incidence with respect to the surface normal (for biaxial tensor properties)
                                                                      
        self.name = name
        self._parameters = None
        self.threads = threads
        self.quad_order = quad_order
        self.backend = backend
        ##Tferron Edits 05/28/2020 Added the energy property to carry through to anisotropic calculations /// And the angle phi for the angle of incidence 
        self._energy = energy ## In eV
        self.phi = phi
        self.pol = pol #Output polarization
        
        # all reflectometry models need a scale factor and background
        self._scale = possibly_create_parameter(scale, name='scale')
        self._bkg = possibly_create_parameter(bkg, name='bkg')

        # we can optimize the resolution (but this is always overridden by
        # x_err if supplied. There is therefore possibly no dependence on it.
        self._dq = possibly_create_parameter(dq, name='dq - resolution')

        self._structure = None
        self.structure = structure
        
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
        

        Returns
        -------
        reflectivity : np.ndarray
            Calculated reflectivity
            
            
        Note:
        -------
        Uses the assigned 'Pol' to determine the output state of 's-pol', 'p-pol' or both
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
        self._energy = energy 

    def model(self, x, p=None, x_err=None):
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
            

        if self.pol == 'fit': ##Data was concatenated prior to fitting / need to make an input thats half the length
            num_q = len(x) + 50 #50 more datapoints than the data used to calculate the spectra (can be adjusted)
            qvals = np.linspace(np.min(x), np.max(x), num_q) ##400 is an arbitrary number for now.
        else:  
            qvals = x     
           
        ##loop over energy here ~~~ ?
        refl = np.zeros((len(qvals),2,2),dtype=float)
        tran = np.zeros((len(qvals),2,2),dtype=complex)
        refl[:,:,:], tran[:,:,:] =  ani_reflectivity(qvals, self.structure.slabs(),
                                    self.structure.tensor(energy=self.energy),
                                    self.energy,
                                    self.phi,
                                    scale=self.scale.value,
                                    bkg=self.bkg.value,
                                    dq=x_err,
                                    threads=self.threads,
                                    quad_order=self.quad_order,
                                    ani_backend=self.backend)
        ## Check what the output is looking for (required to specify polarization for fitting)                            
        if self.pol == 's':
            return refl[:,1,1]#,0]
        elif self.pol == 'p':
            return refl[:,0,0]#,0]
        elif self.pol == 'fit':
            #Find the location that the spol and ppol data are split
            pol_swap_loc = np.argmax(np.abs(np.diff(x))) ##Where does it swap from the maximum Q of spol to the minimum Q at ppol
            spol_qvals = x[:pol_swap_loc+1]
            ppol_qvals = x[pol_swap_loc+1:]
            spol_fit = np.interp(spol_qvals,qvals,refl[:,1,1]) #Interpolate the fit back to the datapoints
            ppol_fit = np.interp(ppol_qvals,qvals,refl[:,0,0])
                  
            return np.concatenate([spol_fit,ppol_fit]) 
        else:
            return refl
            
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


def ani_reflectivity(q, slabs, tensor, energy=250.0, phi=0, scale=1., bkg=0., dq=0., quad_order=17,
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
        
    # constant dq/q smearing
    if isinstance(dq, numbers.Real) and float(dq) == 0:
        if ani_backend == 'uni':
            Refl, Tran = uniaxial_reflectivity(q, slabs, tensor, energy, phi, scale=scale, bkg=bkg, threads=threads, save_components=None)
        else:
            Refl, Tran = yeh_4x4_reflectivity(q, slabs, tensor, energy, phi, scale=scale, bkg=bkg, threads=threads, save_components=None)
        return [Refl, Tran]
            
    elif isinstance(dq, numbers.Real):
        dq = float(dq)
        smear_refl, smear_tran = _smeared_ani_reflectivity(q,
                                         slabs,
                                         tensor, energy, phi,
                                         dq,
                                         threads=threads,save_components=None,
                                         backend=ani_backend)

        return [(scale*smear_tefl + bkg), smear_tran]

    return None

def _smeared_ani_reflectivity(q, w, tensor, Energy, phi, resolution, threads=-1,save_components=None, backend='uni'):
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
        if backend == 'uni':
            return uniaxial_reflectivity(q, w, tensor, Energy, phi, threads=threads,save_components=None)
        else:
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
    if backend == 'uni':
        Refl, Tran = uniaxial_reflectivity(xlin, w, tensor, Energy, phi, threads=threads,save_components=None)
    else:
        Refl, Tran = yeh_4x4_reflectivity(xlin, w, tensor, Energy, phi, threads=threads,save_components=None)
    #Refl, Tran = yeh_4x4_reflectivity(xlin, w, tensor, Energy, phi, threads=threads,save_components=None)
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
    
    
