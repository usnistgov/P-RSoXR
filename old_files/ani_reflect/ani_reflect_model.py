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


"""
Calculates the specular reflectivity from a stratified series of layers
using polarized resonant soft X-ray reflectivity.
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


class PXR_ReflectModel(object):
    r"""
    Parameters
    ----------
    structure : PRSOXS.PXR_Structure object
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
        
        Adding q-smearing greatly reduces the current speed of the calculation.
        Running at ALS 11.0.1.2 likely does not require any q-smearing
        as a result of the low photon energy at the carbon edge.

    """
    def __init__(self, structure, scale=1, bkg=0, name='', dq=0.,
                 energy = None, phi = 0, pol='s', backend = 'uni'): ##Tferron Edits 05/28/2020 Added a energy property to the reflectivity model to carry through to Anisotropic reflectivity
                                                                      ##Tferron Edits 05/28/2020 Added an angle phi representing the azimuthal angle of incidence with respect to the surface normal (for biaxial tensor properties)                                                                   
        self.name = name
        self._parameters = None
        self.backend = backend
        ##Tferron Edits 05/28/2020 Added the energy property to carry through to anisotropic calculations /// And the angle phi for the angle of incidence 
        self._energy = energy ## In eV
        self._phi = phi
        self._pol = pol #Output polarization
        
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
        Energy to calculate the resonant reflectivity.
        """
        return self._energy

    @energy.setter
    def energy(self,energy):
        self._energy = energy
        
    @property
    def pol(self):
        """
        Polarization to calculate the resonant reflectivity. 
            s-polarization: 's'
            p-polarization: 'p'
            concatenated:   'sp'
        """
        return self._pol

    @pol.setter
    def pol(self, pol):
        self._pol = pol 
        
    @property
    def phi(self):
        """
        Azimuthal angle of incidence [deg]. Only used with a biaxial calculation.
        """
        return self._phi

    @phi.setter
    def phi(self, phi):
        self._phi = phi

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
        
        #Multipol fitting is currently done through concatenating s- and p-pol together.
        #A temp x-data set is used to calculate the model based on the q-range
        if self.pol == 'sp':
            concat_loc = np.argmax(np.abs(np.diff(x))) #Location where the q-range swaps for high s-pol to low p-pol
            qvals_spol = x[:concat_loc+1] #Split inputs for later
            qvals_ppol = x[concat_loc+1:] #Split inputs for later
            num_q = concat_loc + 50 #50 more points to make sure the interpolation works
            qvals = np.linspace(np.min(x), np.max(x), num_q)
        else:
            qvals = x
                    
        refl, tran, *components = PXR_reflectivity(qvals, self.structure.slabs(),
                                self.structure.tensor(energy=self.energy),
                                self.energy,
                                self.phi,
                                scale=self.scale.value,
                                bkg=self.bkg.value,
                                dq=x_err,
                                backend=self.backend
                                )
        #Return result based on desired polarization:


        if self.pol == 's':
            output = refl[:,1,1]
        elif self.pol == 'p':
            output = refl[:,0,0]
        elif self.pol == 'sp':
            spol_model = np.interp(qvals_spol, qvals, refl[:,1,1])
            ppol_model = np.interp(qvals_ppol, qvals, refl[:,0,0])
            output =  np.concatenate([spol_model, ppol_model])
        elif self.pol == 'ps':
            spol_model = np.interp(qvals_spol, qvals, refl[:,1,1])
            ppol_model = np.interp(qvals_ppol, qvals, refl[:,0,0])
            output = np.concatenate([ppol_model, spol_model])
        
        else:
            print('No polarizations were chosen for model')
            output = 0
        
        return output

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
        :class:`PRSoXR.PXR_Structure` - object describing the interface of
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


def PXR_reflectivity(q, slabs, tensor, energy=250.0, phi=0, scale=1., bkg=0., dq=0., backend='uni'):
    r"""
    Full calculation for anisotropic reflectivity of a stratified medium
    
    Parameters
    ----------
    q : np.ndarray
        The qvalues required for the calculation.
        :math:`Q=\frac{4Pi}{\lambda}\sin(\Omega)`.
        Units = Angstrom**-1
    slabs : np.ndarray
        coefficients required for the calculation, has shape (2 + N, 4),
        where N is the number of layers

        - slabs[0, 0]
           ignored
        - slabs[N, 0]
           thickness of layer N
        - slabs[N+1, 0]
           ignored

        - slabs[0, 1]
           trace of real index tensor of fronting (/1e-6 Angstrom**-2)
        - slabs[N, 1]
           trace of real index tensor of layer N (/1e-6 Angstrom**-2)
        - slabs[-1, 1]
           trace of real index tensor of backing (/1e-6 Angstrom**-2)

        - slabs[0, 2]
           trace of imaginary index tensor of fronting (/1e-6 Angstrom**-2)
        - slabs[N, 2]
           trace of imaginary index tensor of layer N (/1e-6 Angstrom**-2)
        - slabs[-1, 2]
           trace of imaginary index tensor of backing (/1e-6 Angstrom**-2)

        - slabs[0, 3]
           ignored
        - slabs[N, 3]
           roughness between layer N-1/N
        - slabs[-1, 3]
           roughness between backing and layer N
        
    tensor : 3x3 numpy array
        The full dielectric tensor required for the anisotropic calculation.
        Each component (real and imaginary) is a fit parameter 
        Has shape (2 + N, 3, 3)
        units - unitless
        
    Energy : float
        Energy to calculate the reflectivity profile 
        Used in calculating 'q' and index of refraction for PXR_MaterialSLD objects
        
    phi : float
        Azimuthal angle of incidence for calculating k-vectors.
        This is only required if dealing with a biaxial tensor
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
        if backend == 'uni':
            refl, tran, *components = uniaxial_reflectivity(q, slabs, tensor, energy)
        else:
            refl, tran, *components = yeh_4x4_reflectivity(q, slabs, tensor, energy, phi)
        return (scale*refl + bkg), tran, components
            
    elif isinstance(dq, numbers.Real):
        dq = float(dq)
        smear_refl, smear_tran, *components = _smeared_PXR_reflectivity(q,
                                         slabs,
                                         tensor, energy, phi,
                                         dq,
                                         backend=backend)

        return [(scale*smear_tefl + bkg), smear_tran, components]

    return None

def _smeared_PXR_reflectivity(q, w, tensor, energy, phi, resolution, backend='uni'):
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

    Returns
    -------
    reflectivity: np.ndarray
        The resolution smeared reflectivity
    """

    if resolution < 0.5:
        if backend == 'uni':
            return uniaxial_reflectivity(q, w, tensor, energy)
        else:
            return yeh_4x4_reflectivity(q, w, tensor, energy, phi)

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
        refl, tran, *components = uniaxial_reflectivity(xlin, w, tensor, energy)
    else:
        refl, tran, *components = yeh_4x4_reflectivity(xlin, w, tensor, energy, phi)
    #Refl, Tran = yeh_4x4_reflectivity(xlin, w, tensor, Energy, phi, threads=threads,save_components=None)
    ##Convolve each solution independently
    smeared_ss = np.convolve(refl[:,0,0], gauss_y, mode='same') * (gauss_x[1] - gauss_x[0])
    smeared_pp = np.convolve(refl[:,1,1], gauss_y, mode='same') * (gauss_x[1] - gauss_x[0])
    smeared_sp = np.convolve(refl[:,0,1], gauss_y, mode='same') * (gauss_x[1] - gauss_x[0])
    smeared_ps = np.convolve(refl[:,1,0], gauss_y, mode='same') * (gauss_x[1] - gauss_x[0])

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

    return smeared_output, tran, components
    
    
