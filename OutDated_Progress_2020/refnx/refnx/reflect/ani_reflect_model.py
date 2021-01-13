"""
*Calculates the polarized X-ray reflectivity from an anisotropic stratified
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


# some definitions for resolution smearing
_FWHM = 2 * np.sqrt(2 * np.log(2.0))
_INTLIMIT = 3.5

    ##fundamental constants ##may not need these if converion to Gaussian units works (c=1)
hc = 12398.4193 ##ev*Angstroms
c = 299792458.
mu0 = 4. * np.pi * 1e-7
ep0 = 1. / (c**2 * mu0)
    


"""
Implementation notes
--------------------
1. For _smeared_abeles_fixed I investigated calculating a master curve,
   adjacent data points have overlapping resolution kernels. So instead of
   using e.g. an oversampling factor of 17, one could get away with using
   a factor of 6. This is because the calculated points can be used to smear
   more than one datapoint. One can't use Gaussian quadrature, Simpsons rule is
   needed. Technically the approach works, but turns out to be no faster than
   the Gaussian quadrature with the x17 oversampling (even if everything is
   vectorised). There are a couple of reasons: a) calculating the Gaussian
   weights has to be re-done for all the resolution smearing points for every
   datapoint. For Gaussian quadrature that calculation only needs to be done
   once, because the oversampling points are at constant locations around the
   mean. b) in the implementation I tried the Simpsons rule had to integrate
   e.g. 700 odd points instead of the fixed 17 for the Gaussin quadrature.
"""


def get_reflect_backend(backend='c'):
    r"""

    Parameters
    ----------
    backend: {'python', 'cython', 'c'}, str
        The module that calculates the reflectivity. Speed should go in the
        order cython > c > python. If a particular method is not available the
        function falls back: cython --> c --> python.

    Returns
    -------
    abeles: callable
        The callable that calculates the reflectivity

    """
    if backend == 'cython':
        try:
            from refnx.reflect import _cyreflect as _cy
            f = _cy.abeles
        except ImportError:
            return get_reflect_backend('c')
    elif backend == 'c':
        try:
            from refnx.reflect import _creflect as _c
            f = _c.abeles
        except ImportError:
            return get_reflect_backend('python')
    elif backend == 'python':
        warnings.warn("Using the SLOW reflectivity calculation.")
        from refnx.reflect import _reflect as _py
        f = _py.abeles

    return f


abeles = get_reflect_backend()




def polarized_reflectivity(q, slabs, tensor, Energy = np.array([250]), phi = np.array([0]), scale=1., bkg=0., dq=5., quad_order=17,
                 threads=-1,EFI = False):
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
    Refl = np.zeros((2,2,len(q),len(Energy)),dtype=complex)
    Tran = np.zeros((2,2,len(q),len(Energy)),dtype=complex)
    
    
    
    # constant dq/q smearing
    if isinstance(dq, numbers.Real) and float(dq) == 0:
        if EFI:
            [kx,ky,kz,Dpol,D,Di,P,W,Refl,Tran] = yeh_4x4_reflectivity(q, slabs, tensor, Energy, phi, scale=scale, bkg=bkg, threads=threads,EFI=EFI)
            return [kx,ky,kz,Dpol,D,Di,P,W,Refl,Tran]
        else:
            Refl, Tran = yeh_4x4_reflectivity(q, slabs, tensor, Energy, phi, scale=scale, bkg=bkg, threads=threads,EFI=EFI)
            return Refl, Tran
    elif isinstance(dq, numbers.Real):
        dq = float(dq)
        return (scale *
                _smeared_yeh_4x4_reflectivity(q,
                                         slabs,
                                         tensor, Energy, phi,
                                         dq,
                                         threads=threads,EFI = EFI)) + bkg
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

def _smeared_yeh_4x4_reflectivity(q, w, tensor, Energy, phi, resolution, threads=-1,EFI=False):
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
        return yeh_4x4_reflectivity(q, w, tensor, Energy, phi, threads=threads,EFI=EFI)

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

    rvals, rvals = yeh_4x4_reflectivity(xlin, w, tensor, Energy, phi, threads=threads,EFI=EFI)
    smeared_rvals = np.convolve(rvals, gauss_y, mode='same')
    smeared_rvals *= gauss_x[1] - gauss_x[0]

    # interpolator = InterpolatedUnivariateSpline(xlin, smeared_rvals)
    #
    # smeared_output = interpolator(q)

    tck = splrep(xlin, smeared_rvals)
    smeared_output = splev(q, tck)

    return smeared_output
    
    
# TINY = np.finfo(np.float64).tiny
TINY = 1e-30

"""
import numpy as np
q = np.linspace(0.01, 0.5, 1000)
w = np.array([[0, 2.07, 0, 0],
              [100, 3.47, 0, 3],
              [500, -0.5, 0.00001, 3],
              [0, 6.36, 0, 3]])
"""

"""
The timings for the reflectivity calculation above are (6/3/2019):

_creflect.abeles = 254 us
_reflect.abeles = 433 us
the alternative cython implementation is 572 us.

If TINY is made too small, then the C implementations start too suffer because
the sqrt calculation takes too long. The C implementation is only just ahead of
the python implementation!
"""


def yeh_4x4_reflectivity(q, layers, tensor, Energy, phi, scale=1., bkg=0, threads=0, EFI = False):
    """
    EMpy implementation of the biaxial 4x4 matrix formalism for calculating reflectivity from a stratified
    medium.
    
    Currently uses the PyATMM formalism instead of EMpy......only works with uniaxial stuff so make sure your eyy parameter does not vary
    This is a rather sloppy implementation but it is currently the first attempt until we rewrite wither EMpy or PyATMM into a more user friendly format
    Parameters
    ----------
    q: array_like
        the q values required for the calculation.
        Q = 4 * Pi / lambda * sin(omega).
        Units = Angstrom**-1
    layers: np.ndarray
        coefficients required for the calculation, has shape (2 + N, 4),
        where N is the number of layers
        layers[0, 1] - SLD of fronting (/1e-6 Angstrom**-2)
        layers[0, 2] - iSLD of fronting (/1e-6 Angstrom**-2)
        layers[N, 0] - thickness of layer N
        layers[N, 1] - SLD of layer N (/1e-6 Angstrom**-2)
        layers[N, 2] - iSLD of layer N (/1e-6 Angstrom**-2)
        layers[N, 3] - roughness between layer N-1/N
        layers[-1, 1] - SLD of backing (/1e-6 Angstrom**-2)
        layers[-1, 2] - iSLD of backing (/1e-6 Angstrom**-2)
        layers[-1, 3] - roughness between backing and last layer
    scale: float
        Multiply all reflectivities by this value.
    bkg: float
        Linear background to be added to all reflectivities
    threads: int, optional
        <THIS OPTION IS CURRENTLY IGNORED>
    EFI: True/False
        Set True if you want your output to include dynamical matrices for EFI calculation

    Returns
    -------
    Reflectivity: np.ndarray
        Calculated reflectivity values for each q value.
    """

    ##Organize qvals into proper order
    qvals = np.asfarray(q)
    flatq = qvals.ravel()
    numpnts = flatq.size
    
    ##Grab the number of layers
    nlayers = layers.shape[0]
   
    #Setup wavelength and energy array
    if type(Energy) is not np.ndarray: #Just keep everything as a numpy array
        Energy = np.array([Energy])
    else:
        En = Energy
                        ##hc has been converted into eV*Angstroms
    wls = hc/Energy  ##calculate the wavelength array in Aangstroms for layer calculations
    k0 = 2*np.pi/wls
    numwls = len(k0)
    
    #freq = 2*np.pi * c/wls #Angular frequency
    theta_exp = np.zeros((numpnts,numwls),dtype=float)
    for i in range(numwls):
        theta_exp[:,i] = np.pi/2 - np.arcsin(flatq[:]  / (2*k0[i]))
    ##Generate arrays of data for calculating transfer matrix
    ##Scalar values ~~
    ## Special cases!
    ##kx is constant for each wavelength but changes with angle
    ## Dimensionality ## 
    ## (angle, wavelength)
    kx = np.zeros((numpnts, numwls), dtype=complex)
    ky = np.zeros((numpnts, numwls), dtype=complex)
    for i,kvec in enumerate(k0):
        for j,theta in enumerate(theta_exp):
            kx[j,i] = kvec*np.sin(theta[i]) *np.cos(phi)
            ky[j,i] = kvec*np.sin(theta[i]) *np.sin(phi)
    #kx = [[k * np.sin(theta) for k in k0] for theta in theta_exp ] ##Better way to do this???
    
    ##Must be a faster way to assign these without having to next loops a thousand times
    ## Calculate the eigenvalues corresponding to kz ~~ Each one has 4 solutions
    ## Dimensionality ##
    ## (solution, #layer, angle, wavelength)
    ## Calculate the eignvectors corresponding to each kz ~~ polarization of E and H
    ## Dimensionality ##
    ## (solution, vector components, #layer, angle, wavelength)
    kz = np.zeros((4,nlayers,numpnts,numwls),dtype=complex)
    Dpol = np.zeros((4,3,nlayers,numpnts,numwls),dtype=complex) ##The polarization of the displacement field
    Hpol = np.zeros((4,3,nlayers,numpnts,numwls),dtype=complex) ##The polarization of the magnetic field
    for i in range(numwls): ##Cycle over each wavelength (energy)
        for j, theta in enumerate(theta_exp): ##Cycle over each incident angle (kx and ky)
            for k, epsilon in enumerate(tensor): #Each layer will have a different epsilon and subsequent kz
                kz[:,k,j,i] = calculate_kz(epsilon,kx[j,i],ky[j,i],k0[i])
                Dpol[:,:,k,j,i] , Hpol[:,:,k,j,i] = calculate_Dpol_Hpol(epsilon, kx[j,i], ky[j,i], kz[:,k,j,i], k0[i])
    
    ##Make matrices for the transfer matrix calculation
    ##Dimensionality ##
    ##(Matrix (4,4),#layer,angle,wavelength)
    D = np.zeros((4, 4, nlayers, numpnts, numwls),dtype=complex) ##Dynamic Matrix
    Di = np.zeros((4, 4, nlayers, numpnts, numwls),dtype=complex) ##Dynamic Matrix Inverse
    P = np.zeros((4, 4, nlayers, numpnts, numwls),dtype=complex) ## Propogation Matrix
    W = np.zeros((4, 4, nlayers, numpnts, numwls),dtype=complex) ##Nevot-Croche roughness matrix
    
    Refl = np.zeros((2,2,numpnts,numwls),dtype=float)
    Tran = np.zeros((2,2,numpnts,numwls),dtype=float)
    
    
    ##Calculate dynamic matrices
    for i in range(numwls):
        for j in range(len(theta_exp)):
            for k in range(nlayers):
                D[:,:,k,j,i], Di[:,:,k,j,i] = calculate_D(Dpol[:,:,k,j,i], Hpol[:,:,k,j,i])
                P[:,:,k,j,i] = calculate_P(kz[:,k,j,i],layers[k,0]) ##layers[k,0] is the thicknes of layer k
                W[:,:,k,j,i] = calculate_W(kz[:,k,j,i],kz[:,k-1,j,i],layers[k,3])

    ##Calculate the full system transfer matrix
    ##Dimensionality ##
    ##(Matrix (4,4),angle,wavelength)
    M = np.zeros((4,4,numpnts,numwls), dtype=complex)
    for i in range(numwls):
        for j in range(len(theta_exp)):
            M[:,:,j,i] = np.identity(4)
            for k in range(1,nlayers-1):
                M[:,:,j,i] = np.dot(M[:,:,j,i],np.dot((np.dot(Di[:,:,k-1,j,i],D[:,:,k,j,i])*W[:,:,k,j,i]) , P[:,:,k,j,i]))
            M[:,:,j,i] = np.dot(M[:,:,j,i],(np.dot(Di[:,:,-2,j,i],D[:,:,-1,j,i]) * W[:,:,-1,j,i]))
   
    ##Calculate the final outputs and organize into the appropriate waves for later
    for i in range(numwls):
        for j in range(len(theta_exp)):
            r_pp, r_ps, r_sp, r_ss, t_pp, t_ps, t_sp, t_ss = solve_transfer_matrix(M[:,:,j,i])
            Refl[0,0,j,i] = scale * np.abs(r_ss)**2 + bkg
            Refl[0,1,j,i] = scale * np.abs(r_sp)**2 + bkg
            Refl[1,0,j,i] = scale * np.abs(r_ps)**2 + bkg
            Refl[1,1,j,i] = scale * np.abs(r_pp)**2 + bkg
            Tran[0,0,j,i] = scale * np.abs(t_ss)**2 + bkg
            Tran[0,1,j,i] = scale * np.abs(t_sp)**2 + bkg
            Tran[1,0,j,i] = scale * np.abs(t_ps)**2 + bkg
            Tran[1,1,j,i] = scale * np.abs(t_pp)**2 + bkg
    if EFI:
        return [kx,ky,kz,Dpol,D, Di, P, W, Refl, Tran]
    else:
        return [Refl, Tran]


\
"""
The following functions were adapted from FSRSTools copyright Daniel Dietze ~~

   This file is part of the FSRStools python module.

   The FSRStools python module is free software: you can redistribute it and/or modify
   it under the terms of the GNU General Public License as published by
   the Free Software Foundation, either version 3 of the License, or
   (at your option) any later version.

   The FSRStools python module is distributed in the hope that it will be useful,
   but WITHOUT ANY WARRANTY; without even the implied warranty of
   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
   GNU General Public License for more details.

   You should have received a copy of the GNU General Public License
   along with the FSRStools python module. If not, see <http://www.gnu.org/licenses/>.

   Copyright 2015 Daniel Dietze <daniel.dietze@berkeley.edu>.
"""

def calculate_kz(ep,kx,ky,omega):
        """Calculate propagation constants g along z-axis for current layer.

        :param complex 3x3 tensor ep: optical tensor for the specific layer in question
        :param complex kx: In-plane propagation constant along x-axis.
        :param complex ky: In-plane propagation constant along y-axis.
        :returns: Propagation constants along z-axis. The entries are sorted such that sign( Re(g) ) has the order `+,-,+,-`. However, the polarization (p or s) may not yet be correctly assigned. This is done using :py:func:`calculate_p_q`.
        """
        w=omega
        gsigns = [1, -1, 1, -1]
        
        # set up the coefficients for the fourth-order polynomial that yields the z-propagation constant as the four roots
        # these terms are taken from my Maple calculation of the determinant of the matrix in Eq. 4 of Yeh's paper
        p = np.zeros(5, dtype=complex)
        p[0] = w**2 * ep[2, 2]
        p[1] = w**2 * ep[2, 0] * kx + ky * w**2 * ep[2, 1] + kx * w**2 * ep[0, 2] + w**2 * ep[1, 2] * ky
        p[2] = w**2 * ep[0, 0] * kx**2 + w**2 * ep[1, 0] * ky * kx - w**4 * ep[0, 0] * ep[2, 2] + w**2 * ep[1, 1] * ky**2 + w**4 * ep[1, 2] * ep[2, 1] + ky**2 * w**2 * ep[2, 2] + w**4 * ep[2, 0] * ep[0, 2] + kx**2 * w**2 * ep[2, 2] + kx * w**2 * ep[0, 1] * ky - w**4 * ep[1, 1] * ep[2, 2]
        p[3] = -w**4 * ep[0, 0] * ep[1, 2] * ky + w**2 * ep[2, 0] * kx * ky**2 - w**4 * ep[0, 0] * ky * ep[2, 1] - kx * w**4 * ep[1, 1] * ep[0, 2] + w**4 * ep[1, 0] * ky * ep[0, 2] + kx**2 * ky * w**2 * ep[1, 2] + kx**3 * w**2 * ep[0, 2] + w**4 * ep[1, 0] * ep[2, 1] * kx + ky**3 * w**2 * ep[2, 1] + kx * ky**2 * w**2 * ep[0, 2] - w**4 * ep[2, 0] * ep[1, 1] * kx + ky**3 * w**2 * ep[1, 2] + w**4 * ep[2, 0] * ep[0, 1] * ky + w**2 * ep[2, 0] * kx**3 + kx**2 * ky * w**2 * ep[2, 1] + kx * w**4 * ep[0, 1] * ep[1, 2]
        p[4] = w**6 * ep[2, 0] * ep[0, 1] * ep[1, 2] - w**6 * ep[2, 0] * ep[1, 1] * ep[0, 2] + w**4 * ep[2, 0] * kx**2 * ep[0, 2] + w**6 * ep[0, 0] * ep[1, 1] * ep[2, 2] - w**4 * ep[0, 0] * ep[1, 1] * kx**2 - w**4 * ep[0, 0] * ep[1, 1] * ky**2 - w**4 * ep[0, 0] * kx**2 * ep[2, 2] + w**2 * ep[0, 0] * kx**2 * ky**2 - w**6 * ep[0, 0] * ep[1, 2] * ep[2, 1] - ky**2 * w**4 * ep[1, 1] * ep[2, 2] + ky**2 * w**2 * ep[1, 1] * kx**2 + ky**2 * w**4 * ep[1, 2] * ep[2, 1] + w**6 * ep[1, 0] * ep[2, 1] * ep[0, 2] - w**6 * ep[1, 0] * ep[0, 1] * ep[2, 2] + w**4 * ep[1, 0] * ep[0, 1] * kx**2 + w**4 * ep[1, 0] * ep[0, 1] * ky**2 + w**2 * ep[1, 0] * kx**3 * ky + w**2 * ep[1, 0] * kx * ky**3 + kx**3 * ky * w**2 * ep[0, 1] + kx * ky**3 * w**2 * ep[0, 1] - w**4 * ep[1, 0] * kx * ky * ep[2, 2] + kx * ky * w**4 * ep[2, 1] * ep[0, 2] - kx * ky * w**4 * ep[0, 1] * ep[2, 2] + w**4 * ep[2, 0] * kx * ky * ep[1, 2] + w**2 * ep[0, 0] * kx**4 + ky**4 * w**2 * ep[1, 1]

        # the four solutions for the g's are obtained by numerically solving the polynomial equation
        # these four solutions are not yet in the right order!!
        kz_temp = np.roots(p)
        
        for i in range(3):
            mysign = np.sign(np.real(kz_temp[i]))
            if mysign != gsigns[i]:
                for j in range(i + 1, 4):
                    if mysign != np.sign(np.real(kz_temp[j])):
                        kz_temp[i], kz_temp[j] = kz_temp[j], kz_temp[i]         # swap values
                        break
        return kz_temp
        
        
def calculate_Dpol_Hpol(ep, kx, ky, kz, omega, POI = [0.,1.,0.]):
        """Calculate the electric and magnetic polarization vectors p and q for the four solutions of `kz`.

        .. versionchanged:: 02-05-2016

            Removed a bug in sorting the polarization vectors.

        :param complex 3x3 tensor ep: optical tensor for the specific layer in question
        :param complex kx: In-plane propagation constant along x-axis.
        :param complex ky: In-plane propagation constant along y-axis.
        :param complex 4-entry kz: Eigenvalues for solving characteristic equation, 4 potentially degenerate inputs
        :param float 3 component vector POI: Defines the plane of incidence through a normal vector. Used to define polarization vectors.

        :returns: Electric and magnetic polarization vectors p and q sorted according to (x+, x-, y+, y-).

        .. note:: This function also sorts the in-plane propagation constants according to their polarizations (x+, x-, y+, y-).

        .. important:: Requires prior execution of :py:func:`calculate_g`.
        """
        w = omega
        mu = 1
        c=1
        has_to_sort = False
        
        dpol_temp = np.zeros((4,3),dtype=complex)
        hpol_temp = np.zeros((4,3),dtype=complex)

        # iterate over the four solutions of the z-propagation constant kz
        for i in range(4):
            # this idea is partly based on Reider's book, as the explanation in the papers is misleading

            # define the matrix for getting the co-factors
            # use the complex conjugate to get the phases right!!
            M = np.conj(np.array([[w**2 * mu * ep[0, 0] - ky**2 - kz[i]**2, w**2 * mu * ep[0, 1] + kx * ky, w**2 * mu * ep[0, 2] + kx * kz[i]],
            [w**2 * mu * ep[0, 1] + kx * ky, w**2 * mu * ep[1, 1] - kx**2 - kz[i]**2, w**2 * mu * ep[1, 2] + ky * kz[i]],
            [w**2 * mu * ep[0, 2] + kx * kz[i], w**2 * mu * ep[1, 2] + ky * kz[i], w**2 * mu * ep[2, 2] - ky**2 - kx**2]],
            dtype=np.complex128))

            # get null space to find out which polarization is associated with g[i]
            P, s = null_space(M)

            # directions have to be calculated according to plane of incidence ( defined by (a, b, 0) and (0, 0, 1) )
            # or normal to that ( defined by (a, b, 0) x (0, 0, 1) )
            if(len(s) == 1):    # single component
                has_to_sort = True
                dpol_temp[i,:] = norm(P[0])
            else:

                if(i < 2):  # should be p pol
                    #   print("looking for p:", np.absolute(np.dot(nPOI, P[0])))
                    if(np.absolute(np.dot(POI, P[0])) < 1e-3):
                        # polarization lies in plane of incidence made up by vectors ax + by and z
                        # => P[0] is p pol
                        dpol_temp[i,:] = norm(P[0])
                    #   print("\t-> 0")
                    else:
                        # => P[1] has to be p pol
                        dpol_temp[i,:] = norm(P[1])
                    #   print("\t-> 1")
                else:       # should be s pol
                    #   print("looking for s:", np.absolute(np.dot(nPOI, P[0])))
                    if(np.absolute(np.dot(POI, P[0])) < 1e-3):
                        # polarization lies in plane of incidence made up by vectors ax + by and z
                        # => P[1] is s pol
                        dpol_temp[i,:] = norm(P[1])
                    #   print("\t-> 1")
                    else:
                        # => P[0] has to be s pol
                        dpol_temp[i,:] = norm(P[0])
                    #   print("\t-> 0")


        # if solutions were unique, sort the polarization vectors according to p and s polarization
        # the sign of Re(g) has been taken care of already
        if has_to_sort:
            for i in range(2):
                if(np.absolute(np.dot(POI, dpol_temp[i])) > 1e-3):
                    kz[i], kz[i + 2] = kz[i + 2], kz[i]
                    dpol_temp[[i, i + 2]] = dpol_temp[[i + 2, i]]                 # IMPORTANT! standard python swapping does not work for 2d numpy arrays; use advanced indexing instead

        for i in range(4):
            # select right orientation or p vector - see Born, Wolf, pp 39
            if((i == 0 and np.real(dpol_temp[i][0]) > 0.0) or (i == 1 and np.real(dpol_temp[i][0]) < 0.0) or (i >= 2 and np.real(dpol_temp[i][1]) < 0.0)):
                dpol_temp[i] *= -1.0
            # dpol_temp[i][2] = np.conj(dpol_temp[i][2])
            # calculate the corresponding q-vectors by taking the cross product between the normalized propagation constant and p[i]
            K = np.array([kx, ky, kz[i]], dtype=np.complex128)
            hpol_temp[i] = np.cross(K, dpol_temp[i]) * c / (w * mu)

        return [dpol_temp, hpol_temp]


    # calculate the dynamic matrix and its inverse
def calculate_D(Dpol, Hpol):
    """Calculate the dynamic matrix and its inverse using the previously calculated values for p and q.

    returns: :math:`D`, :math`D^{-1}`

     .. important:: Requires prior execution of :py:func:`calculate_p_q`.
    """
    D_Temp = np.zeros((4,4), dtype=complex)
    Di_Temp = np.zeros((4,4), dtype=complex)
    
    D_Temp[0, 0] = Dpol[0, 0]
    D_Temp[0, 1] = Dpol[1, 0]
    D_Temp[0, 2] = Dpol[2, 0]
    D_Temp[0, 3] = Dpol[3, 0]
    D_Temp[1, 0] = Hpol[0, 1]
    D_Temp[1, 1] = Hpol[1, 1]
    D_Temp[1, 2] = Hpol[2, 1]
    D_Temp[1, 3] = Hpol[3, 1]
    D_Temp[2, 0] = Dpol[0, 1]
    D_Temp[2, 1] = Dpol[1, 1]
    D_Temp[2, 2] = Dpol[2, 1]
    D_Temp[2, 3] = Dpol[3, 1]
    D_Temp[3, 0] = Hpol[0, 0]
    D_Temp[3, 1] = Hpol[1, 0]
    D_Temp[3, 2] = Hpol[2, 0]
    D_Temp[3, 3] = Hpol[3, 0]
    # D_Tempi = np.linalg.pinv(D_Temp, rcond=1e-20)
    Di_Temp = inv(D_Temp)

    return [D_Temp, Di_Temp]
    
    
def calculate_P(kz,d):
    """Calculate the propagation matrix using the previously calculated values for kz.
    
        :param complex 4-entry kz: Eigenvalues for solving characteristic equation, 4 potentially degenerate inputs
        :param float d: thickness of the layer in question. (units: Angstroms)
        returns: :math:`P`

    .. important:: Requires prior execution of :py:func:`calculate_kz`.
    """
    
    P_temp = np.zeros((4,4), dtype=complex)

    P_temp[:,:] = np.diag(np.exp(-1j * kz[:] * d))
    return P_temp
    
    
def calculate_W(kz1,kz2,r):
    """Calculate the roughness matrix usinfg previously caluclated values of kz for adjacent layers '1' and '2'
    
        :param complex 4-entry kz1: Eigenvalues of kz for current layer
        :param complex 4-entry kz2: Eigenvalues of kz for previous layer
        :param float r: roughness of the interface assuming a error function (units: Angstroms)
        returns: :math:`P`

    .. important:: Requires prior execution of :py:func:`calculate_kz`.
    """
    
    W_temp = np.zeros((4,4), dtype=complex)
    eplus = np.exp(-(kz1[:] + kz2[:])**2 * r**2 / 2) 
    eminus = np.exp(-(kz1[:] - kz2[:])**2 * r**2 / 2)

    W_temp[:,:] = [[eminus[0],eplus[1],eminus[2],eplus[3]],
                    [eplus[0],eminus[1],eplus[2],eminus[3]],
                    [eminus[0],eplus[1],eminus[2],eplus[3]],
                    [eplus[0],eminus[1],eplus[2],eminus[3]]
                    ]
    
    return W_temp

def null_space(A, eps=1e-4):
    """Compute the null space of matrix A which is the solution set x to the homogeneous equation Ax = 0.

    :param matrix A: Matrix.
    :param float eps: Maximum size of selected singular value relative to maximum.
    :returns: Null space (list of vectors) and associated singular components.

    .. seealso:: Wikipedia and http://stackoverflow.com/questions/1835246/how-to-solve-homogeneous-linear-equations-with-numpy.
    """
    u, s, vh = np.linalg.svd(A)
    null_space = np.compress(s <= eps * np.amax(s), vh, axis=0)
    return null_space, np.compress(s <= eps * np.amax(s), s, axis=0)


def inv(M):
    """Compute the 'exact' inverse of a 4x4 matrix using the analytical result. This should give a higher precision and speed at a reduced noise.

    :param matrix M: 4x4 Matrix.
    :returns: Inverse of this matrix or Moore-Penrose approximation if matrix cannot be inverted.

    .. seealso:: http://www.cg.info.hiroshima-cu.ac.jp/~miyazaki/knowledge/teche23.html
    """
    assert M.shape == (4, 4)

    # the following equations use algebraic indexing; transpose input matrix to get indexing right
    A = M.T
    detA = A[0, 0] * A[1, 1] * A[2, 2] * A[3, 3] + A[0, 0] * A[1, 2] * A[2, 3] * A[3, 1] + A[0, 0] * A[1, 3] * A[2, 1] * A[3, 2]
    detA = detA + A[0, 1] * A[1, 0] * A[2, 3] * A[3, 2] + A[0, 1] * A[1, 2] * A[2, 0] * A[3, 3] + A[0, 1] * A[1, 3] * A[2, 2] * A[3, 0]
    detA = detA + A[0, 2] * A[1, 0] * A[2, 1] * A[3, 3] + A[0, 2] * A[1, 1] * A[2, 3] * A[3, 0] + A[0, 2] * A[1, 3] * A[2, 0] * A[3, 1]
    detA = detA + A[0, 3] * A[1, 0] * A[2, 2] * A[3, 1] + A[0, 3] * A[1, 1] * A[2, 0] * A[3, 2] + A[0, 3] * A[1, 2] * A[2, 1] * A[3, 0]

    detA = detA - A[0, 0] * A[1, 1] * A[2, 3] * A[3, 2] - A[0, 0] * A[1, 2] * A[2, 1] * A[3, 3] - A[0, 0] * A[1, 3] * A[2, 2] * A[3, 1]
    detA = detA - A[0, 1] * A[1, 0] * A[2, 2] * A[3, 3] - A[0, 1] * A[1, 2] * A[2, 3] * A[3, 0] - A[0, 1] * A[1, 3] * A[2, 0] * A[3, 2]
    detA = detA - A[0, 2] * A[1, 0] * A[2, 3] * A[3, 1] - A[0, 2] * A[1, 1] * A[2, 0] * A[3, 3] - A[0, 2] * A[1, 3] * A[2, 1] * A[3, 0]
    detA = detA - A[0, 3] * A[1, 0] * A[2, 1] * A[3, 2] - A[0, 3] * A[1, 1] * A[2, 2] * A[3, 0] - A[0, 3] * A[1, 2] * A[2, 0] * A[3, 1]

    if detA == 0:
        return np.linalg.pinv(M)

    B = np.zeros(A.shape, dtype=np.complex128)
    B[0, 0] = A[1, 1] * A[2, 2] * A[3, 3] + A[1, 2] * A[2, 3] * A[3, 1] + A[1, 3] * A[2, 1] * A[3, 2] - A[1, 1] * A[2, 3] * A[3, 2] - A[1, 2] * A[2, 1] * A[3, 3] - A[1, 3] * A[2, 2] * A[3, 1]
    B[0, 1] = A[0, 1] * A[2, 3] * A[3, 2] + A[0, 2] * A[2, 1] * A[3, 3] + A[0, 3] * A[2, 2] * A[3, 1] - A[0, 1] * A[2, 2] * A[3, 3] - A[0, 2] * A[2, 3] * A[3, 1] - A[0, 3] * A[2, 1] * A[3, 2]
    B[0, 2] = A[0, 1] * A[1, 2] * A[3, 3] + A[0, 2] * A[1, 3] * A[3, 1] + A[0, 3] * A[1, 1] * A[3, 2] - A[0, 1] * A[1, 3] * A[3, 2] - A[0, 2] * A[1, 1] * A[3, 3] - A[0, 3] * A[1, 2] * A[3, 1]
    B[0, 3] = A[0, 1] * A[1, 3] * A[2, 2] + A[0, 2] * A[1, 1] * A[2, 3] + A[0, 3] * A[1, 2] * A[2, 1] - A[0, 1] * A[1, 2] * A[2, 3] - A[0, 2] * A[1, 3] * A[2, 1] - A[0, 3] * A[1, 1] * A[2, 2]

    B[1, 0] = A[1, 0] * A[2, 3] * A[3, 2] + A[1, 2] * A[2, 0] * A[3, 3] + A[1, 3] * A[2, 2] * A[3, 0] - A[1, 0] * A[2, 2] * A[3, 3] - A[1, 2] * A[2, 3] * A[3, 0] - A[1, 3] * A[2, 0] * A[3, 2]
    B[1, 1] = A[0, 0] * A[2, 2] * A[3, 3] + A[0, 2] * A[2, 3] * A[3, 0] + A[0, 3] * A[2, 0] * A[3, 2] - A[0, 0] * A[2, 3] * A[3, 2] - A[0, 2] * A[2, 0] * A[3, 3] - A[0, 3] * A[2, 2] * A[3, 0]
    B[1, 2] = A[0, 0] * A[1, 3] * A[3, 2] + A[0, 2] * A[1, 0] * A[3, 3] + A[0, 3] * A[1, 2] * A[3, 0] - A[0, 0] * A[1, 2] * A[3, 3] - A[0, 2] * A[1, 3] * A[3, 0] - A[0, 3] * A[1, 0] * A[3, 2]
    B[1, 3] = A[0, 0] * A[1, 2] * A[2, 3] + A[0, 2] * A[1, 3] * A[2, 0] + A[0, 3] * A[1, 0] * A[2, 2] - A[0, 0] * A[1, 3] * A[2, 2] - A[0, 2] * A[1, 0] * A[2, 3] - A[0, 3] * A[1, 2] * A[2, 0]

    B[2, 0] = A[1, 0] * A[2, 1] * A[3, 3] + A[1, 1] * A[2, 3] * A[3, 0] + A[1, 3] * A[2, 0] * A[3, 1] - A[1, 0] * A[2, 3] * A[3, 1] - A[1, 1] * A[2, 0] * A[3, 3] - A[1, 3] * A[2, 1] * A[3, 0]
    B[2, 1] = A[0, 0] * A[2, 3] * A[3, 1] + A[0, 1] * A[2, 0] * A[3, 3] + A[0, 3] * A[2, 1] * A[3, 0] - A[0, 0] * A[2, 1] * A[3, 3] - A[0, 1] * A[2, 3] * A[3, 0] - A[0, 3] * A[2, 0] * A[3, 1]
    B[2, 2] = A[0, 0] * A[1, 1] * A[3, 3] + A[0, 1] * A[1, 3] * A[3, 0] + A[0, 3] * A[1, 0] * A[3, 1] - A[0, 0] * A[1, 3] * A[3, 1] - A[0, 1] * A[1, 0] * A[3, 3] - A[0, 3] * A[1, 1] * A[3, 0]
    B[2, 3] = A[0, 0] * A[1, 3] * A[2, 1] + A[0, 1] * A[1, 0] * A[2, 3] + A[0, 3] * A[1, 1] * A[2, 0] - A[0, 0] * A[1, 1] * A[2, 3] - A[0, 1] * A[1, 3] * A[2, 0] - A[0, 3] * A[1, 0] * A[2, 1]

    B[3, 0] = A[1, 0] * A[2, 2] * A[3, 1] + A[1, 1] * A[2, 0] * A[3, 2] + A[1, 2] * A[2, 1] * A[3, 0] - A[1, 0] * A[2, 1] * A[3, 2] - A[1, 1] * A[2, 2] * A[3, 0] - A[1, 2] * A[2, 0] * A[3, 1]
    B[3, 1] = A[0, 0] * A[2, 1] * A[3, 2] + A[0, 1] * A[2, 2] * A[3, 0] + A[0, 2] * A[2, 0] * A[3, 1] - A[0, 0] * A[2, 2] * A[3, 1] - A[0, 1] * A[2, 0] * A[3, 2] - A[0, 2] * A[2, 1] * A[3, 0]
    B[3, 2] = A[0, 0] * A[1, 2] * A[3, 1] + A[0, 1] * A[1, 0] * A[3, 2] + A[0, 2] * A[1, 1] * A[3, 0] - A[0, 0] * A[1, 1] * A[3, 2] - A[0, 1] * A[1, 2] * A[3, 0] - A[0, 2] * A[1, 0] * A[3, 1]
    B[3, 3] = A[0, 0] * A[1, 1] * A[2, 2] + A[0, 1] * A[1, 2] * A[2, 0] + A[0, 2] * A[1, 0] * A[2, 1] - A[0, 0] * A[1, 2] * A[2, 1] - A[0, 1] * A[1, 0] * A[2, 2] - A[0, 2] * A[1, 1] * A[2, 0]

    return B.T / detA


# evanescent root - see Orfanidis book, Ch 7
def eroot(a):
    """Returns the evanescent root of a number, where the imaginary part has the physically correct sign.

    :param complex a: A number.
    :returns: Square root of a.
    """
    return np.where(np.real(a) < 0 and np.imag(a) == 0, -1j * np.sqrt(np.absolute(a)), np.lib.scimath.sqrt(a))


# normalize a to its length
def norm(a):
    """Normalize a vector to its length.

    :param vector a: A vector.
    :returns: Unit vector with same direction as a.
    """
    return a / np.sqrt(np.dot(a, np.conj(a)))     # use standard sqrt as argument is real and positive

"""
Functions adapted from PyATMM written by __author__ = 'Pavel Dmitriev'
"""


def solve_transfer_matrix(M):

    denom = M[0, 0]*M[2, 2] - M[0, 2]*M[2, 0]
    #print(denom)

    r_ss = (M[1, 0]*M[2, 2] - M[1, 2]*M[2, 0]) / (M[0, 0]*M[2, 2] - M[0, 2]*M[2, 0])
    r_sp = (M[3, 0]*M[2, 2] - M[3, 2]*M[2, 0]) / (M[0, 0]*M[2, 2] - M[0, 2]*M[2, 0])
    r_ps = (M[0, 0]*M[1, 2] - M[1, 0]*M[0, 2]) / (M[0, 0]*M[2, 2] - M[0, 2]*M[2, 0])
    r_pp = (M[0, 0]*M[3, 2] - M[3, 0]*M[0, 2]) / (M[0, 0]*M[2, 2] - M[0, 2]*M[2, 0])
    t_ss = M[2, 2] / (M[0, 0]*M[2, 2] - M[0, 2]*M[2, 0])
    t_sp = -M[2, 0] / (M[0, 0]*M[2, 2] - M[0, 2]*M[2, 0])
    t_ps = -M[0, 2] / (M[0, 0]*M[2, 2] - M[0, 2]*M[2, 0])
    t_pp = M[0, 0] / (M[0, 0]*M[2, 2] - M[0, 2]*M[2, 0])

    return r_ss, r_sp, r_ps, r_pp, t_ss, t_sp, t_ps, t_pp
