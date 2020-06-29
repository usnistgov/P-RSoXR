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
from numba import jit, complex128
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
    

def yeh_4x4_reflectivity(q, layers, tensor, Energy, phi, scale=1., bkg=0, threads=0,save_components=None):
    """
    EMpy implementation of the biaxial 4x4 matrix formalism for calculating reflectivity from a stratified
    medium.
    
    Uses the implementation developed by FSRStools - https://github.com/ddietze/FSRStools - written by Daniel Dietze
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
        
    tensor: np.ndarray
        contains the 3x3 dimensions
    scale: float
        Multiply all reflectivities by this value.
    bkg: float
        Linear background to be added to all reflectivities
    threads: int, optional
        <THIS OPTION IS CURRENTLY IGNORED>


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
   
                        ##hc has been converted into eV*Angstroms
    wl = hc/Energy  ##calculate the wavelength array in Aangstroms for layer calculations
    k0 = 2*np.pi/wl
    
    tensor = np.conj(np.eye(3) - 2*tensor[:])
    
    #freq = 2*np.pi * c/wls #Angular frequency
    theta_exp = np.zeros(numpnts,dtype=float)
    theta_exp = np.pi/2 - np.arcsin(flatq[:]  / (2*k0))

    ##Generate arrays of data for calculating transfer matrix
    ##Scalar values ~~
    ## Special cases!
    ##kx is constant for each wavelength but changes with angle
    ## Dimensionality ## 
    ## (angle, wavelength)
    kx = np.zeros(numpnts, dtype=complex)
    ky = np.zeros(numpnts, dtype=complex)
    kx = k0 * np.sin(theta_exp) * np.cos(phi)
    ky = k0 * np.sin(theta_exp) * np.sin(phi)

    ##Must be a faster way to assign these without having to next loops a thousand times
    ## Calculate the eigenvalues corresponding to kz ~~ Each one has 4 solutions
    ## Dimensionality ##
    ## (solution, #layer, angle, wavelength)
    kz = np.zeros((numpnts,nlayers,4),dtype=complex)
    K = np.zeros((numpnts,nlayers,4,3),dtype=complex)
    
    ## Calculate the eignvectors corresponding to each kz ~~ polarization of E and H
    ## Dimensionality ##
    ## (solution, vector components, #layer, angle, wavelength)
    Dpol = np.zeros((numpnts,nlayers,4,3),dtype=complex) ##The polarization of the displacement field
    Hpol = np.zeros((numpnts,nlayers,4,3),dtype=complex) ##The polarization of the magnetic field

    #kz = calculate_kz(tensor[None],kx[:,None,None],ky[:,None,None],k0)
    #kz[:,:,:] = kz_vecfunc(tensor[None,:],kx[:,None],ky[:,None],k0)
    #print('True kz -', kz)

    #for i, theta in enumerate(theta_exp): ##Cycle over each incident angle (kx and ky)
    #    for j, epsilon in enumerate(tensor): #Each layer will have a different epsilon and subsequent kz
    #        kz[i,j,:] = calculate_kz(epsilon,kx[i],ky[i],k0)
    #        Dpol[i,j,:,:], Hpol[i,j,:,:] = calculate_Dpol(epsilon, kx[i], ky[i], kz[i,j,:], k0)
    for j, epsilon in enumerate(tensor): #Each layer will have a different epsilon and subsequent kz
        kz[:,j,:] = calculate_kz(epsilon,kx,ky,k0)
        Dpol[:,j,:,:], Hpol[:,j,:,:] = calculate_Dpol_q(epsilon, kx, ky, kz[:,j,:], k0)
    #K = np.array([kx,ky,kz],dtype=complex)
    #Hpol = np.cross(K,Dpol) / k0
    #for i in range(4):
        #for j, ep in enumerate(tensor):
    #    K = np.array((kx[:],ky[:],kz[:,:,i]),dtype=complex)
    #    Hpol[:,:,i,:] = np.cross(K[:,:].T,Dpol[:,:,i,:])
    
    
    #            K = np.array([kx, ky, kz[i]], dtype=np.complex128)
    #        hpol_temp[i] = np.cross(K, dpol_temp[i]) * c / (w * mu)
    #Hpol K
    ##Make matrices for the transfer matrix calculation
    ##Dimensionality ##
    ##(Matrix (4,4),#layer,angle,wavelength)
    D = np.zeros((numpnts,nlayers, 4, 4),dtype=complex) ##Dynamic Matrix
    Di = np.zeros((numpnts,nlayers, 4, 4),dtype=complex) ##Dynamic Matrix Inverse
    P = np.zeros((numpnts,nlayers, 4, 4),dtype=complex) ## Propogation Matrix
    W = np.zeros((numpnts,nlayers, 4, 4),dtype=complex) ##Nevot-Croche roughness matrix
    
    Refl = np.zeros((numpnts,2,2),dtype=float)
    Tran = np.zeros((numpnts,2,2),dtype=float)
    
    
    ##Calculate dynamic matrices
    for i in range(numpnts):
        for j in range(nlayers):
            D[i,j,:,:], Di[i,j,:,:] = calculate_D(Dpol[i,j,:,:], Hpol[i,j,:,:])
            P[i,j,:,:] = calculate_P(kz[i,j,:],layers[j,0]) ##layers[k,0] is the thicknes of layer k
            W[i,j,:,:] = calculate_W(kz[i,j,:],kz[i,j-1,:],layers[j,3])

    ##Calculate the full system transfer matrix
    ##Dimensionality ##
    ##(Matrix (4,4),angle,wavelength)
    M = np.zeros((numpnts,4,4), dtype=complex)
    for i in range(len(theta_exp)):
        M[i,:,:] = np.identity(4)
        for j in range(1,nlayers-1):
            M[i,:,:] = np.dot(M[i,:,:],np.dot((np.dot(Di[i,j-1,:,:],D[i,j,:,:])*W[i,j,:,:]) , P[i,j,:,:]))
        M[i,:,:] = np.dot(M[i,:,:],(np.dot(Di[i,-2,:,:],D[i,-1,:,:]) * W[i,-1,:,:]))
   
    ##Calculate the final outputs and organize into the appropriate waves for later
    for i in range(len(theta_exp)):
        r_pp, r_ps, r_sp, r_ss, t_pp, t_ps, t_sp, t_ss = solve_transfer_matrix(M[i,:,:])
        Refl[i,0,0] = scale * np.abs(r_ss)**2 + bkg
        Refl[i,0,1] = scale * np.abs(r_sp)**2 + bkg
        Refl[i,1,0] = scale * np.abs(r_ps)**2 + bkg
        Refl[i,1,1] = scale * np.abs(r_pp)**2 + bkg
        Tran[i,0,0] = scale * np.abs(t_ss)**2 + bkg
        Tran[i,0,1] = scale * np.abs(t_sp)**2 + bkg
        Tran[i,1,0] = scale * np.abs(t_ps)**2 + bkg
        Tran[i,1,1] = scale * np.abs(t_pp)**2 + bkg

    if save_components:
        return (kx, ky, kz, Dpol, D, Di, P, W, Refl, Tran)
    else:
        return [Refl, Tran]



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
def calculate_kz(ep, kx, ky, w):

    """Calculate propagation constants g along z-axis for current layer.

    :param complex 3x3 tensor ep: optical tensor for the specific layer in question
    :param complex kx: In-plane propagation constant along x-axis
        Can be a 1D array for reflectivity
    :param complex ky: In-plane propagation constant along y-axis.
        Can be a 1D array for reflectivity
        
    :returns: Propagation constants along z-axis.
        """
    gsigns = [1, -1, 1, -1]
    
    kz_out = np.zeros((len(kx),4),dtype=complex)
       
    for k in range(len(kx)):
        kz_temp = solv_kz(ep,kx[k],ky[k],w)
        for i in range(3):
            mysign = np.sign(np.real(kz_temp[i]))
            if mysign != gsigns[i]:
                for j in range(i + 1, 4):
                    if mysign != np.sign(np.real(kz_temp[j])):
                        kz_temp[i], kz_temp[j] = kz_temp[j], kz_temp[i]         # swap values
                        break
        kz_out[k,:] = kz_temp
    
    return kz_out
    
@jit(complex128[:](complex128[:,:],complex128,complex128,complex128))          
def solv_kz(ep,kx,ky,w):
    p0 = w**2 * ep[2, 2]
    p1 = w**2 * ep[2, 0] * kx + ky * w**2 * ep[2, 1] + kx * w**2 * ep[0, 2] + w**2 * ep[1, 2] * ky
    p2 = w**2 * ep[0, 0] * kx**2 + w**2 * ep[1, 0] * ky * kx - w**4 * ep[0, 0] * ep[2, 2] + w**2 * ep[1, 1] * ky**2 + w**4 * ep[1, 2] * ep[2, 1] + ky**2 * w**2 * ep[2, 2] + w**4 * ep[2, 0] * ep[0, 2] + kx**2 * w**2 * ep[2, 2] + kx * w**2 * ep[0, 1] * ky - w**4 * ep[1, 1] * ep[2, 2]
    p3 = -w**4 * ep[0, 0] * ep[1, 2] * ky + w**2 * ep[2, 0] * kx * ky**2 - w**4 * ep[0, 0] * ky * ep[2, 1] - kx * w**4 * ep[1, 1] * ep[0, 2] + w**4 * ep[1, 0] * ky * ep[0, 2] + kx**2 * ky * w**2 * ep[1, 2] + kx**3 * w**2 * ep[0, 2] + w**4 * ep[1, 0] * ep[2, 1] * kx + ky**3 * w**2 * ep[2, 1] + kx * ky**2 * w**2 * ep[0, 2] - w**4 * ep[2, 0] * ep[1, 1] * kx + ky**3 * w**2 * ep[1, 2] + w**4 * ep[2, 0] * ep[0, 1] * ky + w**2 * ep[2, 0] * kx**3 + kx**2 * ky * w**2 * ep[2, 1] + kx * w**4 * ep[0, 1] * ep[1, 2]
    p4 = w**6 * ep[2, 0] * ep[0, 1] * ep[1, 2] - w**6 * ep[2, 0] * ep[1, 1] * ep[0, 2] + w**4 * ep[2, 0] * kx**2 * ep[0, 2] + w**6 * ep[0, 0] * ep[1, 1] * ep[2, 2] - w**4 * ep[0, 0] * ep[1, 1] * kx**2 - w**4 * ep[0, 0] * ep[1, 1] * ky**2 - w**4 * ep[0, 0] * kx**2 * ep[2, 2] + w**2 * ep[0, 0] * kx**2 * ky**2 - w**6 * ep[0, 0] * ep[1, 2] * ep[2, 1] - ky**2 * w**4 * ep[1, 1] * ep[2, 2] + ky**2 * w**2 * ep[1, 1] * kx**2 + ky**2 * w**4 * ep[1, 2] * ep[2, 1] + w**6 * ep[1, 0] * ep[2, 1] * ep[0, 2] - w**6 * ep[1, 0] * ep[0, 1] * ep[2, 2] + w**4 * ep[1, 0] * ep[0, 1] * kx**2 + w**4 * ep[1, 0] * ep[0, 1] * ky**2 + w**2 * ep[1, 0] * kx**3 * ky + w**2 * ep[1, 0] * kx * ky**3 + kx**3 * ky * w**2 * ep[0, 1] + kx * ky**3 * w**2 * ep[0, 1] - w**4 * ep[1, 0] * kx * ky * ep[2, 2] + kx * ky * w**4 * ep[2, 1] * ep[0, 2] - kx * ky * w**4 * ep[0, 1] * ep[2, 2] + w**4 * ep[2, 0] * kx * ky * ep[1, 2] + w**2 * ep[0, 0] * kx**4 + ky**4 * w**2 * ep[1, 1]

    return np.roots(np.array([p0,p1,p2,p3,p4]))

def calculate_kz_old(ep,kx,ky,omega):
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
        """
        #PREVIOUS METHOD BY DIETZE
        p = np.zeros(5, dtype=complex)
        p[0] = w**2 * ep[2, 2]
        p[1] = w**2 * ep[2, 0] * kx + ky * w**2 * ep[2, 1] + kx * w**2 * ep[0, 2] + w**2 * ep[1, 2] * ky
        p[2] = w**2 * ep[0, 0] * kx**2 + w**2 * ep[1, 0] * ky * kx - w**4 * ep[0, 0] * ep[2, 2] + w**2 * ep[1, 1] * ky**2 + w**4 * ep[1, 2] * ep[2, 1] + ky**2 * w**2 * ep[2, 2] + w**4 * ep[2, 0] * ep[0, 2] + kx**2 * w**2 * ep[2, 2] + kx * w**2 * ep[0, 1] * ky - w**4 * ep[1, 1] * ep[2, 2]
        p[3] = -w**4 * ep[0, 0] * ep[1, 2] * ky + w**2 * ep[2, 0] * kx * ky**2 - w**4 * ep[0, 0] * ky * ep[2, 1] - kx * w**4 * ep[1, 1] * ep[0, 2] + w**4 * ep[1, 0] * ky * ep[0, 2] + kx**2 * ky * w**2 * ep[1, 2] + kx**3 * w**2 * ep[0, 2] + w**4 * ep[1, 0] * ep[2, 1] * kx + ky**3 * w**2 * ep[2, 1] + kx * ky**2 * w**2 * ep[0, 2] - w**4 * ep[2, 0] * ep[1, 1] * kx + ky**3 * w**2 * ep[1, 2] + w**4 * ep[2, 0] * ep[0, 1] * ky + w**2 * ep[2, 0] * kx**3 + kx**2 * ky * w**2 * ep[2, 1] + kx * w**4 * ep[0, 1] * ep[1, 2]
        p[4] = w**6 * ep[2, 0] * ep[0, 1] * ep[1, 2] - w**6 * ep[2, 0] * ep[1, 1] * ep[0, 2] + w**4 * ep[2, 0] * kx**2 * ep[0, 2] + w**6 * ep[0, 0] * ep[1, 1] * ep[2, 2] - w**4 * ep[0, 0] * ep[1, 1] * kx**2 - w**4 * ep[0, 0] * ep[1, 1] * ky**2 - w**4 * ep[0, 0] * kx**2 * ep[2, 2] + w**2 * ep[0, 0] * kx**2 * ky**2 - w**6 * ep[0, 0] * ep[1, 2] * ep[2, 1] - ky**2 * w**4 * ep[1, 1] * ep[2, 2] + ky**2 * w**2 * ep[1, 1] * kx**2 + ky**2 * w**4 * ep[1, 2] * ep[2, 1] + w**6 * ep[1, 0] * ep[2, 1] * ep[0, 2] - w**6 * ep[1, 0] * ep[0, 1] * ep[2, 2] + w**4 * ep[1, 0] * ep[0, 1] * kx**2 + w**4 * ep[1, 0] * ep[0, 1] * ky**2 + w**2 * ep[1, 0] * kx**3 * ky + w**2 * ep[1, 0] * kx * ky**3 + kx**3 * ky * w**2 * ep[0, 1] + kx * ky**3 * w**2 * ep[0, 1] - w**4 * ep[1, 0] * kx * ky * ep[2, 2] + kx * ky * w**4 * ep[2, 1] * ep[0, 2] - kx * ky * w**4 * ep[0, 1] * ep[2, 2] + w**4 * ep[2, 0] * kx * ky * ep[1, 2] + w**2 * ep[0, 0] * kx**4 + ky**4 * w**2 * ep[1, 1]

        # the four solutions for the g's are obtained by numerically solving the polynomial equation
        # these four solutions are not yet in the right order!!
        kz_temp = np.roots(p)
        """
        kz_temp = solve_kz_vec(ep[0,0],ep[1,1],ep[2,2],kx,ky,omega)
        
        for i in range(3):
            mysign = np.sign(np.real(kz_temp[i]))
            if mysign != gsigns[i]:
                for j in range(i + 1, 4):
                    if mysign != np.sign(np.real(kz_temp[j])):
                        kz_temp[i], kz_temp[j] = kz_temp[j], kz_temp[i]         # swap values
                        break
        return kz_temp

@jit(complex128[:](complex128,complex128,complex128,complex128,complex128,complex128))        
def solve_kz_vec(exx, eyy, ezz, kx, ky, k0):
    #p = np.zeros(5, dtype=complex)
    p0 = k0**2 * ezz
    p2 = k0**2 * exx * kx**2 - k0**4 * exx * ezz + k0**2 * eyy * ky**2 + ky**2 * k0**2 * ezz + kx**2 * k0**2 * ezz - k0**4 * eyy * ezz
    p4 = k0**6 * exx * eyy * ezz - k0**4 * exx * eyy * kx**2 - k0**4 * exx * eyy * ky**2 - k0**4 * exx * kx**2 * ezz + k0**2 * exx * kx**2 * ky**2 - ky**2 * k0**4 * eyy * ezz + ky**2 * k0**2 * eyy * kx**2 + k0**2 * exx * kx**4 + ky**4 * k0**2 * eyy

    return np.roots(np.array([p0,0,p2,0,p4]))
    
kz_vecfunc = np.vectorize(calculate_kz,signature='(n,n),(),(),()->(m)')
                
def calculate_Dpol(ep, kx, ky, kz, omega, POI = [0.,1.,0.]):
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
            dtype=np.complex))
            # get null space to find out which polarization is associated with g[i]
            P, s = null_space(M)
            print(P)
            print('')
            print(s)

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
            
        return [dpol_temp,hpol_temp]
        
def calculate_Dpol_q(ep, kx, ky, kz, w, POI = [0.,1.,0.]):
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
        mu = 1
        c=1
        numpnts =kz.shape[0]

        has_to_sort = np.full(numpnts,True)
        eps = 1e-4

        dpol_temp = np.zeros((numpnts,4,3),dtype=complex)
        hpol_temp = np.zeros((numpnts,4,3),dtype=complex)

        # iterate over the four solutions of the z-propagation constant kz
        for i in range(4):
            # this idea is partly based on Reider's book, as the explanation in the papers is misleading

            # define the matrix for getting the co-factors
            # use the complex conjugate to get the phases right!!
            M = np.rollaxis(np.conj(np.array([[w**2 * mu * ep[0, 0] - ky[:]**2 - kz[:,i]**2, w**2 * mu * ep[0, 1] + kx[:] * ky[:], w**2 * mu * ep[0, 2] + kx[:] * kz[:,i]],
            [w**2 * mu * ep[0, 1] + kx[:] * ky[:], w**2 * mu * ep[1, 1] - kx[:]**2 - kz[:,i]**2, w**2 * mu * ep[1, 2] + ky[:] * kz[:,i]],
            [w**2 * mu * ep[0, 2] + kx[:] * kz[:,i], w**2 * mu * ep[1, 2] + ky[:] * kz[:,i], w**2 * mu * ep[2, 2] - ky[:]**2 - kx[:]**2]],
            dtype=np.complex)),2,0)
            # get null space to find out which polarization is associated with g[i]
            u, s, vh = np.linalg.svd(M)

            for j in range(numpnts):
                P = np.compress(s[j] <= eps * np.amax(s[j]), vh[j], axis=0)
                # directions have to be calculated according to plane of incidence ( defined by (a, b, 0) and (0, 0, 1) )
                # or normal to that ( defined by (a, b, 0) x (0, 0, 1) )
                if(P.shape[0] == 1):    # single component
                    has_to_sort[j] = True
                    dpol_temp[j,i,:] = P[0]/np.linalg.norm(P[0])
                else:
                    if(i < 2):  # should be p pol
                        #   print("looking for p:", np.absolute(np.dot(nPOI, P[0])))
                        if(np.absolute(np.dot(POI, P[0])) < 1e-3):
                        # polarization lies in plane of incidence made up by vectors ax + by and z
                        # => P[0] is p pol
                            dpol_temp[j,i,:] = P[0]/np.linalg.norm(P[0])
                    #   print("\t-> 0")
                        else:
                        # => P[1] has to be p pol
                            dpol_temp[j,i,:] = P[1]/np.linalg.norm(P[1])
                    #   print("\t-> 1")
                    else:       # should be s pol
                    #   print("looking for s:", np.absolute(np.dot(nPOI, P[0])))
                        if(np.absolute(np.dot(POI, P[0])) < 1e-3):
                        # polarization lies in plane of incidence made up by vectors ax + by and z
                        # => P[1] is s pol
                            dpol_temp[j,i,:] = P[1]/np.linalg.norm(P[1])
                    #   print("\t-> 1")
                        else:
                        # => P[0] has to be s pol
                            dpol_temp[j,i,:] = P[0]/np.linalg.norm(P[0])
                    #   print("\t-> 0")


        # if solutions were unique, sort the polarization vectors according to p and s polarization
        # the sign of Re(g) has been taken care of already
        for j in range(numpnts):
            if has_to_sort[j]:
                for i in range(2):
                    if(np.absolute(np.dot(POI, dpol_temp[j,i])) > 1e-3):
                        kz[j,i], kz[j,i + 2] = kz[j,i + 2], kz[j,i]
                        dpol_temp[j,[i, i + 2]] = dpol_temp[j, [i + 2, i]]                 # IMPORTANT! standard python swapping does not work for 2d numpy arrays; use advanced indexing instead

        for i in range(4):
            # select right orientation or p vector - see Born, Wolf, pp 39
            if((i == 0 and any(np.real(dpol_temp[:,i][0])) > 0.0) or (i == 1 and any(np.real(dpol_temp[:,i][0])) < 0.0) or (i >= 2 and any(np.real(dpol_temp[:,i][1])) < 0.0)):
                dpol_temp[:,i] *= -1.0
            # dpol_temp[i][2] = np.conj(dpol_temp[i][2])
            # calculate the corresponding q-vectors by taking the cross product between the normalized propagation constant and p[i]
        for i in range(4):
            K = np.rollaxis(np.array([kx, ky, kz[:,i]], dtype=np.complex128),1,0)
            hpol_temp[:,i,:] = np.cross(K, dpol_temp[:,i,:]) * c / (w * mu)
            
        return [dpol_temp,hpol_temp]


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
    
    #try:
    Di_Temp = np.linalg.pinv(D_Temp)
    #except:
    #    Di_Temp = 0#np.eye(4)#np.linalg.inv(D_temp.round(5))
        ##Reject the solution
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
