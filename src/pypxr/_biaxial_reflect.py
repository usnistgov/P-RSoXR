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
mu0 = 1# 4. * np.pi * 1e-7
ep0 = 1. / (c**2 * mu0)
thresh = 1e-5 # Threshold to determine if imaginary component is rounding error

    

def yeh_4x4_reflectivity(q, layers, tensor, energy, phi):
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
        contains the 1x3x3 dimensions
        First dimension may change in teh fiture to account for multi-energy
        currently it will just cycle
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
    wl = hc/energy  ##calculate the wavelength array in Aangstroms for layer calculations
    k0 = 2*np.pi/wl
    
    #Convert optical constants into dielectric tensor
    tensor = np.conj(np.eye(3) - 2*tensor[:,:,:]) # Old tensor[:,0,:,:]

    #freq = 2*np.pi * c/wls #Angular frequency
    theta_exp = np.zeros(numpnts,dtype=float)
    theta_exp = np.pi/2 - np.arcsin(flatq[:]  / (2*k0))

    ##Generate arrays of data for calculating transfer matrix
    ##Scalar values ~~
    ## Special cases!
    ##kx is constant for each wavelength but changes with angle
    ## Dimensionality ## 
    ## (angle, wavelength)
    kx = k0 * np.sin(theta_exp) * np.cos(phi)
    ky = k0 * np.sin(theta_exp) * np.sin(phi)
    
    # kx = np.zeros(numpnts, dtype=complex)
    # ky = np.zeros(numpnts, dtype=complex)
    # kx = k0 * np.sin(theta_exp) * np.cos(phi)
    # ky = k0 * np.sin(theta_exp) * np.sin(phi)
    
    ## Calculate the eigenvalues corresponding to kz ~~ Each one has 4 solutions
    ## Dimensionality ##
    ## (angle, #layer, solution)
    kz = np.zeros((numpnts,nlayers,4),dtype=complex)
    
    ## Calculate the eignvectors corresponding to each kz ~~ polarization of D and H
    ## Dimensionality ##
    ## (angle, #layers, solution, vector)
    Dpol = np.zeros((numpnts,nlayers,4,3),dtype=complex) ##The polarization of the displacement field
    Hpol = np.zeros((numpnts,nlayers,4,3),dtype=complex) ##The polarization of the magnetic field
    gamma = np.zeros((numpnts,nlayers,4,3),dtype=complex)

    #Cycle through the layers and calculate kz
    for j, epsilon in enumerate(tensor): #Each layer will have a different epsilon and subsequent kz
        kz[:,j,:], gamma[:,j,:,:] = calculate_kz_gamma(epsilon,kx,k0)
        #gamma[:,j,:,:] = calculate_gamma(epsilon, kx, kz[:,j,:])
        #Dpol[:,j,:,:], Hpol[:,j,:,:] = calculate_Dpol(epsilon, kx, ky, kz[:,j,:], k0)

    ##Make matrices for the transfer matrix calculation
    ##Dimensionality ##
    ##(angles, #layers, Matrix (4,4)
    P = calculate_P(numpnts, nlayers, k0*kz[:,:,:], layers[:,0])
    ##Nevot-Croche roughness matrix
    W = calculate_W(numpnts, nlayers, kz[:,:,:], kz[:,:,:], layers[:,3])
    ##Dynamic Matrix and inverse  
    D, Di = calculate_D(numpnts, nlayers, kx, kz[:,:,:], k0, gamma[:,:,:,:])

    ##Calculate the full system transfer matrix
    ##Dimensionality ##
    ##(angles, Matrix (4,4))
    M = np.ones((numpnts,4,4),dtype=complex)
    #Make a (numpnts x 4x4) identity matrix for the TMM - 
    M = np.einsum('...ij,ij->...ij',M,np.identity(4))
    M = calculate_TMM(numpnts,nlayers,M,D,Di,P,W)
    
    ##Calculate the final outputs and organize into the appropriate waves for later
    refl, tran = calculate_output(numpnts, M)

    return refl, tran, k0, kx, ky, kz, gamma, D, Di, P, W, M

def calculate_kz_gamma(ep, kx, k0):

    numpnts = len(kx) # number of q-points

    transmode = np.zeros((numpnts,2), dtype=int)
    reflmode = np.zeros((numpnts,2), dtype=int)
    poynting_vector = np.zeros((numpnts,3,4), dtype=complex)
    pol_condition = np.zeros((numpnts, 4), dtype=float)
    eigenvector_condition = np.zeros((numpnts, 4), dtype=float)
    
    gamma_out = np.zeros((numpnts,4,3), dtype=complex)
    kz_out = np.zeros((numpnts,4),dtype=complex) ## Outwave

    # Calculate Delta Matrix
    delta_matrix, a, m = calculate_delta_matrix(ep, kx/k0)
    k_unsorted, psi_unsorted = np.linalg.eig(delta_matrix)
    
    # Calculate Poynting vector with calculated wavefunctions
    Ex = psi_unsorted[:,0,:]
    Ey = psi_unsorted[:,2,:]
    Hx = -psi_unsorted[:,3,:]
    Hy = psi_unsorted[:,1,:]
    # Equation 17
    Ez = a[:,2,0,None]*Ex + a[:,2,1,None]*Ey + a[:,2,3,None]*Hx + a[:,2,4,None]*Hy
    # Equation 18
    Hz = a[:,5,0,None]*Ex + a[:,5,1,None]*Ey + a[:,5,3,None]*Hx + a[:,5,4,None]*Hy
    
    poynting_vector[:,0,:] = Ey*Hz - Ez*Hy
    poynting_vector[:,1,:] = Ez*Hx - Ex*Hz
    poynting_vector[:,2,:] = Ex*Hy - Ey*Hx
    
    # Calculate the sorting conditions Equation (15)
    pol_condition = np.abs(poynting_vector[:,0,:])**2/(np.abs(poynting_vector[:,0,:])**2 + np.abs(poynting_vector[:,2,:])**2)
    eigenvector_condition = np.abs(psi_unsorted[:,0,:])**2/(np.abs(psi_unsorted[:,0,:])**2 + np.abs(psi_unsorted[:,2,:])**2)
    # Temporary waves to be sorted
    temp_kvec = np.zeros((4), dtype=complex)
    temp_pol_condition = np.zeros((4), dtype=float)
    temp_wave_vector = np.zeros((4), dtype=float)
    
    for i in range(numpnts):
        kt = 0
        kr = 2
        if np.any(np.abs(np.imag(k_unsorted[i,:]))):
            for j, kvec in enumerate(k_unsorted[i,:]):
                if np.imag(kvec) >= 0:
                    temp_kvec[kt] = kvec
                    temp_pol_condition[kt] = pol_condition[i,j]
                    temp_wave_vector[kt] = eigenvector_condition[i,j]

                    kt += 1
                else:
                    temp_kvec[kr] = kvec
                    temp_pol_condition[kr] = pol_condition[i,j]
                    temp_wave_vector[kr] = eigenvector_condition[i,j]

                    kr += 1
        else:
            for j, kvec in enumerate(k_unsorted[i,:]):
                if np.real(kvec) >= 0:
                    temp_kvec[kt] = kvec
                    temp_pol_condition[kt] = pol_condition[i,j]
                    temp_wave_vector[kt] = eigenvector_condition[i,j]

                    kt += 1
                else:
                    temp_kvec[kr] = kvec
                    temp_pol_condition[kr] = pol_condition[i,j]
                    temp_wave_vector[kr] = eigenvector_condition[i,j]

                    kr += 1
                    
        # Assign outputs. Swap components if poynting vector condition
        kz_out[i,0] = temp_kvec[0] if temp_wave_vector[0] > temp_wave_vector[1] else temp_kvec[1]
        kz_out[i,1] = temp_kvec[1] if temp_wave_vector[0] > temp_wave_vector[1] else temp_kvec[0]    
        kz_out[i,2] = temp_kvec[2] if temp_wave_vector[2] > temp_wave_vector[3] else temp_kvec[3]  
        kz_out[i,3] = temp_kvec[3] if temp_wave_vector[2] > temp_wave_vector[3] else temp_kvec[2]

        gamma_out[i,:,:] = calculate_gamma(ep, kx[i]/k0, kz_out[i,:])

    gamma_norm = np.linalg.norm(gamma_out,axis=-1)
    gamma_out /= gamma_norm[:,:,None]
        
    return kz_out, gamma_out
    
    
def calculate_gamma(ep, zeta, kz):
    
    # This calculates the components in Equation 20.
    # A correction is found in the publication Erratum
    
    gamma_out = np.zeros((4,3), dtype=complex)
    #ep *= mu0 # All tensor values are the product mu0*ep
    
    gamma11 = 1.0 + 0.0j
    gamma22 = 1.0 + 0.0j
    gamma42 = 1.0 + 0.0j
    gamma31 = -1.0 + 0.0j
    
    if np.abs(kz[0] - kz[1]) < thresh: # Essentially equal
        gamma12 = 0.0 + 0.0j
        
        gamma13 = -(ep[2,0]+zeta*kz[0])
        gamma13 /= (ep[2,2]-zeta**2)
        
        gamma21 = 0.0 + 0.0j
        
        gamma23 = -ep[2,1]
        gamma23 /= (ep[2,2]-zeta**2)
        
    else:
        gamma12 = ep[1,2]*(ep[2,0]+zeta*kz[0]) - ep[1,0]*(ep[2,2]-zeta**2)
        div = (ep[2,2]-zeta**2)*(ep[1,1]-zeta**2-kz[0]**2) - ep[1,2]*ep[2,1]
        gamma12 /= div if gamma12 != 0 else 1
        
        gamma13 = -(ep[2,0]+zeta*kz[0]) - ep[2,1]*gamma12
        gamma13 /= (ep[2,2]-zeta**2)
        
        gamma21 = ep[2,1]*(ep[0,2]+zeta*kz[1]) - ep[0,1]*(ep[2,2]-zeta**2)
        div = (ep[2,2]-zeta**2)*(ep[0,0]-kz[1]**2) - (ep[0,2]+zeta*kz[1])*(ep[2,0]+zeta*kz[1])
        gamma21 /= div if gamma21 != 0 else 1
        
        gamma23 = -(ep[2,0]+zeta*kz[1])*gamma21-ep[2,1]
        gamma23 /= (ep[2,2]-zeta**2)
    
    if np.abs(kz[2] - kz[3]) < thresh: # Essentially equal
        gamma32 = 0.0 + 0.0j
        gamma33 = (ep[2,0]+zeta*kz[2])/(ep[2,2]-zeta**2)
        gamma41 = 0.0 + 0.0j
        gamma43 = -(ep[2,1])/(ep[2,2]-zeta**2)
        
    else:
        gamma32 = ep[1,0]*(ep[2,2]+zeta**2) - ep[1,2]*(ep[2,0]+zeta*kz[2])
        div = (ep[2,2]-zeta**2)*(ep[1,1]-zeta**2-kz[2]**2) - ep[1,2]*ep[2,1]
        gamma32 /= div if gamma32 != 0 else 1
        
        gamma33 = ep[2,0] + zeta*kz[2] + ep[2,1]*gamma32
        gamma33 /= (ep[2,2]-zeta**2)
        
        gamma41 = ep[2,1]*(ep[0,2]+zeta*kz[3]) - ep[0,1]*(ep[2,2]-zeta**2)
        div = (ep[2,2]-zeta**2)*(ep[0,0]-kz[3]**2)
        gamma41 /= div if gamma41 != 0 else 1
        
        gamma43 = -(ep[2,0]+zeta*kz[3])*gamma41 - ep[2,1]
        gamma43 /= (ep[2,2] - zeta**2)
        
    gamma1 = [gamma11, gamma12, gamma13]
    gamma2 = [gamma21, gamma22, gamma23]
    gamma3 = [gamma31, gamma32, gamma33]
    gamma4 = [gamma41, gamma42, gamma43]
    
    gamma_out[0,:] = np.array(gamma1)
    gamma_out[1,:] = np.array(gamma2)
    gamma_out[2,:] = np.array(gamma3)
    gamma_out[3,:] = np.array(gamma4)
        
    return gamma_out   

    
def calculate_delta_matrix(ep, zeta):

    numpnts = len(zeta)
    # Matrix elements from Equation (4) 
    m = np.zeros((6, 6), dtype=complex) # Equation (4)
    delta = np.zeros((numpnts, 4, 4), dtype=complex) # Equation (8)
    a = np.zeros((numpnts, 6, 6), dtype=complex) # Equation (9)
    # Calculate M
    m[0:3, 0:3] = ep.copy()
    m[3:6, 3:6] = mu0*np.identity(3)#mu0*np.identity(3)
    
    b = m[2,2]*m[5,5] - m[2,5]*m[5,2] # Equation 10
    a[:,2,0] = (m[None,5,0]*m[None,2,5] - m[None,2,0]*m[None,5,5])/b
    a[:,2,1] = ((m[None,5,1]-zeta)*m[None,2,5] - m[None,2,1]*m[None,5,5])/b
    a[:,2,3] = (m[None,5,3]*m[None,2,5] - m[None,2,3]*m[None,5,5])/b
    a[:,2,4] = (m[None,5,4]*m[None,2,5] - (m[None,2,4]+zeta)*m[5,5])/b
    a[:,5,0] = (m[None,5,2]*m[None,2,0] - m[None,2,2]*m[None,5,0])/b
    a[:,5,1] = (m[None,5,2]*m[None,2,1] - m[None,2,2]*(m[None,5,1]-zeta))/b
    a[:,5,3] = (m[None,5,2]*m[None,2,3] - m[None,2,2]*m[None,5,3])/b
    a[:,5,4] = (m[None,5,2]*(m[None,2,4]+zeta) - m[None,2,2]*m[None,5,4])/b

    delta[:,0,0] = m[None,4,0] + (m[None,4,2]+zeta)*a[:,2,0] + m[None,4,5]*a[:,5,0]
    delta[:,0,1] = m[None,4,4] + (m[None,4,2]+zeta)*a[:,2,4] + m[None,4,5]*a[:,5,4]
    delta[:,0,2] = m[None,4,1] + (m[None,4,2]+zeta)*a[:,2,1] + m[None,4,5]*a[:,5,1]
    delta[:,0,3] = -1*(m[None,4,3] + (m[None,4,2]+zeta)*a[:,2,3] + m[None,4,5]*a[:,5,3])
    delta[:,1,0] = m[None,0,0] + m[None,0,2]*a[:,2,0] + m[None,0,5]*a[:,5,0]
    delta[:,1,1] = m[None,0,4] + m[None,0,2]*a[:,2,4] + m[None,0,5]*a[:,5,4]
    delta[:,1,2] = m[None,0,1] + m[None,0,2]*a[:,2,1] + m[None,0,5]*a[:,5,1]
    delta[:,1,3] = -1*(m[None,0,3] + m[None,0,2]*a[:,2,3] + m[None,0,5]*a[:,5,3])
    delta[:,2,0] = -1*(m[None,3,0] + m[None,3,2]*a[:,2,0] + m[None,3,5]*a[:,5,0])
    delta[:,2,1] = -1*(m[None,3,4] + m[None,3,2]*a[:,2,4] + m[None,3,5]*a[:,5,4])
    delta[:,2,2] = -1*(m[None,3,1] + m[None,3,2]*a[:,2,1] + m[None,3,5]*a[:,5,1])
    delta[:,2,3] = m[None,3,3] + m[None,3,2]*a[:,2,3] + m[None,3,5]*a[:,5,3]
    delta[:,3,0] = m[None,1,0] + m[None,1,2]*a[:,2,0] + (m[None,1,5]-zeta)*a[:,5,0]
    delta[:,3,1] = m[None,1,4] + m[None,1,2]*a[:,2,4] + (m[None,1,5]-zeta)*a[:,5,4]
    delta[:,3,2] = m[None,1,1] + m[None,1,2]*a[:,2,1] + (m[None,1,5]-zeta)*a[:,5,1]
    delta[:,3,3] = -1*(m[None,1,3] + m[None,1,2]*a[:,2,3] + (m[None,1,5]-zeta)*a[:,5,3])
    
    return delta, a, m
    
    
def calculate_P(numpnts, nlayers, kz, d):
    """
    Calculate the propagation matrix using the previously calculated values for kz.
    
        :param complex 4-entry kz: Eigenvalues for solving characteristic equation, 4 potentially degenerate inputs
        :param float d: thickness of the layer in question. (units: Angstroms)
        returns: :math:`P`

    .. important:: Requires prior execution of :py:func:`calculate_kz`.
    """

    #Create the diagonal components in the propogation matrix
    #Cast into a 4x4 version through redundent broadcasting
    diagonal_components = np.exp(-1j * kz[:,:,:,None] * d[None,:,None,None]) 
    #Element by element multiplication with the identity over each q-point
    P_temp = np.einsum('...jk,jk->...jk',diagonal_components,np.identity(4)) 
    
    return P_temp
    
def calculate_W(numpnts, nlayers, kz1, kz2, r):
    """
    Calculate the roughness matrix usinfg previously caluclated values of kz for adjacent layers '1' and '2'
    
        :param complex 4-entry kz1: Eigenvalues of kz for current layer
        :param complex 4-entry kz2: Eigenvalues of kz for previous layer
        :param float r: roughness of the interface assuming an error function (units: Angstroms)
        returns: :math:`P`

    .. important:: Requires prior execution of :py:func:`calculate_kz`.
    """
        
    W_temp = np.zeros((numpnts, nlayers, 4,4), dtype=np.complex_)
    eplus = np.zeros((numpnts, nlayers, 4), dtype=np.complex_)
    eminus = np.zeros((numpnts, nlayers, 4), dtype=np.complex_)
    """ 
    for i in range(numpnts):
        for j in range(nlayers):
            eplus[i,j,:] = np.exp(-(kz1[i, j, :] + kz2[i, j-1, :])**2 * r[j]**2 / 2) 
            eminus[i,j,:] = np.exp(-(kz1[i, j, :] - kz2[i, j-1, :])**2 * r[j]**2 / 2)
    """
    kz2 = np.roll(kz2, 1, axis=1) #Reindex to allow broadcasting in the next step....see commented loop
    #for j in range(nlayers):
    eplus[:,:,:] = np.exp(-(kz1[:, :, :] + kz2[:, :, :])**2 * r[None,:,None]**2 / 2) 
    eminus[:,:,:] = np.exp(-(kz1[:, :, :] - kz2[:, :, :])**2 * r[None,:,None]**2 / 2)
    """
    W_temp[:,:] = [[eminus[0],eplus[1],eminus[2],eplus[3]],
                    [eplus[0],eminus[1],eplus[2],eminus[3]],
                    [eminus[0],eplus[1],eminus[2],eplus[3]],
                    [eplus[0],eminus[1],eplus[2],eminus[3]]
                    ]
    """
    W_temp[:, :, 0, 0] = 1#eminus[:, :, 0]
    W_temp[:, :, 0, 1] = 1#eplus[:, :, 1]
    W_temp[:, :, 0, 2] = 1#eminus[:, :, 2]
    W_temp[:, :, 0, 3] = 1#eplus[:, :, 3]
    W_temp[:, :, 1, 0] = 1#eplus[:, :, 0]
    W_temp[:, :, 1, 1] = 1#eminus[:, :, 1]
    W_temp[:, :, 1, 2] = 1#eplus[:, :, 2]
    W_temp[:, :, 1, 3] = 1#eminus[:, :, 3]
    W_temp[:, :, 2, 0] = 1#eminus[:, :, 0]
    W_temp[:, :, 2, 1] = 1#eplus[:, :, 1]
    W_temp[:, :, 2, 2] = 1#eminus[:, :, 2]
    W_temp[:, :, 2, 3] = 1#eplus[:, :, 3]
    W_temp[:, :, 3, 0] = 1#eplus[:, :, 0]
    W_temp[:, :, 3, 1] = 1#eminus[:, :, 1]
    W_temp[:, :, 3, 2] = 1#eplus[:, :, 2]
    W_temp[:, :, 3, 3] = 1#eminus[:, :, 3]
    
    return W_temp
    
def calculate_D(numpnts, nlayers, kx, kz, k0, gamma):
    
    #mu0 = 1
    d_Temp = np.zeros((numpnts, nlayers, 4, 4), dtype=complex)
    di_temp = np.zeros((numpnts, nlayers, 4, 4), dtype=complex)
    
    d_Temp[:, :, 0, 0] = gamma[:, :, 0, 0]
    d_Temp[:, :, 0, 1] = gamma[:, :, 1, 0]
    d_Temp[:, :, 0, 2] = gamma[:, :, 2, 0]
    d_Temp[:, :, 0, 3] = gamma[:, :, 3, 0]
    d_Temp[:, :, 1, 0] = gamma[:, :, 0, 1]
    d_Temp[:, :, 1, 1] = gamma[:, :, 1, 1]
    d_Temp[:, :, 1, 2] = gamma[:, :, 2, 1]
    d_Temp[:, :, 1, 3] = gamma[:, :, 3, 1]
    d_Temp[:, :, 2, 0] = (1/mu0)*(kz[:,:,0]*gamma[:,:,0,0] - (kx[:,None]/k0)*gamma[:,:,0,2])
    d_Temp[:, :, 2, 1] = (1/mu0)*(kz[:,:,1]*gamma[:,:,1,0] - (kx[:,None]/k0)*gamma[:,:,1,2])
    d_Temp[:, :, 2, 2] = (1/mu0)*(kz[:,:,2]*gamma[:,:,2,0] - (kx[:,None]/k0)*gamma[:,:,2,2])
    d_Temp[:, :, 2, 3] = (1/mu0)*(kz[:,:,3]*gamma[:,:,3,0] - (kx[:,None]/k0)*gamma[:,:,3,2])
    d_Temp[:, :, 3, 0] = (1/mu0)*kz[:,:,0]*gamma[:,:,0,1]
    d_Temp[:, :, 3, 1] = (1/mu0)*kz[:,:,1]*gamma[:,:,1,1]
    d_Temp[:, :, 3, 2] = (1/mu0)*kz[:,:,2]*gamma[:,:,2,1]
    d_Temp[:, :, 3, 3] = (1/mu0)*kz[:,:,3]*gamma[:,:,3,1]
    
    d_Temp[:,:,0,:] = gamma[:,:,:,0].copy()
    d_Temp[:,:,1,:] = gamma[:,:,:,1].copy()
    d_Temp[:,:,2,:] = (kz[:,:,:]*gamma[:,:,:,0])-(kx[:,None,None]/k0)*gamma[:,:,:,2]
    d_Temp[:,:,3,:] = kz[:,:,:]*gamma[:,:,:,1]
    """
    for i in range(numpnts):
        for j in range(nlayers):
            Di_Temp[i,j,:,:] = np.linalg.pinv(D_Temp[i,j,:,:])
    """
    #Try running an a matrix inversion for the tranfer matrix.
    #If it fails, run a pseudo-inverse
    #Update 07/07/2021: I don't think the uniaxial calculation will error...changing pinv to inv
    #                   for default calculation
    try:
        di_Temp = np.linalg.inv(d_Temp) #Broadcasted along the 'numpnts' dimension
    except LinAlgError:
        di_Temp = np.linalg.pinv(d_Temp)

    return [d_Temp, di_Temp]
    
def calculate_TMM(numpnts,nlayers,M,D,Di,P,W):
    for j in range(1,nlayers-1):
        A = np.einsum('...ij,...jk ->...ik',Di[:,j-1,:,:], D[:,j,:,:])
        B = A*W[:,j,:,:]
        C = np.einsum('...ij,...jk ->...ik',B, P[:,j,:,:])
        M[:,:,:] = np.einsum('...ij,...jk ->...ik',M[:,:,:], C)
    AA = np.einsum('...ij,...jk ->...ik',Di[:,-2,:,:], D[:,-1,:,:])
    BB = AA*W[:,-1,:,:]
    M[:,:,:] = np.einsum('...ij,...jk ->...ik',M[:,:,:], BB)
    return M

    
def calculate_output(numpnts, M_full):

    refl = np.zeros((numpnts,2,2),dtype=np.float_)
    tran = np.zeros((numpnts,2,2),dtype=np.complex_)
    
    convert_to_yeh = np.array([[1,0,0,0],
                               [0,0,1,0],
                               [0,1,0,0],
                               [0,0,0,1]])                            
    convert_to_yeh_invert = np.linalg.inv(convert_to_yeh)
    
    M = np.einsum('...ij,...jk,...kl',convert_to_yeh_invert, M_full, convert_to_yeh)
    
    denom = M[:,0,0]*M[:,2,2] - M[:,0,2]*M[:,2,0]
    r_pp = (M[:,1,0]*M[:,2,2] - M[:,1,2]*M[:,2,0]) / denom#(M[:,0, 0]*M[2, 2] - M[0, 2]*M[2, 0])
    r_ps = (M[:,3,0]*M[:,2,2] - M[:,3,2]*M[:,2,0]) / denom#(M[0, 0]*M[2, 2] - M[0, 2]*M[2, 0])
    r_sp = (M[:,0,0]*M[:,1,2] - M[:,1,0]*M[:,0,2]) / denom #(M[0, 0]*M[2, 2] - M[0, 2]*M[2, 0])
    r_ss = (M[:,0,0]*M[:,3,2] - M[:,3,0]*M[:,0,2]) / denom #(M[0, 0]*M[2, 2] - M[0, 2]*M[2, 0])
    t_pp = M[:,2,2] / denom#(M[0, 0]*M[2, 2] - M[0, 2]*M[2, 0])
    t_ps = -M[:,2,0] / denom#(M[0, 0]*M[2, 2] - M[0, 2]*M[2, 0])
    t_sp = -M[:,0,2] / denom#(M[0, 0]*M[2, 2] - M[0, 2]*M[2, 0])
    t_ss = M[:,0,0] / denom#(M[0, 0]*M[2, 2] - M[0, 2]*M[2, 0])


    #r_pp, r_ps, r_sp, r_ss, t_pp, t_ps, t_sp, t_ss = solve_transfer_matrix(M[i,:,:])
    refl[:,0,0] = np.abs(r_ss)**2
    refl[:,0,1] = np.abs(r_sp)**2
    refl[:,1,0] = np.abs(r_ps)**2
    refl[:,1,1] = np.abs(r_pp)**2
    tran[:,0,0] = t_ss
    tran[:,0,1] = t_sp
    tran[:,1,0] = t_ps
    tran[:,1,1] = t_pp
    
    return refl, tran
    
def calculate_TMM(numpnts,nlayers,M,D,Di,P,W):
    for j in range(1,nlayers-1):
        A = np.einsum('...ij,...jk ->...ik',Di[:,j-1,:,:], D[:,j,:,:])
        B = A*W[:,j,:,:]
        C = np.einsum('...ij,...jk ->...ik',B, P[:,j,:,:])
        M[:,:,:] = np.einsum('...ij,...jk ->...ik',M[:,:,:], C)
    AA = np.einsum('...ij,...jk ->...ik',Di[:,-2,:,:], D[:,-1,:,:])
    BB = AA*W[:,-1,:,:]
    M[:,:,:] = np.einsum('...ij,...jk ->...ik',M[:,:,:], BB)
    return M
    
    