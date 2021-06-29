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
#from numba import njit, complex128
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
TINY = 1e-30
    

def uniaxial_reflectivity(q, layers, tensor, energy, phi, scale=1., bkg=0, threads=0,save_components=None):
    """
    EMpy implementation of the uniaxial 4x4 matrix formalism for calculating reflectivity from a stratified
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
    #Plane of incidence - required to define polarization vectors
    OpticAxis = np.array([0.,0.,1.])

    ##Organize qvals into proper order
    qvals = np.asfarray(q)
    flatq = qvals.ravel()
    numpnts = flatq.size #Number of q-points
    
    ##Grab the number of layers
    nlayers = layers.shape[0]
   
                        ##hc has been converted into eV*Angstroms
    wl = hc/energy  ##calculate the wavelength array in Aangstroms for layer calculations
    k0 = 2*np.pi/(wl)
    
    #Convert optical constants into dielectric tensor
    tensor = np.conj(np.eye(3) - 2*tensor[:,:,:]) #From tensor[:,0,:,:]
    
    
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
    ky = np.zeros(numpnts, dtype=complex) #Used to keep the k vectors three components later on for cross / dot products
    kx = k0 * np.sin(theta_exp) * np.cos(phi)
    #ky = k0 * np.sin(theta_exp) * np.sin(phi)

    ## Calculate the eigenvalues corresponding to kz ~~ Each one has 4 solutions
    ## Dimensionality ##
    ## (angle, #layer, solution)
    kz = np.zeros((numpnts,nlayers,4),dtype=complex)
    
    ## Calculate the eignvectors corresponding to each kz ~~ polarization of D and H
    ## Dimensionality ##
    ## (angle, #layers, solution, vector)
    Dpol = np.zeros((numpnts,nlayers,4,3),dtype=complex) ##The polarization of the displacement field
    Hpol = np.zeros((numpnts,nlayers,4,3),dtype=complex) ##The polarization of the magnetic field

    #Cycle through the layers and calculate kz
    for j, epsilon in enumerate(tensor): #Each layer will have a different epsilon and subsequent kz
        kz[:,j,:] = calculate_kz_uni(epsilon,kx,ky,k0,opticaxis=OpticAxis)
        Dpol[:,j,:,:], Hpol[:,j,:,:] = calculate_Dpol_uni(epsilon, kx, ky, kz[:,j,:], k0,opticaxis=OpticAxis)

    ##Make matrices for the transfer matrix calculation
    ##Dimensionality ##
    ##(Matrix (4,4),#layer,angle,wavelength)
    D = np.zeros((numpnts,nlayers, 4, 4),dtype=complex) ##Dynamic Matrix
    Di = np.zeros((numpnts,nlayers, 4, 4),dtype=complex) ##Dynamic Matrix Inverse
    P = np.zeros((numpnts,nlayers, 4, 4),dtype=complex) ## Propogation Matrix
    W = np.zeros((numpnts,nlayers, 4, 4),dtype=complex) ##Nevot-Croche roughness matrix
    
    Refl = np.zeros((numpnts,2,2),dtype=float)
    Tran = np.zeros((numpnts,2,2),dtype=complex)
    
    #calculate the propagation matrices
    P[:,:,:,:] = calculate_P(numpnts, nlayers, kz[:,:,:], layers[:,0]) ##layers[k,0] is the thicknes of layer k
    #calculate the roughness matrices
    W[:,:,:,:] = calculate_W(numpnts, nlayers, kz[:,:,:], kz[:,:,:], layers[:,3])
    #calculate the Dynamical matrices  
    D[:,:,:,:], Di[:,:,:,:] = calculate_D(numpnts,nlayers,Dpol[:,:,:,:], Hpol[:,:,:,:])
    
    ##Calculate the full system transfer matrix
    ##Dimensionality ##
    ##(Matrix (4,4),wavelength)
    M = np.ones((numpnts,4,4),dtype=complex)
    #Make a (numpnts x 4x4) identity matrix for the TMM - 
    M = np.einsum('...ij,ij->...ij',M,np.identity(4))
    M = calulate_TMM(numpnts,nlayers,M,D,Di,P,W)
    ##Calculate the final outputs and organize into the appropriate waves for later
    Refl, Tran = calculate_output(numpnts, scale, bkg, M)

    if save_components:
        return (kx, ky, kz, Dpol,Hpol, D, Di, P, W, Refl, Tran)
    else:
        return [Refl, Tran]

"""
The following functions were adapted from PyATMM copyright Pavel Dmitriev
"""
def calculate_kz_uni(ep, kx, ky, k0, opticaxis=([0., 1., 0.])):

    #Calculate ordinary and extraordinary components from the tensor

    e_o = ep[0,0]
    e_e = ep[2,2]
    nu = (e_e - e_o) / e_o #intermediate birefringence from reference
    k_par = np.sqrt(kx**2 + ky**2) #Magnitude of parallel component
    #l = [kx/k_par, ky/k_par, 0]
    
    kz_ord = np.zeros(len(kx), dtype=np.complex_)
    kz_extraord = np.zeros(len(kx), dtype=np.complex_)
    kz_out = np.zeros((len(kx),4), dtype=np.complex_)

    
    #n = [0, 0, 1] #Normal vector
    #if not numpy.isclose(k_par, 0):
    #    l = [kx/k_par, ky/k_par, 0]
    #    assert numpy.isclose(numpy.dot(l, l), 1)
    #else:
    #    l = [0, 0, 0]

    #Dot product between optical axis and vector normal and perpindicular component
    na = 1 #numpy.dot(n, opticAxis)
    la = 0 #numpy.dot(l, opticAxis)

    kz_ord = np.sqrt(e_o * k0**2 - k_par[:]**2)#, dtype=np.complex128)

    kz_extraord = (1 / (1 + nu * na**2)) * (-nu * k_par[:] * na*la
                                                + np.sqrt(e_o * k0**2 * (1 + nu) * (1 + nu * na**2)
                                                            - k_par[:]**2 * (1 + nu * (la**2 + na**2))))

    kz_out[:,2] = kz_ord 
    kz_out[:,3] = -kz_ord
    kz_out[:,0] = kz_extraord
    kz_out[:,1] = -kz_extraord
    return kz_out

def calculate_Dpol_uni(ep, kx, ky, kz, k0, opticaxis=([0., 1., 0.])):

    # For now optic axis should be aligned to main axes
    #assert numpy.allclose(opticAxis, [0, 0, 1]) \
    #       or numpy.allclose(opticAxis, [0, 1, 0]) \
    #       or numpy.allclose(opticAxis, [1, 0, 0])
    # In general, as long as k-vector and optic axis are not colinear, this should work
    
    #assert all(not np.allclose(opticaxis, [kx, ky, np.abs(g)]) for g in kz)
    #assert np.isclose(np.dot(opticaxis, opticaxis), 1.)

    e_o = ep[0,0]
    e_e = ep[2,2]
    nu = (e_e - e_o) / e_o #intermediate birefringence from reference

    kvec = np.zeros((len(kx), 4,3), dtype=np.complex_)
    kdiv = np.zeros((len(kx), 4), dtype=np.complex_)
    dpol_temp = np.zeros((len(kx), 4, 3), dtype=np.complex_)
    hpol_temp = np.zeros((len(kx), 4, 3), dtype=np.complex_)
    
    #create k-vector
    kvec[:,:,0] = kx[:,None]
    kvec[:,:,1] = ky[:,None]
    kvec[:,:,2] = kz

    #'normalize' k-vector not the magnitude norm...some confusion here
    #for i in range(len(kx)):
    #    for j in range(4):
    #        kdiv[i,j] = np.sqrt(np.dot(kvec[i,j],kvec[i,j]))
    
    kdiv = np.sqrt(np.einsum('ijk,ijk->ij',kvec,kvec)) #Performs the commented out dot product calculation
    
    knorm = kvec / kdiv [:,:,None]#(np.linalg.norm(kvec,axis=-1)[:,:,None])
    
    #calc propogation of k along optical axis
    kpol = np.dot(knorm,opticaxis)

    #kap = [numpy.asarray([kx, ky, g]) for g in kz]
    #kap = [numpy.divide(kap_i, numpy.sqrt(numpy.dot(kap_i, kap_i))) for kap_i in kap]
    #ka = [numpy.dot(opticAxis, kap_i) for kap_i in kap]

    dpol_temp[:,2,:] = np.cross(opticaxis[None, :], knorm[:, 2, :])
    dpol_temp[:,3,:] = np.cross(opticaxis[None, :], knorm[:, 3, :])
    dpol_temp[:,0,:] = np.subtract(opticaxis[None, :], ((1 + nu)/(1+nu*kpol[:, 0, None]**2))*kpol[:, 0, None] * knorm[:, 0, :])
    dpol_temp[:,1,:] = np.subtract(opticaxis[None, :], ((1 + nu)/(1+nu*kpol[:, 1, None]**2))*kpol[:, 1, None] * knorm[:, 1, :])

       
    dpol_norm = np.linalg.norm(dpol_temp,axis=-1)
    dpol_temp /= dpol_norm[:,:,None]
    hpol_temp = np.cross(kvec, dpol_temp) * (1/k0)
    
    return dpol_temp, hpol_temp


# calculate the dynamic matrix and its inverse  
def calculate_D(numpnts,nlayers, Dpol, Hpol):
    """Calculate the dynamic matrix and its inverse using the previously calculated values for p and q.

    returns: :math:`D`, :math`D^{-1}`

     .. important:: Requires prior execution of :py:func:`calculate_p_q`.
    """
    D_Temp = np.zeros((numpnts,nlayers,4,4), dtype=np.complex_)
    Di_Temp = np.zeros((numpnts,nlayers,4,4), dtype=np.complex_)
    
    D_Temp[:, :, 0, 0] = Dpol[:, :, 0, 0]
    D_Temp[:, :, 0, 1] = Dpol[:, :, 1, 0]
    D_Temp[:, :, 0, 2] = Dpol[:, :, 2, 0]
    D_Temp[:, :, 0, 3] = Dpol[:, :, 3, 0]
    D_Temp[:, :, 1, 0] = Hpol[:, :, 0, 1]
    D_Temp[:, :, 1, 1] = Hpol[:, :, 1, 1]
    D_Temp[:, :, 1, 2] = Hpol[:, :, 2, 1]
    D_Temp[:, :, 1, 3] = Hpol[:, :, 3, 1]
    D_Temp[:, :, 2, 0] = Dpol[:, :, 0, 1]
    D_Temp[:, :, 2, 1] = Dpol[:, :, 1, 1]
    D_Temp[:, :, 2, 2] = Dpol[:, :, 2, 1]
    D_Temp[:, :, 2, 3] = Dpol[:, :, 3, 1]
    D_Temp[:, :, 3, 0] = Hpol[:, :, 0, 0]
    D_Temp[:, :, 3, 1] = Hpol[:, :, 1, 0]
    D_Temp[:, :, 3, 2] = Hpol[:, :, 2, 0]
    D_Temp[:, :, 3, 3] = Hpol[:, :, 3, 0]
    
    """
    for i in range(numpnts):
        for j in range(nlayers):
            Di_Temp[i,j,:,:] = np.linalg.pinv(D_Temp[i,j,:,:])
    """
    
    Di_Temp = np.linalg.pinv(D_Temp)

    return [D_Temp, Di_Temp]

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
    W_temp[:, :, 0, 0] = eminus[:, :, 0]
    W_temp[:, :, 0, 1] = eplus[:, :, 1]
    W_temp[:, :, 0, 2] = eminus[:, :, 2]
    W_temp[:, :, 0, 3] = eplus[:, :, 3]
    W_temp[:, :, 1, 0] = eplus[:, :, 0]
    W_temp[:, :, 1, 1] = eminus[:, :, 1]
    W_temp[:, :, 1, 2] = eplus[:, :, 2]
    W_temp[:, :, 1, 3] = eminus[:, :, 3]
    W_temp[:, :, 2, 0] = eminus[:, :, 0]
    W_temp[:, :, 2, 1] = eplus[:, :, 1]
    W_temp[:, :, 2, 2] = eminus[:, :, 2]
    W_temp[:, :, 2, 3] = eplus[:, :, 3]
    W_temp[:, :, 3, 0] = eplus[:, :, 0]
    W_temp[:, :, 3, 1] = eminus[:, :, 1]
    W_temp[:, :,3, 2] = eplus[:, :, 2]
    W_temp[:, :, 3, 3] = eminus[:, :, 3]
    
    return W_temp

#Unable to calculate the iterative TMM through broadcasting...Numba decorator provides performance increase.
"""
def calulate_TMM(numpnts,nlayers,M,D,Di,P,W):
    
    for i in range(numpnts):
        for j in range(1,nlayers-1):
            M[i,:,:] = np.dot(M[i,:,:],np.dot((np.dot(Di[i,j-1,:,:],D[i,j,:,:])*W[i,j,:,:]) , P[i,j,:,:]))
        M[i,:,:] = np.dot(M[i,:,:],(np.dot(Di[i,-2,:,:],D[i,-1,:,:]) * W[i,-1,:,:]))
    return M
"""

#Iterative dot product over each element in the first axis. Equivalent to using Numba on the above calculation
##np.einsum('...ij,...jk ->...ik',A[:,j-1,:,:], B[:,j,:,:])
#Does not result in errors and is cleaner in general.

def calulate_TMM(numpnts,nlayers,M,D,Di,P,W):
    for j in range(1,nlayers-1):
        A = np.einsum('...ij,...jk ->...ik',Di[:,j-1,:,:], D[:,j,:,:])
        B = A*W[:,j,:,:]
        C = np.einsum('...ij,...jk ->...ik',B, P[:,j,:,:])
        M[:,:,:] = np.einsum('...ij,...jk ->...ik',M[:,:,:], C)
    AA = np.einsum('...ij,...jk ->...ik',Di[:,-2,:,:], D[:,-1,:,:])
    BB = AA*W[:,-1,:,:]
    M[:,:,:] = np.einsum('...ij,...jk ->...ik',M[:,:,:], BB)
    return M

def calculate_output(numpnts, scale, bkg, M_full):

    Refl = np.zeros((numpnts,2,2),dtype=np.float_)
    Tran = np.zeros((numpnts,2,2),dtype=np.complex_)
    
    #for i in range(numpnts):
    M = M_full #[i,:,:]
    
    denom = M[:,0,0]*M[:,2,2] - M[:,0,2]*M[:,2,0]
    r_ss = (M[:,1,0]*M[:,2,2] - M[:,1,2]*M[:,2,0]) / denom#(M[:,0, 0]*M[2, 2] - M[0, 2]*M[2, 0])
    r_sp = (M[:,3,0]*M[:,2,2] - M[:,3,2]*M[:,2,0]) / denom#(M[0, 0]*M[2, 2] - M[0, 2]*M[2, 0])
    r_ps = (M[:,0,0]*M[:,1,2] - M[:,1,0]*M[:,0,2]) / denom #(M[0, 0]*M[2, 2] - M[0, 2]*M[2, 0])
    r_pp = (M[:,0,0]*M[:,3,2] - M[:,3,0]*M[:,0,2]) / denom #(M[0, 0]*M[2, 2] - M[0, 2]*M[2, 0])
    t_ss = M[:,2,2] / denom#(M[0, 0]*M[2, 2] - M[0, 2]*M[2, 0])
    t_sp = -M[:,2,0] / denom#(M[0, 0]*M[2, 2] - M[0, 2]*M[2, 0])
    t_ps = -M[:,0,2] / denom#(M[0, 0]*M[2, 2] - M[0, 2]*M[2, 0])
    t_pp = M[:,0,0] / denom#(M[0, 0]*M[2, 2] - M[0, 2]*M[2, 0])


    #r_pp, r_ps, r_sp, r_ss, t_pp, t_ps, t_sp, t_ss = solve_transfer_matrix(M[i,:,:])
    Refl[:,0,0] = scale * np.abs(r_ss)**2 + bkg
    Refl[:,0,1] = scale * np.abs(r_sp)**2 + bkg
    Refl[:,1,0] = scale * np.abs(r_ps)**2 + bkg
    Refl[:,1,1] = scale * np.abs(r_pp)**2 + bkg
    Tran[:,0,0] = t_ss #scale * np.abs(t_ss)**2 + bkg
    Tran[:,0,1] = t_sp #scale * np.abs(t_sp)**2 + bkg
    Tran[:,1,0] = t_ps #scale * np.abs(t_ps)**2 + bkg
    Tran[:,1,1] = t_pp #scale * np.abs(t_pp)**2 + bkg
        
    return Refl, Tran
 