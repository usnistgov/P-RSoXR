# -*- coding: utf-8 -*-

import numpy as np
import os
#from refnx.ani_reflect.ani_structure import ani_NexafsSLD


class NexafsDataset(object):
    #Class for loading NEXAFS data for use in reflecivity modeling
        
    def __init__(self, data=None):
        self.filename=None
        self.name=None
        
        self._en = np.zeros(0)
        self._delta = np.zeros(0)
        self._beta = np.zeros(0)
        
        self.isAnisotropic = False
        
        ##tensor components if needed
        self._xx = np.zeros(0)
        self._ixx = np.zeros(0)
        self._yy = np.zeros(0)
        self._iyy = np.zeros(0)        
        self._zz = np.zeros(0)
        self._izz = np.zeros(0)
                
        #Try to load the data
        if hasattr(data, 'read') or type(data) is str: #borrowed from refnx to check if data is a file
            self.load(data)        
        elif isinstance(data, NexafsDataset): #potentially copy a dataset
            self.name = data.name
            self.filename = data.name
            self._en = data._en
            self._delta = data._delta
            self._beta = data._beta
            self._tensor = data._tensor
        elif data is not None:
            if len(data) == 7: ##Assume that the data describes a tensor
                self._en = np.array(data[0],dtype = float)
                self._xx = np.array(data[1],dtype = float)
                self._ixx = np.array(data[2],dtype = float)
                self._yy = np.array(data[3],dtype = float)
                self._iyy = np.array(data[4],dtype = float)       
                self._zz = np.array(data[5],dtype = float)
                self._izz = np.array(data[6],dtype = float)
                
                self._tensor = np.array([[self._xx + 1j*self._ixx, 0, 0],
                                   [0, self._yy + 1j*self._iyy, 0],
                                   [0, 0, self._zz + 1j*self._izz]]
                                   ,dtype=complex)
                
                self._delta = np.array(np.trace(np.real(self._tensor)),dtype=float)
                self._beta = np.array(np.trace(np.imag(self._tensor)),dtype=float)
                
                self.isAnisotropic = True

                
            elif len(data) == 3: ##Assume that the data describes a uniaxial material
                self._en = np.array(data[0],dtype=float)
                self._delta = np.array(data[1],dtype=float)
                self._beta = np.array(data[2], dtype=float)
                
                self.tensor = np.eye(3) * (self._delta[:,None,None] + 1j*self._beta[:,None,None])
                
            else:
                raise RuntimeError('Length of data is confusing')
    
    def __len__(self):
        """
        return the number of datapoints in the set
        
        """
        return len(self._en)  
        
    @property
    def en(self):
        return self._en
    @property
    def beta(self):
        return self._beta
    
    @property
    def delta(self):
        return self._delta
        
    @property   
    def tensor(self):
        return self._tensor             
        
    def load(self, file):
    
        self.filename = os.path.abspath(file)
        self.name = os.path.splitext(os.path.basename(file))[0]
        
        ##Not sure if the file contains optical constants or a full dielectric tensor
        Tempload = np.loadtxt(file)
        Tempload = np.rollaxis(Tempload,1,0)
        numcols = len(Tempload)
        
        #Guess the type of data given a number of columns
        if numcols == 3:
            self._en = np.array(Tempload[0],dtype=float)
            self._delta = np.array(Tempload[1],dtype=float)
            self._beta = np.array(Tempload[2],dtype=float)
            
            self._tensor = np.eye(3) * (self._delta[:,None,None] + 1j*self._beta[:,None,None])

            
        elif numcols == 7:
            self._en = np.array(Tempload[0],dtype = float)
            self._xx = np.array(Tempload[1],dtype = float)
            self._ixx = np.array(Tempload[2],dtype = float)
            self._yy = np.array(Tempload[3],dtype = float)
            self._iyy = np.array(Tempload[4],dtype = float)  
            self._zz = np.array(Tempload[5],dtype = float)
            self._izz = np.array(Tempload[6],dtype = float)
            
            self._tensor  = np.array([[1,0,0],[0,0,0],[0,0,0]]) * (self._xx[:,None,None] + 1j*self._ixx[:,None,None])
            self._tensor += np.array([[0,0,0],[0,1,0],[0,0,0]]) * (self._yy[:,None,None] + 1j*self._iyy[:,None,None])
            self._tensor += np.array([[0,0,0],[0,0,0],[0,0,1]]) * (self._zz[:,None,None] + 1j*self._izz[:,None,None])
                
            self._delta = np.trace(np.real(self._tensor),axis1=1,axis2=2)
            self._beta = np.trace(np.imag(self._tensor),axis1=1,axis2=2)
    
            self.isAnisotropic = True

            
        
        
    def plot(self, fig=None,autoformat=False):
    
        import matplotlib.pyplot as plt
        
        if fig is None:
            fig = plt.figure()
            ax = fig.add_subplot(111)
            ax.set_title(self.name)
        else:
            ax = fig.gca()
        
        ax.plot(self.en, self.delta, label='delta')
        ax.plot(self.en, self.beta, label='beta')
        
        if autoformat:
            plt.xlim(280,320)
            plt.ylim(-0.01,0.01)
            plt.xlabel('Photon Energy [eV]')
            plt.ylabel('optical constant')
            plt.legend()

        
        return fig, ax 
    
    
    
    #def __call__(self, energy=250, name=''):
    #    return ani_NexafsSLD(self, energy=energy, name=name)
        