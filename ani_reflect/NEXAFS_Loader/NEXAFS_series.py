# -*- coding: utf-8 -*-

import os

import numpy as np
import xarray as xr
import pandas as pd
import warnings
from refnx.ani_reflect.Data1D_series import Data1D_series
import kkcalc.data as kkdata
import kkcalc.kk as kk
   

class NexafsSeries(Data1D_series):

    def __init__(self, data=None,vary_coord=[None]):
        self._coords = ['angle', 'energy']
        self._experiment_name='nexafs'
        
        super(NexafsSeries, self).__init__(data=data,
                                           coords=self._coords,
                                           vary_coord=vary_coord,
                                           experiment_name=self._experiment_name)
                                           
                                           
class NEXAFS(object):

    def __init__(self, input_data, 
                        chemformula='C',
                        density=1.2,
                        datatype='photoabsorption',
                        splice_min=280,
                        splice_max=320,
                        init_kk=True,
                        slope_constant=None,
                        gamma = -1):
                        
        #Generate Datasets to be compiled into raw data
        if type(input_data) is xr.Dataset:
            self.input_data = input_data
        elif type(input_data) is NexafsSeries:
            self.input_data = input_data.data
        self.nexafsseries = input_data
        #self.input_data = input_data.data #Specifically grab the dataset from the input NEXAFSSeries Object
        self.coord0 = 'angle'#input_data._coords[0] #Grab the specific label for coord 0
        self.coord1 = 'energy'#input_data._coords[1] #Grab the specific label for coord 1
        
        #Conversions from input data to be stored as .data
        self._asf = None #conversion to atomic scattering factor from input data
        self._delta = None
        self._beta = None
        self._bareatom = None
        #polynomial coefficients for calculating delta
        self._poly_asf = []

        #Other inputs
        self.init_kk = init_kk
        self.slope_constant = slope_constant #slope from Stohr 9.17
        self.gamma = xr.DataArray(np.array([gamma]), [('gamma', np.array([gamma]))]) #Molecular tilt of surface from nexafs - goofy format for xr.broadcast function
        self.density = density #Material density
        self.datatype = datatype #Input type of data ['photoabsorption', 'ASF', 'beta']
        self.splice_eV = np.array([splice_min, splice_max])

        #run initial calculations given inputs
        if 'gamma' not in self.input_data.coords:
            self.input_data, _ = xr.broadcast(self.input_data, self.gamma )

        self.chemformula = chemformula #build chemical information and calculate ASF from the input data

        #Other things
        self._birefringence = None
        self._dichroism = None

    #Full set of Data including the imported NEXAFS, ASF, Delta, beta for the original energy range
    @property
    def data(self):
        return self._data 
        
    @data.setter
    def data(self, data):
        if type(data) is xr.Dataset:
            self._data = data
        else:
            data_filled = [d for d in data if d]
            self._data = xr.merge(data_filled)  
                
    @property
    def asf(self):
        return self.data.asf
        
    @asf.setter
    def asf(self, dataset):
        self._asf = dataset
        #From new asf generate beta
        self._beta, _ = self.process_data(self._asf, 'beta')
        self.data = (self.input_data, self._asf, self._bareatom, self._delta, self._beta, self._poly_asf)
        if(self.init_kk):
            self.delta = self.kkcalc(self.data)
    @property
    def delta(self):
        if self._delta is None:
            self.delta = self.kkcalc(self.data)
        return self.data.delta
    
    @delta.setter
    def delta(self, delta_input):
        self._delta = delta_input
        self.data = (self.input_data, self._asf, self._bareatom, self._delta, self._beta, self._poly_asf)
    
    @property
    def beta(self):
        return self.data.beta
    
    @property
    def birefringence(self):
        return self.data.beta.isel(angle=0) - self.data.beta.isel(angle=-1)
        
    @property
    def dichroism(self):    
        return self.data.delta.isel(angle=-1) - self.data.delta.isel(angle=0) #convention....
    
    @property
    def chemformula(self):
        return self._chemformula
        
    @chemformula.setter
    def chemformula(self, formula):
        #Setup up basic info on input formula
        self._chemformula = formula #chemical formula with same notation as QANT
        self.stoichiometry = kkdata.ParseChemicalFormula(self._chemformula) #calculate array to parse formula
        self.stitch_threshold = sum([Z*count for Z, count in self.stoichiometry]) #Scale factor for stitching polynomials together
        self.formula_mass = kkdata.calculate_FormulaMass(self.stoichiometry) #calculate molar mass
        self.relative_correction = kk.calc_relativistic_correction(self.stoichiometry) #calculate relativistic correction to KK
        #Calculate the bare atom spectra
        self.ASF_E, self.ASF_Data = self.calculate_bareatom(self.stoichiometry) #calculate the Henke data from chemical formula                                    
        #Convert the data into atomic scattering factor 'asf' by scaling to bare atom at given splice points.
        self._poly_asf, self.asf = self.process_data(self.input_data, 'asf') #convert the input data into asf
        
    def extrapolate_oc(self,gamma_out=None):
        #Generate output gamma values
        if gamma_out and self.slope_constant:
            gamma_cos = 3*(np.cos(np.radians(gamma_out)))**2 - 1
            gamma_slope = -self.slope_constant/4 * gamma_cos
            gamma_slope = gamma_slope/ (3*(np.cos(np.radians(self.gamma)))**2 - 1) 
        else:
            gamma_out = np.array([-1])
            gamma_slope = np.array([1])
            
        gamma_xr = xr.DataArray(gamma_slope, [('gamma', gamma_out)])
        #Create new angle coord. that is from Stohr 9.17
        temp_beta = self.data.beta.assign_coords(angle=(3*(np.cos(np.radians(self.data.angle)))**2 - 1))
        #Calculate the experimentally determined fit parameters for every energy
        linear_fit = temp_beta.polyfit(dim='angle', deg=1) #Fit to a line
        #Rescale the slope based on the desired gamma values for the output to contain
        gamma_xr, _ = xr.broadcast(gamma_xr, linear_fit) #Match dimensions through explicit broadcasting
        #Temp conversion until I figure out how to do this properly....
        temp = gamma_xr.isel(degree=0) * linear_fit.isel(degree=0)
        linear_fit_scale = xr.concat([temp,linear_fit.isel(degree=1)],dim='degree')
        
        #This is pretty dumb here...I think it may be a bug in xr.polyval -- but this seems to work for the calculated output angles
        extrapolated_angle = xr.DataArray(np.linspace(2,-1,2), dims=['angle'], coords={'angle' : np.linspace(2,-1,2)})
        
        output = xr.polyval(coord=extrapolated_angle.angle, coeffs=linear_fit_scale).rename({'polyfit_coefficients' : 'nexafs'}) #extrapolate to zero
        return NEXAFS(output, chemformula=self.chemformula, density=self.density, datatype='beta', splice_min=self.splice_eV[0], splice_max=self.splice_eV[1])
        
    def kkcalc(self, data_in):
        compile_output = []
        en_data = data_in.coords[self.coord1].values
        for gamma in data_in.coords['gamma'].values:
            compile_delta = []
            data = data_in.sel(gamma=gamma)
            for i, angle in enumerate(data.coords[self.coord0]):
                trim_poly = data.poly_asf.values[i][:-1,:] #Reverse the fact that I need equal elements for xarray but not kkcalc.....
                temp_delta_stitched = kk.KK_PP(data.coords['energy_s'].values, data.coords['energy_s'].values,
                                                trim_poly, self.relative_correction)
                temp_delta = np.interp(en_data, data.coords['energy_s'].values, temp_delta_stitched)
                temp_input = np.rollaxis(np.array([en_data,temp_delta]),-1)
                temp_delta = kkdata.convert_Beta_to_ASF(temp_input,density=self.density,formula_mass=self.formula_mass,stoichiometry=self.stoichiometry,reverse=True)
                compile_delta.append(xr.Dataset({'delta': ([self.coord0, self.coord1], [temp_delta[:,1]]),
                                                },
                                                coords={
                                                self.coord0: [angle.values],
                                                self.coord1: self.data.coords[self.coord1]
                                                },        
                                                )
                                            )
                
            compile_output.append(xr.concat(compile_delta, dim=self.coord0, join='outer'))
        return xr.concat(compile_output, dim='gamma', join='outer')
    """
    
    Calculate the bare atom spectrum and store when given a molecule    
    
    """    
    def calculate_bareatom(self, stoichiometry):
        ASF_E , ASF_Data = kkdata.calculate_asf(stoichiometry)
        temp_e, temp_ba = kkdata.coeffs_to_linear(ASF_E, ASF_Data, self.stitch_threshold) #build full dataset
        temp_input = np.rollaxis(np.concatenate((np.array([temp_e]),np.array([temp_ba]))),-1) 
        temp_output2D = kkdata.convert_Beta_to_ASF(temp_input,
                                                    density=self.density,
                                                    formula_mass=self.formula_mass,
                                                    stoichiometry=self.stoichiometry,
                                                    reverse=True)
                                        
        temp_output = np.interp(self.input_data.coords[self.coord1].values, temp_output2D[:,0], temp_output2D[:,1])
        temp_energy = self.input_data.coords[self.coord1]
        self._bareatom = xr.Dataset({'bareatom' : (self.coord1, temp_output)
                                    },
                                    coords={
                                    self.coord1: temp_energy                           
                                    }
                                    )
        temp_output = temp_output2D[:,1]
        temp_energy = temp_output2D[:,0]
        self._bareatom_stitch = xr.Dataset({'bareatom' : (self.coord1, temp_output)
                                        },
                                        coords={
                                        self.coord1: temp_energy                          
                                        })
        return ASF_E, ASF_Data
    """
    
    Switchboard to convert data between raw nexafs, asf, and beta. 
    Annoying structured to convert between using kkcalc and xarray.
    
    """   
    def process_data(self, data_in, calc):#, datatype=self.datatype):
        compile_output = []
        compile_poly = []
        datatype=self.datatype         
        for gamma in data_in.coords['gamma'].values:
            data = data_in.sel(gamma=gamma)
            compile_result = []
            poly_result = []
            for angle in data.coords[self.coord0]:
                temp_input = xr_to_array(data.sel(angle=angle.values), self.coord1, reverse=True)
                
                if calc == 'asf':
                    temp_output = kkdata.convert_data(temp_input, datatype, 'ASF', Density=self.density, Formula_Mass=self.formula_mass)
                    Full_E, Full_Coeffs, temp_output, temp_splice_points = kkdata.merge_spectra(temp_output,
                                                                                                self.ASF_E,
                                                                                                self.ASF_Data,
                                                                                                merge_points=self.splice_eV,
                                                                                                add_background=False,
                                                                                                fix_distortions=False,
                                                                                                plotting_extras=True)
                    #This is a stupid append that is required for kkcalc...
                    poly_stitch = np.zeros((len(Full_Coeffs)+1, 5))
                    poly_stitch[:-1,:]=Full_Coeffs
                    poly_result.append(xr.Dataset({'poly_asf': (['gamma', self.coord0, 'energy_s', 'coeffs'], [[poly_stitch]]), #
                                                },
                                                coords={
                                                self.coord0: [angle.values],
                                                'energy_s': Full_E, #All of these should be the same
                                                'coeffs': [0,1,2,3,4], #Coefficients of the polynomial in powers of (1/E^(n-1))
                                                'gamma': np.array([gamma])
                                                },        
                                                )
                                            )
                    
                elif calc == 'beta':
                    temp_output = kkdata.convert_Beta_to_ASF(temp_input,
                                                    density=self.density,
                                                    formula_mass=self.formula_mass,
                                                    stoichiometry=self.stoichiometry,
                                                    reverse=True)
                                                    
                compile_result.append(xr.Dataset({calc: ([self.coord0, self.coord1], [temp_output[:,1]]),
                                                },
                                                coords={
                                                self.coord0: [angle.values],
                                                self.coord1: data.coords[self.coord1]
                                                },        
                                                )
                                            )
                
            compile_output.append(xr.concat(compile_result, dim=self.coord0, join='outer'))
            if len(poly_result) > 0:
                compile_poly.append(xr.concat(poly_result, dim=self.coord0, join='outer'))
        if len(compile_poly) > 0:
            return  xr.concat(compile_poly, dim='gamma', join='outer'), xr.concat(compile_output, dim='gamma', join='outer')
        else:
            return xr.concat(compile_output, dim='gamma', join='outer'), 0

            
#Quick function to extract the data from an xarray and compile into a numpy array along with a coord.
#This is to quickly make Xarray compatable with some older software that wants 2D numpy arrays
def xr_to_array(dataarray, coord, reverse=False):
    xw = np.array([dataarray.coords[coord]])
    yw = np.array(dataarray.to_array())
    
    output = np.concatenate((xw,yw))
    if reverse:
        output = np.rollaxis(output,-1)
    return output
            

    
