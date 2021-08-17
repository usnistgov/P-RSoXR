# -*- coding: utf-8 -*-

import os
import re

import numpy as np
import xarray as xr
import pandas as pd
import warnings
   

class Data1D_series(object):
    """
    
    Class of data that describes related 1D data that contain a secondary (or more) adjustable parameters
    
    Example: NEXAFS data taken as a function of angle
        Primary dataset is Intensity vs. Energy --- Secondary parameter is incident angle
    Example: P-RSoXR data taken as a function of energy and angle
        Primary dataset is Reflectance vs. Q --- Alternative parameters in polarization and energy
            Everything is important to collate from a single experiment.   
    
    
    """
    
    def __init__(self, data=None, coords=[], vary_coord=[None], experiment_name='data', mask=None):
        self.filename = []
        self.name = []

        #self.metadata = kwds
        #Series of input meta data that defines the xarray that will be generated
        #Predefine all this crap in a child function and call this initialization for easier assignment
        self._coords = coords
        self._vary_coord = vary_coord
        self._experiment_name = experiment_name
        #self._data = xr.DataArray()
        temp_data=[]
        for d in data:
            if type(d) == str:
                temp_data.append(self.load(d))
        self.data = temp_data
            
        
        #self.data = (self._load_data, self._load_data_err)
            
            
        self._mask = None
        #if mask is not None:
        #    self._mask = np.broadcast_to(mask, self._data.shape)

        
    def __len__(self):
        """
        the number of sets within the data
        """
        
        return len(self._data)
    def __str__(self):
        """
        Return info on the generated data
        """
        
        return str(self._data)
        
    @property
    def coords(self, data=None):
        if data is not None:
            return self.data['data'].coords
        else:
            return self.data.coords
    
    @property
    def dims(self, data=None):
        if data is not None:
        
            return self.data['data'].dims
        else:
            return self.data.dims
            
    @property
    def data(self):
        """
        Returns DataSet with the data / uncertainty of the loaded files
        """
        return self._dataset
        
        
    @data.setter
    def data(self, data_tuple):
        compile_data=[]
        for i, d in enumerate(data_tuple):
        ##Parse into lists of the information I want
            compile_data.append(self.make_DataSet(d, self._vary_coord[i]))
            
        self._dataset = xr.concat(compile_data,dim=self._coords[0],join='outer') #Generate dataset
        self._dataset = self._dataset.interpolate_na(dim=self._coords[1]) #Interpolate nans
        self._dataset = self._dataset.dropna(dim='energy') #Kill the potential nans at the start and end
        
        
        
      
    def make_DataSet(self, data, vary_coord):
        coord_list = np.array(data[0][0], dtype=float) #Grab the x axis information
        data_list = [np.array(data[0][1], dtype=float)] #Grab the y axis information
        coord_err_list = None
        data_err_list = None
        
        if data[0][1] is not None:
            coord_err_list = np.array(data[0][1], dtype=float)
            
        if data[1][1] is not None:
            data_err_list = np.array(data[1][1], dtype=float)
        
        if data_err_list is not None:
            temp_DataSet = xr.Dataset({self._experiment_name: (self._coords, data_list),
                                    self._experiment_name + '_err': (self._coords, data_err_list)
                                    },
                                    coords={
                                    self._coords[0]: [vary_coord],
                                    self._coords[1]: coord_list
                                    },        
                                    )
        else:
            temp_DataSet = xr.Dataset({self._experiment_name: (self._coords, data_list),
                                    },
                                    coords={
                                    self._coords[0]: [vary_coord],
                                    self._coords[1]: coord_list
                                    },        
                                    )
                 
        return temp_DataSet
    
    
    def load(self, file, y_err=None, x_err=None, primary_dim='x', secondary_dim='y'):
        """
        Borrowing the generic loader from data1d refnx.
        """

        with open(file, 'r') as f:
            lines = list(reversed(f.readlines())) #Start from the back and move to the front --
            x = [] #Temporary waves to append to xarray
            y = []
            y_err = []
            x_err = []
            #How many columns exist
            numcols = 0
            for line in lines:
                try:
                    # parse a line for numerical tokens separated by whitespace
                    # or comma
                    nums = [float(tok) for tok in
                            re.split(r"\s|,|/t", line)
                            if len(tok)]
                    if len(nums) in [0, 1]:
                        # might be trailing newlines at the end of the file,
                        # just ignore those
                        continue
                    if not numcols:
                        # figure out how many columns one has
                        numcols = len(nums)
                    elif len(nums) != numcols:
                        # if the number of columns changes there's an issue
                        break
                    
                    x.append(nums[0])
                    y.append(nums[1])
                    if len(nums) > 2:
                        y_err.append(nums[2])
                    if len(nums) > 3:
                        x_err.append(nums[3])
                except ValueError:
                    # you should drop into this if you can't parse tokens into
                    # a series of floats. But the text may be meta-data, so
                    # try to carry on.
                    continue
                
        x.reverse()
        y.reverse()
        y_err.reverse()
        x_err.reverse()

        if len(x) == 0:
            raise RuntimeError("Datafile didn't appear to contain any data (or"
                                " was the wrong format)")
        if numcols < 3:
            y_err = None
        if numcols < 4:
            x_err = None
            
        if hasattr(f, 'read'):
            fname = f.name
        else:
            fname = f
        
        self.filename.append(fname)
        self.name.append(os.path.splitext(os.path.basename(fname))[0])      
        
        return ([x,y], [x_err,y_err])
        