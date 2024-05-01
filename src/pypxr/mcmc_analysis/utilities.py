import numpy as np
import pickle
import h5py
import matplotlib.pyplot as plt

from refnx.analysis import integrated_time
"""
Load prsoxr data that has been saved as hdf5
"""

def compile_data_hdf5(hdf5, en_list=[None], pol_list=[None], concat=True):
    out_data = []
    out_mask = []
    for energy in hdf5:
        if hdf5[energy]['en'] in en_list:
            temp_data = []
            temp_mask = []
            for pol in [key for key, value in hdf5[energy].items() if 'p' in key]:
                if float(pol[1:]) in pol_list:
                    val = hdf5[energy][pol]
                    temp_data.append(val)
                    temp_mask.append(val.T[1] > 0)
            if concat:
                out_data.append(np.concatenate(temp_data, axis=0))
                out_mask.append(np.concatenate(temp_mask, axis=0))
            else:
                for item, mask in zip(temp_data, temp_mask):
                    out_data.append(item)
                    out_mask.append(mask)
    return out_data, out_mask

"""
Load prsoxr data that has been saved as hdf5
"""

def load_prsoxr_hdf5(file, path):
    f_load = path + file
    energy_list = []
    pol_list = []
    hdf5_data = []
    out = {}
    with h5py.File(f_load, mode='r') as f:
        measurement = f.require_group('MEASUREMENT')
        for energy in measurement:
            for pol in measurement[energy]:
                for data in measurement[energy][pol]:
                    if data == 'DATA':
                        energy_list.append(float(energy.replace('pt', '.')[3:]))
                        pol_list.append(float(pol[-3:]))
                        hdf5_data.append(measurement[energy][pol][data][()])
    energy_unique = sorted(list(set(energy_list)))
    for j, energy in enumerate(energy_unique):
        index = [i for i, x in enumerate(energy_list) if x==energy]
        out_en = out['en'+str(j)] = {}
        out_en['en'] = energy
        for i in index:
            out_en_pol = out_en['p'+str(int(pol_list[i]))] = hdf5_data[i]

    return out

class LogpExtra_rough(object):
    def __init__(self, objective):
        # we'll store the parameters and objective in this object
        # this will be necessary for pickling in the future
        self.objective = objective ##Full list of parameters

    def __call__(self, model, data):
        ##First constraint condition ---
        ##Load Parameters of interest for each objective
        for pars in self.objective.parameters:
            thick_pars = sort_pars(pars.flattened(),'thick')
            rough_pars = sort_pars(pars.flattened(),'rough')
            ##Check that the roughness is not out of control
            for i, rough in enumerate(rough_pars[1:-1], start=1): ##Sort through the layers
                interface_limit = np.sqrt(2*np.pi)*rough.value/2
                top = [(thick_pars[i-1].value - interface_limit) < 0, thick_pars[i-1].value != 0]
                bottom = [(thick_pars[i].value - interface_limit) < 0, thick_pars[i].value != 0]
                if all(top) or all(bottom):
                    return -np.inf
        return 0 ##If all the layers are within the constraint return 0


    ##Function to sort through ALL parameters in an objective and return based on name keyword
    ##Returns a list of parameters for further use
def sort_pars(pars, str_check, vary=None):
    temp = []
    num = len(pars)
    for i in range(num):
        if str_check in pars[i].name:
            if vary == True:
                if pars[i].vary == True:
                    temp.append(pars[i])
            elif vary == False:
                if pars[i].vary == False:
                    temp.append(pars[i])
            else:
                temp.append(pars[i])
    return temp

"""
Quick Script to construct a tensor object given isotropic
components and difference values. This can be a beneficial
parameterization.

Parameters
==========
complex_inputs : zip(r, i, dr, di)
    list of objects zipped together: r - real magnitude, i - imaginary magnitude, dr - birefringence, di - dichroism
"""
def build_tensor(complex_inputs):
    out = []
    for r, i, dr, di in complex_inputs:
        dxx = r + (1/3)*dr
        bxx = i + (1/3)*di
        nxx = dxx + 1j*bxx

        dzz = r - (2/3)*dr
        bzz = i - (2/3)*di
        nzz = dzz + 1j*bzz

        out.append(np.diag([nxx, nxx, nzz]))
    return out
