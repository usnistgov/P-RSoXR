import numpy as np
import h5py

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
    
"""
Quick Script to construct a tensor object given isotropic
components and difference values. This can be a beneficial
parameterization.
"""
def build_tensor(complex_inputs):
    out = []
    for r, i, dr, di in complex_inputs:
        dxx = r - (1/3)*dr
        bxx = i + (1/3)*di
        nxx = dxx + 1j*bxx

        dzz = r + (2/3)*dr
        bzz = i - (2/3)*di
        nzz = dzz + 1j*bzz
        
        out.append(np.diag([nxx, nxx, nzz]))
    return out