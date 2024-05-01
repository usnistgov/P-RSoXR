import numpy as np


class LogpExtra(object):
    def __init__(self, objective):
        # we'll store the parameters and objective in this object
        # this will be necessary for pickling in the future
        self.objective = objective  ##Full list of parameters

    def __call__(self, model, data):
        ##First constraint condition ---
        ##Load Parameters of interest for each objective
        for pars in self.objective.parameters:
            thick_pars = sort_pars(pars.flattened(), "thick")
            rough_pars = sort_pars(pars.flattened(), "rough")
            bire_pars = sort_pars(pars.flattened(), "bire")
            delta_val = sort_pars(pars.flattened(), "dt", vary=True)
            ##Check that the roughness is not out of control
            for i in range(len(rough_pars)):  ##Sort through the # of layers
                if (
                    rough_pars[i].vary or thick_pars[i].vary
                ):  # Only constrain parameters that vary
                    interface_limit = (
                        np.sqrt(2 * np.pi) * rough_pars[i].value / 2
                    )  # (rough_pars[i].value/(np.sqrt(2*np.pi)))**2
                    if (
                        float(thick_pars[i].value - interface_limit) < 0
                    ):  # If the interface width is above the corresponding thickness, set logp to -inf
                        return -np.inf
            # Check to see if the Birefringence is physical based on current trace
            for bire in bire_pars:
                if bire.vary and delta_val[0] > 0:
                    if (
                        float(bire - 3 * delta_val[0]) > 0
                        or float(bire + 3 * delta_val[0] / 2) < 0
                    ):
                        return -np.inf
                if bire.vary and delta_val[0] < 0:
                    if (
                        float(bire - 3 * delta_val[0]) < 0
                        or float(bire + 3 * delta_val[0] / 2) > 0
                    ):
                        return -np.inf

        return 0  ##If all the layers are within the constraint return 0


class LogpExtra_rough(object):
    def __init__(self, objective):
        # we'll store the parameters and objective in this object
        # this will be necessary for pickling in the future
        self.objective = objective  ##Full list of parameters

    def __call__(self, model, data):
        ##First constraint condition ---
        ##Load Parameters of interest for each objective
        for pars in self.objective.parameters:
            thick_pars = sort_pars(pars.flattened(), "thick")
            rough_pars = sort_pars(pars.flattened(), "rough")
            ##Check that the roughness is not out of control
            for i in range(len(rough_pars)):  ##Sort through the # of layers
                if (
                    rough_pars[i].vary or thick_pars[i].vary
                ):  # Only constrain parameters that vary
                    interface_limit = (
                        np.sqrt(2 * np.pi) * rough_pars[i].value / 2
                    )  # (rough_pars[i].value/(np.sqrt(2*np.pi)))**2
                    if (
                        float(thick_pars[i].value - interface_limit) < 0
                    ):  # If the interface width is above the corresponding thickness, set logp to -inf
                        return -np.inf

        return 0  ##If all the layers are within the constraint return 0

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
