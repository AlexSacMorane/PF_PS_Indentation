# -*- encoding=utf-8 -*-

import numpy as np
import math

# ------------------------------------------------------------------------------------------------------------------------------------------ #

def create_ic(dict_user, dict_sample):
    '''
    Create initial conditions with 1 grain and an indenter.
    Mesh, phase field and solute maps are generated
    '''    
    # ------------------------------------------------------------------------------------------------------------------------------------------ #
    # Create initial mesh
    print("Creating initial mesh")

    x_L = np.linspace(dict_user['x_min'], dict_user['x_max'], dict_user['n_mesh_x'])
    y_L = np.linspace(dict_user['y_min'], dict_user['y_max'], dict_user['n_mesh_y'])

    # ------------------------------------------------------------------------------------------------------------------------------------------ #
    # Create initial phase and solute maps

    print("Creating initial phase field and solute maps")

    eta_map = np.zeros((dict_user['n_mesh_y'], dict_user['n_mesh_x']))
    c_map = np.zeros((dict_user['n_mesh_y'], dict_user['n_mesh_x']))
    
    # iteration on x
    for i_x in range(len(x_L)):
        x = x_L[i_x]
        # iteration on y
        for i_y in range(len(y_L)):
            y = y_L[i_y]

            # eta 
            if y <= dict_user['d_indenter']-dict_user['w_int']/2 :
                eta_map[-1-i_y, i_x] = 1
            elif dict_user['d_indenter']-dict_user['w_int']/2 < y and y < dict_user['d_indenter']+dict_user['w_int']/2:
                distance = y-dict_user['d_indenter']
                eta_map[-1-i_y, i_x] = 0.5*(1+math.cos(math.pi*(distance+dict_user['w_int']/2)/dict_user['w_int']))
            elif dict_user['d_indenter']+dict_user['w_int']/2 <= y :
                eta_map[-1-i_y, i_x] = 0

            # solute 
            c_map[-1-i_y, i_x] = dict_user['C_eq'] # system at the equilibrium initialy

    # save dict
    dict_sample['eta_map'] = eta_map
    dict_sample['c_map'] = c_map
    dict_sample['x_L'] = x_L
    dict_sample['y_L'] = y_L
