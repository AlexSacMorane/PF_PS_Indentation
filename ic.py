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

    print('Mesh size:',len(x_L),'-',len(y_L),'\n')

    # ------------------------------------------------------------------------------------------------------------------------------------------ #
    # Create initial phase and solute maps

    print("Creating initial phase field and solute maps")

    eta_map = np.zeros((dict_user['n_mesh_y'], dict_user['n_mesh_x']))
    c_map = np.zeros((dict_user['n_mesh_y'], dict_user['n_mesh_x']))
    
    # cst
    R_gas = dict_user['R_cst']
    Temp = dict_user['temperature'] 
    V_m = dict_user['V_m']

    # solid activity 
    a_s = math.exp(dict_user['pressure_applied']*V_m/(R_gas*Temp))
    if dict_user['control_technique'] == 'constant':
        a_s_p = a_s*(1+dict_user['control_front_factor'])
        a_s_m = a_s*(1-dict_user['control_front_factor'])

    # iteration on x
    for i_x in range(len(x_L)):
        x = x_L[i_x]
        if dict_user['adapt_ic']:
            # look for x in L_adapt_x_ic
            i_adapt_x_ic = 0
            while not (dict_user['L_adapt_x_ic'][i_adapt_x_ic] <= x and x <= dict_user['L_adapt_x_ic'][i_adapt_x_ic+1]):
                i_adapt_x_ic = i_adapt_x_ic + 1
            # read data 
            f_a_s_m = dict_user['L_adapt_as_ic'][i_adapt_x_ic]
            f_a_s_p = dict_user['L_adapt_as_ic'][i_adapt_x_ic+1]
            x_m = dict_user['L_adapt_x_ic'][i_adapt_x_ic]
            x_p = dict_user['L_adapt_x_ic'][i_adapt_x_ic+1]
        # iteration on y
        for i_y in range(len(y_L)):
            y = y_L[i_y]

            # eta 
            if y <= dict_user['h_grain']-dict_user['w_int']/2 :
                eta_map[-1-i_y, i_x] = 1
            elif dict_user['h_grain']-dict_user['w_int']/2 < y and y < dict_user['h_grain']+dict_user['w_int']/2:
                distance = y-dict_user['h_grain']
                eta_map[-1-i_y, i_x] = 0.5*(1+math.cos(math.pi*(distance+dict_user['w_int']/2)/dict_user['w_int']))
            elif dict_user['h_grain']+dict_user['w_int']/2 <= y :
                eta_map[-1-i_y, i_x] = 0

            # solute 
            # system at the equilibrium initialy
            if dict_user['control_technique'] == 'constant':
                a_s_ij = a_s_p + (a_s_m-a_s_p)*(x-x_L[0])/(x_L[-1]-x_L[0])
            elif dict_user['adapt_ic']:
                f_as = f_a_s_p + (f_a_s_m-f_a_s_p)*(x-x_m)/(x_p-x_m)
                a_s_ij = a_s*f_as
            else :
                a_s_ij = a_s
            if dict_user['h_grain']-dict_user['size_tube']/2 <= y and y <= dict_user['h_grain']+dict_user['size_tube']/2:
                c_map[-1-i_y, i_x] = dict_user['C_eq']*a_s_ij
            else :
                c_map[-1-i_y, i_x] = dict_user['C_eq'] 
            
    # save dict
    dict_sample['eta_map'] = eta_map.copy()
    dict_sample['c_map'] = c_map.copy()
    dict_sample['x_L'] = x_L.copy()
    dict_sample['y_L'] = y_L.copy()
