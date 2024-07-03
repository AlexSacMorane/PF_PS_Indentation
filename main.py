# -*- encoding=utf-8 -*-

import math, os, errno, pickle, time, shutil
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# own
from ic import *
from tools import *
from Parameters import *
from prepare_pf import *

# ------------------------------------------------------------------------------------------------------------------------------------------ #
# Functions

def run_moose(dict_user, dict_sample):
    '''
    Prepare and run moose simulation.
    '''
    # compute data
    compute_as(dict_user, dict_sample) # from prepare_pf.py
    compute_kc(dict_user, dict_sample) # from prepare_pf.py
    compute_c(dict_user, dict_sample) # from prepare_pf.py
    compute_ed_in_film(dict_user, dict_sample) # from tools.py
    compute_sat_in_film(dict_user, dict_sample) # from tools.py

    # write data
    write_eta_txt(dict_user, dict_sample) # from prepare_pf.py
    write_c_txt(dict_user, dict_sample) # from prepare_pf.py
    write_as_txt(dict_user, dict_sample) # from prepare_pf.py
    write_kc_txt(dict_user, dict_sample) # from prepare_pf.py

    # plot data
    if 'tilt' in dict_user['L_figures']:
        plot_ed(dict_user, dict_sample) # from tools.py
    if 'tilt_in_film' in dict_user['L_figures']:
        plot_ed_in_film(dict_user, dict_sample) # from tools.py
    if 'as' in dict_user['L_figures']:
        plot_as(dict_user, dict_sample) # from tools.py
    if 'sat_in_film' in dict_user['L_figures']:
        plot_sat_in_film(dict_user, dict_sample) # from tools.py
    if 'c_removed' in dict_user['L_figures']:
        plot_c_removed(dict_user, dict_sample) # from tools.py

    # compute mass
    compute_mass(dict_user, dict_sample) # from tools.py

    # generate .i file
    write_i(dict_user, dict_sample) # in prepare_pf.py
    
    # run pf simulation    
    os.system('mpiexec -n '+str(dict_user['n_proc'])+' ~/projects/moose/modules/phase_field/phase_field-opt -i pf.i')
    
    # sort output
    last_j_str = sort_files(dict_user, dict_sample) # in tools.py

    # read data
    read_vtk(dict_user, dict_sample, last_j_str) # in tools.py

    # compute mass and mass_loss
    # plot data
    compute_mass_loss(dict_user, dict_sample, 'moose') # from tools.pys

# ------------------------------------------------------------------------------------------------------------------------------------------ #

def track_front(dict_user, dict_sample):
    '''
    Track the front.
    '''
    # initialization
    tempo_y_L = []
    eta_map = dict_sample['eta_map'].copy()
    # iterate on the x-axis
    for i_x in range(len(dict_sample['x_L'])):
        # search the y coordinate of the front 
        i_y = 0
        while not (0.5<=eta_map[-1-i_y, i_x] and eta_map[-1-i_y-1, i_x]<=0.5):
            i_y = i_y + 1
        # linear interpolation
        y_front = (0.5-eta_map[-1-i_y, i_x])/(eta_map[-1-(i_y+1), i_x]-eta_map[-1-i_y, i_x])*(dict_sample['y_L'][i_y+1]-dict_sample['y_L'][i_y])+dict_sample['y_L'][i_y]
        # save in tempo list
        tempo_y_L.append(y_front)
    # save data
    dict_sample['current_front'] = tempo_y_L
    # tracker
    dict_user['y_front_L'].append(np.mean(tempo_y_L))
    dict_user['min_y_front_L'].append(np.min(tempo_y_L))
    dict_user['max_y_front_L'].append(np.max(tempo_y_L))

# ------------------------------------------------------------------------------------------------------------------------------------------ #

def track_eta_profile(dict_user, dict_sample):
    '''
    Track the profile of eta at specific points.
    '''
    # initialization
    eta_map = dict_sample['eta_map'].copy()
    # tracker
    dict_user['L_L_eta_center'].append(eta_map[:, 0].copy())
    dict_user['L_L_eta_ext'].append(eta_map[:, -1].copy()) 

# ------------------------------------------------------------------------------------------------------------------------------------------ #
# Plan
    
# get parameters
dict_user = get_parameters() # from Parameters.py
dict_sample = {}

# folders
create_folder('vtk') # from tools.py
create_folder('plot') # from tools.py
create_folder('data') # from tools.py
create_folder('input') # from tools.py
create_folder('dict') # from tools.py

# if saved check the folder does not exist
if dict_user['save_simulation']:
    # name template id k_diss_k_prec_D_solute_force_applied
    name = str(int(dict_user['k_diss']))+'_'+str(int(dict_user['k_prec']))+'_'+str(int(10*dict_user['D_solute']))+'_'+str(int(dict_user['pressure_applied']))
    # check
    if Path('../Data_PressureSolution_Indenter_2D/'+name).exists():
        raise ValueError('Simulation folder exists: please change parameters')

# init Map
dict_sample['Map_known'] = False
# hint -> check into a database

# compute performances
tic = time.perf_counter()

# ------------------------------------------------------------------------------------------------------------------------------------------ #
# Create initial condition

create_ic(dict_user, dict_sample) # from ic.py

# Plot
if 'ic' in dict_user['L_figures']:
    fig, (ax1, ax2) = plt.subplots(1,2,figsize=(16,9))
    # eta
    im = ax1.imshow(dict_sample['eta_map'], interpolation = 'nearest', extent=(dict_sample['x_L'][0],dict_sample['x_L'][-1],dict_sample['y_L'][0],dict_sample['y_L'][-1]))
    fig.colorbar(im, ax=ax1)
    ax1.set_title(r'Map of $\eta$',fontsize = 30)
    # solute
    im = ax2.imshow(dict_sample['c_map'], interpolation = 'nearest', extent=(dict_sample['x_L'][0],dict_sample['x_L'][-1],dict_sample['y_L'][0],dict_sample['y_L'][-1]))
    fig.colorbar(im, ax=ax2)
    ax2.set_title(r'Map of solute',fontsize = 30)
    # close
    fig.tight_layout()
    fig.savefig('plot/map_ic.png')
    plt.close(fig)

# track interface
track_front(dict_user, dict_sample)
track_eta_profile(dict_user, dict_sample)
dict_user['time_L'].append(0)
   
# ------------------------------------------------------------------------------------------------------------------------------------------ #
# Performances

# ------------------------------------------------------------------------------------------------------------------------------------------ #
# Main

dict_sample['i_ite'] = 0
loop_cond = True
while loop_cond :
    dict_sample['i_ite'] = dict_sample['i_ite'] + 1

    # output
    print('\n',dict_sample['i_ite'],'/',dict_user['n_ite_max'],'\n')

    # prepare, run and read simulation
    run_moose(dict_user, dict_sample)

    # track and plot the interface
    track_front(dict_user, dict_sample)
    track_eta_profile(dict_user, dict_sample)
    dict_user['time_L'].append(dict_user['time_L'][-1]+dict_user['dt_PF']*dict_user['n_t_PF'])
    if 'front' in dict_user['L_figures']:
        plot_front(dict_user, dict_sample) # from tools.py
    if 'eta_profile' in dict_user['L_figures']:
        plot_eta_profile(dict_user, dict_sample) # from tools.py

    # plot tracker on the mean c value in the well
    if 'm_c_well' in dict_user['L_figures']:
        plot_m_c_well(dict_user, dict_sample) # from tools.py

    # plot configuration
    if 'config' in dict_user['L_figures']:
        plot_config(dict_user, dict_sample) # from tools.py

    # check loop conditions
    if dict_sample['i_ite'] >= dict_user['n_ite_max']: # maximum number of iterations
        loop_cond = False
        print('\nEnd because of maximum number of iterations reached')
    if dict_user['min_y_front_L'][-1] < dict_user['y_min'] + 0.8*dict_user['size_tube']: # minimum size of the sample
        loop_cond = False
        print('\nEnd because of minimum sample size reached')
        print(dict_sample['i_ite'],'iterations done on',dict_user['n_ite_max'],'asked')

# ------------------------------------------------------------------------------------------------------------------------------------------ #
# Post Processing

# fit the front evolution
if 'fit' in dict_user['L_figures']:
    plot_fit(dict_user, dict_sample) # from tools.py

# compare the size of the front evolution with the mesh and size tube
m_d_y_front = (dict_user['y_front_L'][0]-dict_user['y_front_L'][-1])/(len(dict_user['y_front_L'])-1)
print()
print('Total Delta y_front / grain height (%):', int((dict_user['y_front_L'][0]-dict_user['y_front_L'][-1])/dict_user['h_grain']*100))
print('Mean delta y_front per ite / Mesh size (%):', int(m_d_y_front/dict_user['m_size_mesh']*100))
#print('Mesh size:', dict_user['m_size_mesh'])
#print('Tube size:', dict_user['size_tube'])

# ------------------------------------------------------------------------------------------------------------------------------------------ #
# close simulation

# save
with open('dict/dict_user', 'wb') as handle:
    pickle.dump(dict_user, handle, protocol=pickle.HIGHEST_PROTOCOL)
with open('dict/dict_sample', 'wb') as handle:
    pickle.dump(dict_sample, handle, protocol=pickle.HIGHEST_PROTOCOL)

# compute performances
tac = time.perf_counter()
hours = (tac-tic)//(60*60)
minutes = (tac-tic - hours*60*60)//(60)
seconds = int(tac-tic - hours*60*60 - minutes*60)
print("\nSimulation time : "+str(hours)+" hours "+str(minutes)+" minutes "+str(seconds)+" seconds")
print('Simulation ends')

# sort files
reduce_n_vtk_files(dict_user, dict_sample) # from tools.py

# copy and paste to Data folder
if dict_user['save_simulation']:
    os.mkdir('../Data_PressureSolution_Indentation_2D/'+name)
    shutil.copytree('dict', '../Data_PressureSolution_Indentation_2D/'+name+'/dict')
    shutil.copytree('plot', '../Data_PressureSolution_Indentation_2D/'+name+'/plot')
    shutil.copy('ic.py', '../Data_PressureSolution_Indentation_2D/'+name+'/ic.py')
    shutil.copy('main.py', '../Data_PressureSolution_Indentation_2D/'+name+'/main.py')
    shutil.copy('Parameters.py', '../Data_PressureSolution_Indentation_2D/'+name+'/Parameters.py')
    shutil.copy('pf_base.i', '../Data_PressureSolution_Indentation_2D/'+name+'/pf_base.i')
    shutil.copy('prepare_pf.py', '../Data_PressureSolution_Indentation_2D/'+name+'/prepare_pf.py')
    shutil.copy('tools.py', '../Data_PressureSolution_Indentation_2D/'+name+'/tools.py')

