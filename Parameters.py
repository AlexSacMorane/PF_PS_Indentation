#------------------------------------------------------------------------------------------------------------------------------------------ #
# Librairies
#------------------------------------------------------------------------------------------------------------------------------------------#

import numpy as np
import math

#------------------------------------------------------------------------------------------------------------------------------------------ #
# Parameters
#------------------------------------------------------------------------------------------------------------------------------------------#

def get_parameters():
    '''
    Define the parameters used in the simulation.
    '''
    #---------------------------------------------------------------------#
    # Norrmalization
    n_dist = 100*1e-6 # m
    n_time = 24*60*60 # s
    n_mol = 0.73*1e3 * n_dist**3 # mol

    #---------------------------------------------------------------------#
    # Algorithm

    n_ite_max = 50 # number of iterations
    n_proc = 8 # number of processors used
    j_total = 0 # index global of results
    j_movie = 0 # index global of the movie
    save_simulation = False # indicate if the simulation is saved
    n_max_vtk_files = 2 # maximum number of vtk files (can be None to save all files)

    # Select Figures to plot
    # Available:
    # ic, diff_map, tilt, processor, config, movie
    # front, m_c_well, fit, mass_loss, eta_profile, as
    # tilt_in_film, sat_in_film, c_removed, c_removed_map
    L_figures = ['front', 'm_c_well', 'config', 'sat_in_film', 'movie', 'as']
    
    # max profile in plot
    max_plot = 10

    # the time stepping and duration of PF simulation
    dt_PF = (0.0001*24*60*60)/n_time # time step
    n_t_PF = 300 # number of iterations
    # n_t_PF*dt_PF gives the total time duration

    #---------------------------------------------------------------------#
    # Sollicitation
    # the solid activity is computed with as=exp(PV/RT)

    # pressure
    pressure_applied = (200*1e6)/(1/(n_dist*n_time**2)) # (kg m-1 s-2)/(m-1 s-2)
    # temperature
    temperature = 623 # K 
    # molar volume
    V_m = (2.2*1e-5)/(n_dist**3/n_mol) # (m3 mol-1)/(m3 mol-1)
    # constant
    R_cst = (8.32)/(n_dist**2/(n_time**2*n_mol)) # (kg m2 s-2 mol-1 K-1)/(m2 s-2 mol-1)

    # adapt ic
    adapt_ic = False
    L_adapt_as_ic = [1.151, 1.138, 1.094, 1.001, 0.842, 0.686]
    L_adapt_x_ic = [0, 0.22, 0.435, 0.65, 0.865, 1.075]   

    # control the front
    control_front = True
    # monitor, constant
    control_technique = 'monitor'
    # monitor 
    control_front_factor = 40 #m/m / -
    # constant adaptation
    #control_front_factor = 0.04

    #---------------------------------------------------------------------#
    # Indenter description

    # the size of grain / indenter
    d_indenter = (400*1e-6)/n_dist # m/m
    h_grain = (15*1e-6)/n_dist # m/m

    #---------------------------------------------------------------------#
    # Mesh

    # size of the mesh
    m_size_mesh = (d_indenter/2)/200

    #---------------------------------------------------------------------#
    # PF material parameters

    # the energy barrier
    Energy_barrier = 1
    # number of mesh in the interface
    n_int = 6
    # the interface thickness
    w = m_size_mesh*n_int
    # the gradient coefficient
    kappa_eta = Energy_barrier*w*w/9.86
    # Mobility
    mobility = (100*1e-6/(24*60*60))/(n_dist/n_time) # m.s-1/(m.s-1) 
    # the criteria on residual
    crit_res = 1e-7

    #---------------------------------------------------------------------#
    # Indenter description

    # Solute well
    size_solute_well = 8*m_size_mesh # 

    # Tubes
    size_tube = m_size_mesh*15 # m/m
    
    #---------------------------------------------------------------------#
    # Mesh

    x_min = 0
    x_max = d_indenter/2 + size_tube
    y_min = 0
    y_max = h_grain + size_solute_well + size_tube
    n_mesh_x = int((x_max-x_min)/m_size_mesh+1)
    n_mesh_y = int((y_max-y_min)/m_size_mesh+1)

    #---------------------------------------------------------------------#
    # kinetics of dissolution, precipitation and diffusion
    
    # molar concentration at the equilibrium
    C_eq = (0.73*1e3)/(n_mol/n_dist**3) # (mol m-3)/(mol m-3)

    # it affects the tilting coefficient in Ed
    k_diss = 1*(0.005)/(m_size_mesh) # ed_j = ed_i*m_i/m_j
    k_prec = 0 # -
    # here
    # k_prec = k_diss/C_eq
    
    # diffusion of the solute
    # linked to the size of the tube
    D_solute = (4e-14/size_tube)/(n_dist*n_dist/n_time) # (m2 s-1)/(m2 s-1)
    
    #---------------------------------------------------------------------#
    # trackers

    y_front_L = []
    min_y_front_L = []
    max_y_front_L = []
    m_c_well_L = []
    time_L = []
    moose_eta = []
    moose_c = []
    moose_mass = []
    moose_eta_p = []
    moose_c_p = []
    moose_mass_p = []
    L_m_ed_in_film = []
    L_L_ed_in_film = []
    L_m_sat_in_film = []
    L_L_sat_in_film = []
    L_L_eta_center = []
    L_L_eta_ext = []
    L_L_as = []
    s_solute_moved_L = []
    L_L_p_solute_moved = []

    #---------------------------------------------------------------------#
    # dictionnary

    dict_user = {
    'n_dist': n_dist,
    'n_time': n_time,
    'n_mol': n_mol,
    'n_ite_max': n_ite_max,
    'n_proc': n_proc,
    'j_total': j_total,
    'j_movie': j_movie,
    'save_simulation': save_simulation,
    'n_max_vtk_files': n_max_vtk_files,
    'L_figures': L_figures,
    'max_plot': max_plot,
    'kappa_eta': kappa_eta,
    'mobility': mobility,
    'crit_res': crit_res,
    'n_int': n_int,
    'w_int': w,
    'Energy_barrier': Energy_barrier,
    'n_t_PF': n_t_PF,
    'k_diss': k_diss,
    'k_prec': k_prec,
    'C_eq': C_eq,
    'D_solute': D_solute,
    'dt_PF': dt_PF,
    'pressure_applied': pressure_applied, 
    'adapt_ic': adapt_ic,
    'L_adapt_as_ic': L_adapt_as_ic,
    'L_adapt_x_ic': L_adapt_x_ic,
    'control_front': control_front,
    'control_technique': control_technique,
    'control_front_factor': control_front_factor,
    'temperature': temperature,
    'R_cst': R_cst,
    'V_m': V_m,
    'x_min': x_min,
    'x_max': x_max,
    'y_min': y_min,
    'y_max': y_max,
    'm_size_mesh': m_size_mesh,
    'n_mesh_x': n_mesh_x,
    'n_mesh_y': n_mesh_y,
    'd_indenter': d_indenter,
    'h_grain': h_grain,
    'size_solute_well': size_solute_well,
    'size_tube': size_tube,
    'y_front_L': y_front_L,
    'min_y_front_L': min_y_front_L,
    'max_y_front_L': max_y_front_L,
    'm_c_well_L': m_c_well_L,
    'time_L': time_L,
    'moose_eta': moose_eta,
    'moose_c': moose_c,
    'moose_m': moose_mass,
    'moose_eta_p': moose_eta_p,
    'moose_c_p': moose_c_p,
    'moose_m_p': moose_mass_p,
    'L_m_ed_in_film': L_m_ed_in_film,
    'L_L_ed_in_film': L_L_ed_in_film,
    'L_m_sat_in_film': L_m_sat_in_film,
    'L_L_sat_in_film': L_L_sat_in_film,
    'L_L_eta_center': L_L_eta_center,
    'L_L_eta_ext': L_L_eta_ext,
    'L_L_as': L_L_as,
    's_solute_moved_L': s_solute_moved_L,
    'L_L_p_solute_moved': L_L_p_solute_moved
    }

    return dict_user
