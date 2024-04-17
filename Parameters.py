#------------------------------------------------------------------------------------------------------------------------------------------ #
# Librairies
#------------------------------------------------------------------------------------------------------------------------------------------#

import numpy as np

#------------------------------------------------------------------------------------------------------------------------------------------ #
# Parameters
#------------------------------------------------------------------------------------------------------------------------------------------#

def get_parameters():
    '''
    Define the parameters used in the simulation.
    '''

    #---------------------------------------------------------------------#
    # Algorithm

    n_ite_max = 20 # numbe of iteration
    n_proc = 4 # number of processors used
    j_total = 0 # index global of results
    save_simulation = False # indicate if the simulation is saved
    n_max_vtk_files = 10 # maximum number of vtk files (can be None to save all files)

    # Select Figures to plot
    # Available:
    # ic, diff_map, tilt, processor, config
    # front, m_c_well, fit
    L_figures = ['tilt', 'front', 'm_c_well', 'config', 'fit', 'diff_map']

    # the time stepping and duration of PF simulation
    dt_PF = 0.01 # time step
    n_t_PF = 10 # number of iterations
    # n_t_PF*dt_PF gives the total time duration

    #---------------------------------------------------------------------#
    # Sollicitation

    pressure_applied = 1e6 # Pa
    
    #---------------------------------------------------------------------#
    # Indenter description

    # the size of grain / indenter
    d_indenter = 1 # m
    
    # Solute well
    size_solute_well = 0.2 # m
    size_tube = d_indenter/20 # m
    L_coordinates_tube = [] 
    # tubes at x_min and x_max are also added
    
    #---------------------------------------------------------------------#
    # Mesh

    x_min = -d_indenter/2
    x_max =  d_indenter/2
    y_min = 0
    y_max = d_indenter + 1.2*size_solute_well
    n_mesh_x = 100
    n_mesh_y = 100
    m_size_mesh = ((x_max-x_min)/(n_mesh_x-1)+(y_max-y_min)/(n_mesh_y-1))/2

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

    #---------------------------------------------------------------------#
    # kinetics of dissolution, precipitation and diffusion
    
    # it affects the tilting coefficient in Ed
    k_diss = 20 # mol.m-2.s-1
    k_prec = 20

    # molar concentration at the equilibrium
    C_eq = 1 # number of C_ref, mol m-3

    # diffusion of the solute
    D_solute = 10 # m2 s-1
    # structural element
    n_struct_element = int(round(size_tube/m_size_mesh,0))
    struct_element = np.array(np.ones((n_struct_element,n_struct_element)), dtype=bool) # for dilation

    #---------------------------------------------------------------------#
    # trackers

    y_front_L = []
    m_c_well_L = []

    #---------------------------------------------------------------------#
    # dictionnary

    dict_user = {
    'n_ite_max': n_ite_max,
    'n_proc': n_proc,
    'j_total': j_total,
    'save_simulation': save_simulation,
    'n_max_vtk_files': n_max_vtk_files,
    'L_figures': L_figures,
    'kappa_eta': kappa_eta,
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
    'x_min': x_min,
    'x_max': x_max,
    'y_min': y_min,
    'y_max': y_max,
    'n_mesh_x': n_mesh_x,
    'n_mesh_y': n_mesh_y,
    'd_indenter': d_indenter,
    'size_solute_well': size_solute_well,
    'size_tube': size_tube,
    'L_coordinates_tube': L_coordinates_tube, 
    'struct_element': struct_element,
    'y_front_L': y_front_L,
    'm_c_well_L': m_c_well_L
    }

    return dict_user
