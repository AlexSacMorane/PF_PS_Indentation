# -*- encoding=utf-8 -*-

import math
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import binary_dilation, binary_erosion

# -----------------------------------------------------------------------------#

def compute_as(dict_user, dict_sample):
  '''
  Compute solid activity due to vertical pressure applied.
  '''
  # control of the front
  if dict_user['control_front']:
    if dict_user['control_technique'] == 'monitor':
      compute_as_control(dict_user, dict_sample)
    if dict_user['control_technique'] == 'constant':
      compute_as_control_cst(dict_user, dict_sample)
  # no control of the front
  else : 
    compute_as_no_control(dict_user, dict_sample)

# -----------------------------------------------------------------------------#

def compute_as_control(dict_user, dict_sample):
  '''
  Compute solid activity due to vertical pressure applied.
  '''
  # cst
  R_gas = dict_user['R_cst']
  Temp = dict_user['temperature'] 
  V_m = dict_user['V_m']

  # solid activity 
  a_s = math.exp(dict_user['pressure_applied']*V_m/(R_gas*Temp))

  # init
  dict_sample['as_map'] = np.zeros((dict_user['n_mesh_y'], dict_user['n_mesh_x']))
  # track
  as_L = []
  # iterate on x
  for i_x in range(len(dict_sample['x_L'])):
    # check the front profiles
    y_front = dict_sample['current_front'][i_x]
    # compute control
    control = (y_front-dict_user['y_front_L'][-1])*dict_user['control_front_factor']
    # track
    as_L.append(a_s*(1+control))
    # iterate on y
    for i_y in range(len(dict_sample['y_L'])):
      y = dict_sample['y_L'][i_y]
      if y < dict_user['y_max']-dict_user['size_solute_well'] : # in the grain initially
        dict_sample['as_map'][-1-i_y, i_x] = a_s*(1+control)
      else : # not in the grain initially
        dict_sample['as_map'][-1-i_y, i_x] = 1
  # track
  dict_user['L_L_as'].append(as_L)

# -----------------------------------------------------------------------------#

def compute_as_control_cst(dict_user, dict_sample):
  '''
  Compute solid activity due to vertical pressure applied.
  '''
  # cst
  R_gas = dict_user['R_cst']
  Temp = dict_user['temperature'] 
  V_m = dict_user['V_m']

  # solid activity 
  a_s = math.exp(dict_user['pressure_applied']*V_m/(R_gas*Temp))
  a_s_p = a_s*(1+dict_user['control_front_factor'])
  a_s_m = a_s*(1-dict_user['control_front_factor'])

  # init
  dict_sample['as_map'] = np.zeros((dict_user['n_mesh_y'], dict_user['n_mesh_x']))
  # track
  as_L = []
  # iterate on x
  for i_x in range(len(dict_sample['x_L'])):
    x = dict_sample['x_L'][i_x]
    a_s_i_x = a_s_p + (a_s_m-a_s_p)*(x-dict_sample['x_L'][0])/(dict_sample['x_L'][-1]-dict_sample['x_L'][0])
    as_L.append(a_s_i_x)
    # iterate on y
    for i_y in range(len(dict_sample['y_L'])):
      y = dict_sample['y_L'][i_y]
      if y < dict_user['y_max']-dict_user['size_solute_well'] : # in the grain initially
        dict_sample['as_map'][-1-i_y, i_x] = a_s_i_x
      else : # not in the grain initially
        dict_sample['as_map'][-1-i_y, i_x] = 1
  # track
  dict_user['L_L_as'].append(as_L)

# -----------------------------------------------------------------------------#

def compute_as_no_control(dict_user, dict_sample):
  '''
  Compute solid activity due to vertical pressure applied.
  '''
  # cst
  R_gas = dict_user['R_cst']
  Temp = dict_user['temperature'] 
  V_m = dict_user['V_m']

  # solid activity 
  a_s = math.exp(dict_user['pressure_applied']*V_m/(R_gas*Temp))

  # init
  dict_sample['as_map'] = np.zeros((dict_user['n_mesh_y'], dict_user['n_mesh_x']))
  # iterate on mesh
  for i_x in range(len(dict_sample['x_L'])):
      for i_y in range(len(dict_sample['y_L'])):
          y = dict_sample['y_L'][i_y]
          if y < dict_user['y_max']-dict_user['size_solute_well'] : # in the grain initially
              dict_sample['as_map'][-1-i_y, i_x] = a_s
          else : # not in the grain initially
              dict_sample['as_map'][-1-i_y, i_x] = 1

# -----------------------------------------------------------------------------#

def compute_c(dict_user, dict_sample):
  '''
  Compute solute concentration map.
  '''
  # compute bool map
  comb_map = compute_bool_map(dict_user, dict_sample)

  # initialization
  c_map = dict_sample['c_map'].copy()
  c_map_new = c_map.copy()
  c_removed_map = np.zeros((dict_user['n_mesh_y'], dict_user['n_mesh_x']))

  # track solute moved
  s_solute_moved = 0
  L_p_solute_moved = []
  # iterate on x coordinate
  for i_x in range(len(dict_sample['x_L'])):
    s_solute_moved_i_x = 0 # quantity to move at this x
    s_solute_moved_fluid_i_x = 0 # quantity to move to fluid at this x
    L_i_y_fluid = [] # list of i_y corresponding to the fluid
    # iterate o y coordinate
    for i_y in range(len(dict_sample['y_L'])):
      # detect if this is the fluid
      if dict_sample['current_front'][i_x]-dict_user['size_tube']/2 <= dict_sample['y_L'][i_y] and\
          dict_sample['y_L'][i_y] <= dict_sample['current_front'][i_x]+dict_user['size_tube']/2:
        L_i_y_fluid.append(i_y)
      # detection of solute to move
      if not comb_map[-1-i_y, i_x] and c_map_new[-1-i_y, i_x] != dict_user['C_eq']: # threshold value
        # compute solute to move
        solute_to_move = c_map_new[-1-i_y, i_x] - dict_user['C_eq']
        # detect if this solute must go to the fluid
        factor = 1.5 # increase the fluid zone
        if dict_sample['current_front'][i_x]-factor*dict_user['size_tube']/2 <= dict_sample['y_L'][i_y] and\
            dict_sample['y_L'][i_y] <= dict_sample['current_front'][i_x]+factor*dict_user['size_tube']/2:
          s_solute_moved_fluid_i_x = s_solute_moved_fluid_i_x + solute_to_move    
        # reset value to equilibrium
        c_map_new[-1-i_y, i_x] = dict_user['C_eq']
        # track
        s_solute_moved = s_solute_moved + solute_to_move
        s_solute_moved_i_x = s_solute_moved_i_x + solute_to_move
        c_removed_map[-1-i_y, i_x] = solute_to_move
      
    # iterate on y coordinate of the fluid
    for i_y in L_i_y_fluid:
      # move solute in the fluid
      c_map_new[-1-i_y, i_x] = c_map_new[-1-i_y, i_x] + s_solute_moved_fluid_i_x/len(L_i_y_fluid)
      # track
      c_removed_map[-1-i_y, i_x] = c_removed_map[-1-i_y, i_x] - s_solute_moved_fluid_i_x/len(L_i_y_fluid)
    # search i_y near the front coordinate
    L_search = list(abs(np.array(dict_sample['y_L']-dict_sample['current_front'][i_x])))
    i_y_f = L_search.index(min(L_search))
    as_ij = dict_sample['as_map'][-1-i_y_f, i_x]
    # compute tracker
    p_solute_moved = s_solute_moved_i_x/(dict_user['C_eq']*as_ij)
    L_p_solute_moved.append(p_solute_moved)
  # tracker
  dict_user['s_solute_moved_L'].append(s_solute_moved) 
  dict_user['L_L_p_solute_moved'].append(L_p_solute_moved) 
  dict_user['c_removed_map'] = c_removed_map.copy()

  # plot 
  if 'c_removed_map' in dict_user['L_figures']:
    fig, (ax1) = plt.subplots(1,1,figsize=(16,9))
    im = ax1.imshow(dict_user['c_removed_map'], interpolation = 'nearest', extent=(dict_sample['x_L'][0],dict_sample['x_L'][-1],dict_sample['y_L'][0],dict_sample['y_L'][-1]))
    fig.colorbar(im, ax=ax1)
    ax1.set_title(r'Map of c removed',fontsize = 30)
    fig.tight_layout()
    fig.savefig('plot/map_c_removed.png')
    plt.close(fig)

  # extract solute from the well
  # track mean value
  m_c_well = 0
  n_c_well = 0 
  # iterate on the mesh
  for i_x in range(len(dict_sample['x_L'])):
    for i_y in range(len(dict_sample['y_L'])):  
      # upper zone
      if dict_sample['y_L'][i_y] > dict_user['y_max'] - dict_user['size_solute_well']:
        # track the mean value in the upper zone
        m_c_well = m_c_well + c_map_new[-1-i_y, i_x]
        n_c_well = n_c_well + 1
        # change value to have a well zone
        c_map_new[-1-i_y, i_x] = dict_user['C_eq']
  # track mean
  dict_user['m_c_well_L'].append(m_c_well/n_c_well) 

  # save data
  dict_sample['c_map'] = c_map_new.copy()

# -----------------------------------------------------------------------------#

def compute_kc(dict_user, dict_sample):
    '''
    Compute diffusion map.
    '''
    # compute bool map
    comb_map = compute_bool_map(dict_user, dict_sample)

    # plot
    if 'diff_map' in dict_user['L_figures']:
      fig, (ax1) = plt.subplots(1,1,figsize=(16,9))
      im = ax1.imshow(comb_map, interpolation = 'nearest', extent=(dict_sample['x_L'][0],dict_sample['x_L'][-1],dict_sample['y_L'][0],dict_sample['y_L'][-1]))
      fig.colorbar(im, ax=ax1)
      ax1.set_title(r'Bool',fontsize = 30)
      fig.tight_layout()
      fig.savefig('plot/map_diffusion.png')
      plt.close(fig)

    # assign real value
    kc_map = comb_map*dict_user['D_solute']
    dict_sample['kc_map'] = kc_map

    # iterate on the mesh
    for i_x in range(len(dict_sample['x_L'])):
      for i_y in range(len(dict_sample['y_L'])):  
        # increase diffusion in the well
        if dict_sample['y_L'][i_y] > dict_user['y_max'] - dict_user['size_solute_well']:
          dict_sample['kc_map'][-1-i_y, i_x] = dict_sample['kc_map'][-1-i_y,i_x] * 100
        # increase diffusion in the tube
        elif dict_sample['x_L'][i_x] > dict_user['x_max'] - dict_user['size_tube'] and\
           dict_sample['y_L'][i_y] > dict_sample['current_front'][i_x] + dict_user['size_tube']/2:
          dict_sample['kc_map'][-1-i_y, i_x] = dict_sample['kc_map'][-1-i_y,i_x] * 100

# -----------------------------------------------------------------------------#

def compute_bool_map(dict_user, dict_sample):
  '''
  Compute bool map for c and diffusion.
  '''
  # init
  comb_map = np.array(np.zeros((dict_user['n_mesh_y'], dict_user['n_mesh_x'])), dtype = bool)
    
  # add thin fluid film
  # iterate on x
  for i_x in range(len(dict_sample['x_L'])):
    y_front = dict_sample['current_front'][i_x]
    # iterate on y
    for i_y in range(len(dict_sample['y_L'])):
      y = dict_sample['y_L'][i_y]
      # thin fluid film
      if y_front-dict_user['size_tube']/2 <= y and y <= y_front+dict_user['size_tube']/2:
        comb_map[-1-i_y, i_x] = True
      else :
        comb_map[-1-i_y, i_x] = False

  # add tubes
  # iterate on the x coordinates
  for i_x in range(len(dict_sample['x_L'])):
    # tube at the right
    if dict_user['x_max'] - dict_user['size_tube'] < dict_sample['x_L'][i_x]: 
      i_y = 0
      while not comb_map[i_y, i_x]:
        comb_map[i_y, i_x] = True
        i_y = i_y + 1

  # add top reservoir
  for i_y in range(len(dict_sample['y_L'])):
      if dict_sample['y_L'][i_y] > dict_user['y_max'] - dict_user['size_solute_well']:
        comb_map[-1-i_y, :] = True

  return comb_map

#-------------------------------------------------------------------------------

def write_as_txt(dict_user, dict_sample):
    '''
    Write a .txt file needed for MOOSE simulation.

    This .txt represents the map of the solid activity.
    '''
    file_to_write = open('data/as.txt','w')
    # x
    file_to_write.write('AXIS X\n')
    line = ''
    for x in dict_sample['x_L']:
        line = line + str(x)+ ' '
    line = line + '\n'
    file_to_write.write(line)
    # y
    file_to_write.write('AXIS Y\n')
    line = ''
    for y in dict_sample['y_L']:
        line = line + str(y)+ ' '
    line = line + '\n'
    file_to_write.write(line)
    # data
    file_to_write.write('DATA\n')
    for j in range(len(dict_sample['y_L'])):
        for i in range(len(dict_sample['x_L'])):
            file_to_write.write(str(dict_sample['as_map'][-1-j,i])+'\n')
    # close
    file_to_write.close()

#-------------------------------------------------------------------------------

def write_kc_txt(dict_user, dict_sample):
    '''
    Write a .txt file needed for MOOSE simulation.

    This .txt represents the map of the diffusion.
    '''
    file_to_write = open('data/kc.txt','w')
    # x
    file_to_write.write('AXIS X\n')
    line = ''
    for x in dict_sample['x_L']:
        line = line + str(x)+ ' '
    line = line + '\n'
    file_to_write.write(line)
    # y
    file_to_write.write('AXIS Y\n')
    line = ''
    for y in dict_sample['y_L']:
        line = line + str(y)+ ' '
    line = line + '\n'
    file_to_write.write(line)
    # data
    file_to_write.write('DATA\n')
    for j in range(len(dict_sample['y_L'])):
        for i in range(len(dict_sample['x_L'])):
            file_to_write.write(str(dict_sample['kc_map'][-1-j,i])+'\n')
    # close
    file_to_write.close()

#-------------------------------------------------------------------------------

def write_c_txt(dict_user, dict_sample):
    '''
    Write a .txt file needed for MOOSE simulation.

    This .txt represents the map of the solute concentration.
    '''
    file_to_write = open('data/c.txt','w')
    # x
    file_to_write.write('AXIS X\n')
    line = ''
    for x in dict_sample['x_L']:
        line = line + str(x)+ ' '
    line = line + '\n'
    file_to_write.write(line)
    # y
    file_to_write.write('AXIS Y\n')
    line = ''
    for y in dict_sample['y_L']:
        line = line + str(y)+ ' '
    line = line + '\n'
    file_to_write.write(line)
    # data
    file_to_write.write('DATA\n')
    for j in range(len(dict_sample['y_L'])):
        for i in range(len(dict_sample['x_L'])):
            file_to_write.write(str(dict_sample['c_map'][-1-j,i])+'\n')
    # close
    file_to_write.close()

#-------------------------------------------------------------------------------

def write_eta_txt(dict_user, dict_sample):
    '''
    Write a .txt file needed for MOOSE simulation.

    This .txt represents the map of phase.
    '''
    file_to_write = open('data/eta.txt','w')
    # x
    file_to_write.write('AXIS X\n')
    line = ''
    for x in dict_sample['x_L']:
        line = line + str(x)+ ' '
    line = line + '\n'
    file_to_write.write(line)
    # y
    file_to_write.write('AXIS Y\n')
    line = ''
    for y in dict_sample['y_L']:
        line = line + str(y)+ ' '
    line = line + '\n'
    file_to_write.write(line)
    # data
    file_to_write.write('DATA\n')
    for j in range(len(dict_sample['y_L'])):
        for i in range(len(dict_sample['x_L'])):
            file_to_write.write(str(dict_sample['eta_map'][-1-j,i])+'\n')
    # close
    file_to_write.close()

#-------------------------------------------------------------------------------

def write_i(dict_user, dict_sample):
  '''
  Create the .i file to run MOOSE simulation.

  The file is generated from a template nammed pf_base.i
  '''
  file_to_write = open('pf.i','w')
  file_to_read = open('pf_base.i','r')
  lines = file_to_read.readlines()
  file_to_read.close()

  j = 0
  for line in lines :
    j = j + 1
    if j == 4:
      line = line[:-1] + ' ' + str(len(dict_sample['x_L'])-1)+'\n'
    elif j == 5:
      line = line[:-1] + ' ' + str(len(dict_sample['y_L'])-1)+'\n'
    elif j == 6:
      line = line[:-1] + ' ' + str(min(dict_sample['x_L']))+'\n'
    elif j == 7:
      line = line[:-1] + ' ' + str(max(dict_sample['x_L']))+'\n'
    elif j == 8:
      line = line[:-1] + ' ' + str(min(dict_sample['y_L']))+'\n'
    elif j == 9:
      line = line[:-1] + ' ' + str(max(dict_sample['y_L']))+'\n'
    elif j == 83:
      line = line[:-1] + "'1 "+str(dict_user['kappa_eta'])+" 1'\n"
    elif j == 105:
      line = line[:-1] + ' ' + str(dict_user['Energy_barrier'])+"'\n"
    elif j == 118:
      line = line[:-1] + "'" + str(dict_user['C_eq']) + ' ' + str(dict_user['k_diss']) + ' ' + str(dict_user['k_prec']) + "'\n"
    elif j == 175 or j == 176 or j == 178 or j == 179:
      line = line[:-1] + ' ' + str(dict_user['crit_res']) +'\n'
    elif j == 182:
      line = line[:-1] + ' ' + str(dict_user['dt_PF']*dict_user['n_t_PF']) +'\n'
    elif j == 186:
      line = line[:-1] + ' ' + str(dict_user['dt_PF']) +'\n'
    file_to_write.write(line)

  file_to_write.close()
