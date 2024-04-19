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
    # cst
    R_gas = 82.06e5 # cm3 Pa K-1 mol-1
    Temp = 25+278   # K
    V_m = 27.1      # cm3 mol-1

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
  c_map = dict_sample['c_map'] 
  c_map_new = c_map

  if dict_user['push_in_out'] :
    # push out solute
    for i_x in range(len(dict_sample['x_L'])):
      for i_y in range(len(dict_sample['y_L'])):  
        # push solute out of the solid
        if not comb_map[i_y, i_x] and c_map[i_y, i_x] > dict_user['C_eq']: # threshold value
          solute_moved = False
          size_window = 1
          # compute solute to move
          solute_to_move = c_map[i_y, i_x] - dict_user['C_eq']
          while not solute_moved :
              i_window = 0
              while not solute_moved and i_window <= size_window:
                n_node_available = 0

                #Look to move horizontaly and vertically
                if i_window == 0 :
                  top_available = False
                  down_available = False
                  left_available = False
                  right_available = False
                  #to the top
                  if i_y - size_window > 0:
                    top_available = comb_map[i_y-size_window, i_x]
                    if comb_map[i_y-size_window, i_x] :
                      n_node_available = n_node_available + 1
                  #to the down
                  if i_y + size_window < len(dict_sample['y_L']):
                    down_available = comb_map[i_y+size_window, i_x]
                    if comb_map[i_y+size_window, i_x] :
                      n_node_available = n_node_available + 1
                  #to the left
                  if i_x - size_window > 0:
                    left_available = comb_map[i_y, i_x-size_window]
                    if comb_map[i_y, i_x-size_window] :
                      n_node_available = n_node_available + 1
                  #to the right
                  if i_x + size_window < len(dict_sample['x_L']):
                    right_available = comb_map[i_y, i_x+size_window]
                    if comb_map[i_y, i_x+size_window] :
                      n_node_available = n_node_available + 1

                  #move solute if at least one node is available
                  if n_node_available != 0 :
                    #to the top
                    if top_available:
                      c_map_new[i_y-size_window, i_x] = c_map_new[i_y-size_window, i_x] + solute_to_move/n_node_available
                    #to the down
                    if down_available:
                      c_map_new[i_y+size_window, i_x] = c_map_new[i_y+size_window, i_x] + solute_to_move/n_node_available
                    #to the left
                    if left_available:
                      c_map_new[i_y, i_x-size_window] = c_map_new[i_y, i_x-size_window] + solute_to_move/n_node_available
                    #to the right
                    if right_available:
                      c_map_new[i_y, i_x+size_window] = c_map_new[i_y, i_x+size_window] + solute_to_move/n_node_available
                    c_map_new[i_y, i_x] = dict_user['C_eq']
                    solute_moved = True

                #Look to move diagonally
                else :
                  top_min_available = False
                  top_max_available = False
                  down_min_available = False
                  down_max_available = False
                  left_min_available = False
                  left_max_available = False
                  right_min_available = False
                  right_max_available = False
                  #to the top
                  if i_y - size_window > 0:
                    if i_x - i_window > 0 :
                      top_min_available = comb_map[i_y-size_window, i_x-i_window]
                      if comb_map[i_y-size_window, i_x-i_window] :
                        n_node_available = n_node_available + 1
                    if i_x + i_window < len(dict_sample['x_L']):
                      top_max_available = comb_map[i_y-size_window, i_x+i_window]
                      if comb_map[i_y-size_window, i_x+i_window] :
                        n_node_available = n_node_available + 1
                  #to the down
                  if i_y + size_window < len(dict_sample['y_L']):
                    if i_x - i_window > 0 :
                      down_min_available = comb_map[i_y+size_window, i_x-i_window]
                      if comb_map[i_y+size_window, i_x-i_window] :
                        n_node_available = n_node_available + 1
                    if i_x + i_window < len(dict_sample['x_L']):
                      down_max_available = comb_map[i_y+size_window, i_x+i_window]
                      if comb_map[i_y+size_window, i_x+i_window] :
                        n_node_available = n_node_available + 1
                  #to the left
                  if i_x - size_window > 0:
                    if i_y - i_window > 0 :
                      left_min_available = comb_map[i_y-i_window, i_x-size_window]
                      if comb_map[i_y-i_window, i_x-size_window] :
                        n_node_available = n_node_available + 1
                    if i_y + i_window < len(dict_sample['y_L']):
                      left_max_available = comb_map[i_y+i_window, i_x-size_window]
                      if comb_map[i_y+i_window, i_x-size_window] :
                        n_node_available = n_node_available + 1
                  #to the right
                  if i_x + size_window < len(dict_sample['x_L']):
                    if i_x - i_window > 0 :
                      right_min_available = comb_map[i_y-i_window, i_x+size_window]
                      if comb_map[i_y-i_window, i_x+size_window] :
                        n_node_available = n_node_available + 1
                    if i_y + i_window < len(dict_sample['y_L']):
                      right_max_available = comb_map[i_y+i_window, i_x+size_window]
                      if comb_map[i_y+i_window, i_x+size_window] :
                        n_node_available = n_node_available + 1

                  #move solute if et least one node is available
                  if n_node_available != 0 :
                    #to the top
                    if top_min_available:
                      c_map_new[i_y-size_window, i_x-i_window] = c_map_new[i_y-size_window, i_x-i_window] + solute_to_move/n_node_available
                    if top_max_available:
                      c_map_new[i_y-size_window, i_x+i_window] = c_map_new[i_y-size_window, i_x+i_window] + solute_to_move/n_node_available
                    #to the down
                    if down_min_available:
                      c_map_new[i_y+size_window, i_x-i_window] = c_map_new[i_y+size_window, i_x-i_window] + solute_to_move/n_node_available
                    if down_max_available:
                      c_map_new[i_y+size_window, i_x+i_window] = c_map_new[i_y+size_window, i_x+i_window] + solute_to_move/n_node_available
                    #to the left
                    if left_min_available:
                      c_map_new[i_y-i_window, i_x-size_window] = c_map_new[i_y-i_window, i_x-size_window] + solute_to_move/n_node_available
                    if left_max_available:
                      c_map_new[i_y+i_window, i_x-size_window] = c_map_new[i_y+i_window, i_x-size_window] + solute_to_move/n_node_available
                    #to the right
                    if right_min_available:
                      c_map_new[i_y-i_window, i_x+size_window] = c_map_new[i_y-i_window, i_x+size_window] + solute_to_move/n_node_available
                    if right_max_available:
                      c_map_new[i_y+i_window, i_x+size_window] = c_map_new[i_y+i_window, i_x+size_window] + solute_to_move/n_node_available
                    c_map_new[i_y, i_x] = dict_user['C_eq']
                    solute_moved = True
                i_window = i_window + 1
              size_window = size_window + 1   

        # push solute in of the solid
        if not comb_map[i_y, i_x] and c_map[i_y, i_x] < dict_user['C_eq']: # threshold value
          solute_moved = False
          size_window = 1
          # compute solute to move
          solute_to_move = dict_user['C_eq'] - c_map[i_y, i_x]
          while not solute_moved :
            i_window = 0
            while not solute_moved and i_window <= size_window:
              n_node_available = 0

              #Look to move horizontaly and vertically
              if i_window == 0 :
                top_available = False
                down_available = False
                left_available = False
                right_available = False
                #to the top
                if i_y - size_window > 0:
                  top_available = comb_map[i_y-size_window, i_x]
                  if comb_map[i_y-size_window, i_x] :
                    n_node_available = n_node_available + 1
                #to the down
                if i_y + size_window < len(dict_sample['y_L']):
                  down_available = comb_map[i_y+size_window, i_x]
                  if comb_map[i_y+size_window, i_x] :
                    n_node_available = n_node_available + 1
                #to the left
                if i_x - size_window > 0:
                  left_available = comb_map[i_y, i_x-size_window]
                  if comb_map[i_y, i_x-size_window] :
                    n_node_available = n_node_available + 1
                #to the right
                if i_x + size_window < len(dict_sample['x_L']):
                  right_available = comb_map[i_y, i_x+size_window]
                  if comb_map[i_y, i_x+size_window] :
                    n_node_available = n_node_available + 1

                #move solute if et least one node is available
                if n_node_available != 0 :
                  #to the top
                  if top_available:
                    c_map_new[i_y-size_window, i_x] = c_map_new[i_y-size_window, i_x] - solute_to_move/n_node_available
                  #to the down
                  if down_available:
                    c_map_new[i_y+size_window, i_x] = c_map_new[i_y+size_window, i_x] - solute_to_move/n_node_available
                  #to the left
                  if left_available:
                    c_map_new[i_y, i_x-size_window] = c_map_new[i_y, i_x-size_window] - solute_to_move/n_node_available
                  #to the right
                  if right_available:
                    c_map_new[i_y, i_x+size_window] = c_map_new[i_y, i_x+size_window] - solute_to_move/n_node_available
                  c_map_new[i_y, i_x] = 1
                  solute_moved = True

              #Look to move diagonally
              else :
                top_min_available = False
                top_max_available = False
                down_min_available = False
                down_max_available = False
                left_min_available = False
                left_max_available = False
                right_min_available = False
                right_max_available = False
                #to the top
                if i_y - size_window > 0:
                  if i_x - i_window > 0 :
                    top_min_available = comb_map[i_y-size_window, i_x-i_window]
                    if comb_map[i_y-size_window, i_x-i_window] :
                      n_node_available = n_node_available + 1
                  if i_x + i_window < len(dict_sample['x_L']):
                    top_max_available = comb_map[i_y-size_window, i_x+i_window]
                    if comb_map[i_y-size_window, i_x+i_window] :
                      n_node_available = n_node_available + 1
                #to the down
                if i_y + size_window < len(dict_sample['y_L']):
                  if i_x - i_window > 0 :
                    down_min_available = comb_map[i_y+size_window, i_x-i_window]
                    if comb_map[i_y+size_window, i_x-i_window] :
                      n_node_available = n_node_available + 1
                  if i_x + i_window < len(dict_sample['x_L']):
                    down_max_available = comb_map[i_y+size_window, i_x+i_window]
                    if comb_map[i_y+size_window, i_x+i_window] :
                      n_node_available = n_node_available + 1
                #to the left
                if i_x - size_window > 0:
                  if i_y - i_window > 0 :
                    left_min_available = comb_map[i_y-i_window, i_x-size_window]
                    if comb_map[i_y-i_window, i_x-size_window] :
                      n_node_available = n_node_available + 1
                  if i_y + i_window < len(dict_sample['y_L']):
                    left_max_available = comb_map[i_y+i_window, i_x-size_window]
                    if comb_map[i_y+i_window, i_x-size_window] :
                      n_node_available = n_node_available + 1
                #to the right
                if i_x + size_window < len(dict_sample['x_L']):
                  if i_x - i_window > 0 :
                    right_min_available = comb_map[i_y-i_window, i_x+size_window]
                    if comb_map[i_y-i_window, i_x+size_window] :
                      n_node_available = n_node_available + 1
                  if i_y + i_window < len(dict_sample['y_L']):
                    right_max_available = comb_map[i_y+i_window, i_x+size_window]
                    if comb_map[i_y+i_window, i_x+size_window] :
                      n_node_available = n_node_available + 1

                #move solute if et least one node is available
                if n_node_available != 0 :
                  #to the top
                  if top_min_available:
                    c_map_new[i_y-size_window, i_x-i_window] = c_map_new[i_y-size_window, i_x-i_window] - solute_to_move/n_node_available
                  if top_max_available:
                    c_map_new[i_y-size_window, i_x+i_window] = c_map_new[i_y-size_window, i_x+i_window] - solute_to_move/n_node_available
                  #to the down
                  if down_min_available:
                    c_map_new[i_y+size_window, i_x-i_window] = c_map_new[i_y+size_window, i_x-i_window] - solute_to_move/n_node_available
                  if down_max_available:
                    c_map_new[i_y+size_window, i_x+i_window] = c_map_new[i_y+size_window, i_x+i_window] - solute_to_move/n_node_available
                  #to the left
                  if left_min_available:
                    c_map_new[i_y-i_window, i_x-size_window] = c_map_new[i_y-i_window, i_x-size_window] - solute_to_move/n_node_available
                  if left_max_available:
                    c_map_new[i_y+i_window, i_x-size_window] = c_map_new[i_y+i_window, i_x-size_window] - solute_to_move/n_node_available
                  #to the right
                  if right_min_available:
                    c_map_new[i_y-i_window, i_x+size_window] = c_map_new[i_y-i_window, i_x+size_window] - solute_to_move/n_node_available
                  if right_max_available:
                    c_map_new[i_y+i_window, i_x+size_window] = c_map_new[i_y+i_window, i_x+size_window] - solute_to_move/n_node_available
                  c_map_new[i_y, i_x] = dict_user['C_eq']
                  solute_moved = True
              i_window = i_window + 1
            size_window = size_window + 1   
      
  # extract solute from the well
  # track mean value
  m_c_well = 0
  n_c_well = 0 
  # iterate on the mesh
  for i_x in range(len(dict_sample['x_L'])):
    for i_y in range(len(dict_sample['y_L'])):  
      if dict_sample['y_L'][i_y] > dict_user['y_max'] - dict_user['size_solute_well']:
        # track the mean value in the upper zone
        m_c_well = m_c_well + c_map_new[-1-i_y, i_x]
        n_c_well = n_c_well + 1
        # change value to have a well zone
        c_map_new[-1-i_y, i_x] = dict_user['C_eq']
  # track mean
  dict_user['m_c_well_L'].append(m_c_well/n_c_well) 

  # save data
  dict_sample['c_map'] = c_map_new

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
    # tube at the left
    if dict_sample['x_L'][i_x] <= dict_user['x_min'] + dict_user['size_tube']: 
      i_y = 0
      while not comb_map[i_y, i_x]:
        comb_map[i_y, i_x] = True
        i_y = i_y + 1
    # tube at the right
    if dict_user['x_max'] - dict_user['size_tube'] < dict_sample['x_L'][i_x]: 
      i_y = 0
      while not comb_map[i_y, i_x]:
        comb_map[i_y, i_x] = True
        i_y = i_y + 1
    # other tubes
    for coordinate_tube in dict_user['L_coordinates_tube']:
      if coordinate_tube - dict_user['size_tube']/2 < dict_sample['x_L'][i_x] and \
          dict_sample['x_L'][i_x] <= coordinate_tube + dict_user['size_tube']/2 : 
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
    elif j == 81:
      line = line[:-1] + "'1 "+str(dict_user['kappa_eta'])+" 1'\n"
    elif j == 103:
      line = line[:-1] + ' ' + str(dict_user['Energy_barrier'])+"'\n"
    elif j == 116:
      line = line[:-1] + "'1 " + str(dict_user['k_diss']) + ' ' + str(dict_user['k_prec']) + "'\n"
    elif j == 180:
      line = line[:-1] + ' ' + str(dict_user['dt_PF']*dict_user['n_t_PF']) +'\n'
    elif j == 173 or j == 174 or j == 176 or j == 177:
      line = line[:-1] + ' ' + str(dict_user['crit_res']) +'\n'
    elif j == 184:
      line = line[:-1] + ' ' + str(dict_user['dt_PF']) +'\n'
    file_to_write.write(line)

  file_to_write.close()
