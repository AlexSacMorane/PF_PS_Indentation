# -*- encoding=utf-8 -*-

from pathlib import Path
import shutil, os, pickle, math, vtk
import numpy as np
import matplotlib.pyplot as plt
from vtk.util.numpy_support import vtk_to_numpy

#------------------------------------------------------------------------------------------------------------------------------------------ #

def create_folder(name):
    '''
    Create a new folder. If it already exists, it is erased.
    '''
    if Path(name).exists():
        shutil.rmtree(name)
    os.mkdir(name)

#------------------------------------------------------------------------------------------------------------------------------------------ #

def plot_ed(dict_user, dict_sample):
    '''
    Plot the tilting term ed (and Ed).
    Ed(eta) = ed*h(eta)
    '''
    # init
    ed_map = np.zeros((dict_user['n_mesh_y'], dict_user['n_mesh_x']))
    Ed_map = np.zeros((dict_user['n_mesh_y'], dict_user['n_mesh_x']))
    # Iterate on the mesh
    for i_x in range(dict_user['n_mesh_x']):
        for i_y in range(dict_user['n_mesh_y']):
            c_ij = dict_sample['c_map'][i_y, i_x]
            as_ij = dict_sample['as_map'][i_y, i_x]
            eta_ij = dict_sample['eta_map'][i_y, i_x]
            # dissolution
            if c_ij < dict_user['C_eq']*as_ij:
                ed_map[i_y, i_x] = dict_user['k_diss']*as_ij*(1-c_ij/(dict_user['C_eq']*as_ij))
                Ed_map[i_y, i_x] = ed_map[i_y, i_x]*(3*eta_ij**2 - 2*eta_ij**3)
            # precipitation
            else :
                ed_map[i_y, i_x] = dict_user['k_prec']*as_ij*(1-c_ij/(dict_user['C_eq']*as_ij))
                Ed_map[i_y, i_x] = ed_map[i_y, i_x]*(3*eta_ij**2 - 2*eta_ij**3)

    # plot
    fig, (ax1, ax2) = plt.subplots(1,2,figsize=(16,9))
    # ed
    im = ax1.imshow(ed_map, interpolation = 'nearest', extent=(dict_sample['x_L'][0],dict_sample['x_L'][-1],dict_sample['y_L'][0],dict_sample['y_L'][-1]))
    fig.colorbar(im, ax=ax1)
    ax1.set_title(r'Map of $e_d$',fontsize = 30)
    # Ed
    im = ax2.imshow(Ed_map, interpolation = 'nearest', extent=(dict_sample['x_L'][0],dict_sample['x_L'][-1],dict_sample['y_L'][0],dict_sample['y_L'][-1]))
    fig.colorbar(im, ax=ax2)
    ax2.set_title(r'Map of $E_d(\eta)=e_d\times h(\eta)$',fontsize = 30)
    # close
    fig.tight_layout()
    fig.savefig('plot/map_tilt.png')
    plt.close(fig)

#------------------------------------------------------------------------------------------------------------------------------------------ #

def compute_ed_in_film(dict_user, dict_sample):
    '''
    Plot the tilting term ed in the thin fluid film.
    '''
    # init
    L_ed = []
    # Iterate on the mesh
    for i_x in range(dict_user['n_mesh_x']):
        # search i_y near the front coordinate
        L_search = list(abs(np.array(dict_sample['y_L']-dict_sample['current_front'][i_x])))
        i_y = L_search.index(min(L_search))
        # search data needed
        c_ij = dict_sample['c_map'][-1-i_y, i_x]
        as_ij = dict_sample['as_map'][-1-i_y, i_x]
        # dissolution
        if c_ij < dict_user['C_eq']*as_ij:
            ed_ij = dict_user['k_diss']*as_ij*(1-c_ij/(dict_user['C_eq']*as_ij))
        # precipitation
        else :
            ed_ij = dict_user['k_prec']*as_ij*(1-c_ij/(dict_user['C_eq']*as_ij))
        L_ed.append(ed_ij)
    # save data
    dict_user['L_L_ed_in_film'].append(L_ed)
    dict_user['L_m_ed_in_film'].append(np.mean(L_ed))

#------------------------------------------------------------------------------------------------------------------------------------------ #

def plot_ed_in_film(dict_user, dict_sample):
    '''
    Plot the evolution of the profile of the tilting term ed in the thin fluid film and the mean value.
    '''
    # plot
    fig, (ax1) = plt.subplots(1,1,figsize=(16,9))
    ax1.plot(dict_user['time_L'], dict_user['L_m_ed_in_film'])
    ax1.set_xlabel('Time (-)')
    ax1.set_ylabel('Mean ed in the fluid film (-)')
    fig.tight_layout()
    fig.savefig('plot/evol_m_ed_film_t.png')
    plt.close(fig)

    # plot
    fig, (ax1) = plt.subplots(1,1,figsize=(16,9))
    for i_L_ed_in_film in range(len(dict_user['L_L_ed_in_film'])):
        L_ed_in_film = dict_user['L_L_ed_in_film'][i_L_ed_in_film]
        ax1.plot(dict_sample['x_L'], L_ed_in_film, label='t='+str(dict_user['time_L'][i_L_ed_in_film]))
    ax1.legend()
    ax1.set_xlabel('x coordinate (-)')
    ax1.set_ylabel('Ed in the fluid film (-)')
    fig.tight_layout()
    fig.savefig('plot/evol_profile_ed_film_t.png')
    plt.close(fig)


#------------------------------------------------------------------------------------------------------------------------------------------ #

def plot_config(dict_user, dict_sample):
    '''
    Plot the map of the configuration (solute and phase).
    '''
    # plot
    fig, (ax1, ax2) = plt.subplots(1,2,figsize=(16,9))
    # solute
    im = ax1.imshow(dict_sample['eta_map'], interpolation = 'nearest', extent=(dict_sample['x_L'][0],dict_sample['x_L'][-1],dict_sample['y_L'][0],dict_sample['y_L'][-1]))
    fig.colorbar(im, ax=ax1)
    ax1.set_title(r'Map of $\eta$',fontsize = 30)
    # phase
    im = ax2.imshow(dict_sample['c_map'], interpolation = 'nearest', extent=(dict_sample['x_L'][0],dict_sample['x_L'][-1],dict_sample['y_L'][0],dict_sample['y_L'][-1]))
    fig.colorbar(im, ax=ax2)
    ax2.set_title(r'Map of $c$',fontsize = 30)
    # close
    fig.tight_layout()
    fig.savefig('plot/map_config.png')
    plt.close(fig)

#------------------------------------------------------------------------------------------------------------------------------------------ #

def plot_front(dict_user, dict_sample):
    '''
    Plot the evolution of the y coordinate of the front.
    '''
    # plot
    fig, (ax1) = plt.subplots(1,1,figsize=(16,9))
    ax1.plot(dict_user['time_L'], dict_user['y_front_L'])
    ax1.set_xlabel('Time (-)')
    ax1.set_ylabel('y coordinate of the front (-)')
    fig.tight_layout()
    fig.savefig('plot/evol_front_t.png')
    plt.close(fig)
    
    # plot
    fig, (ax1) = plt.subplots(1,1,figsize=(16,9))
    ax1.plot(dict_user['time_L'], dict_user['min_y_front_L'])
    ax1.plot(dict_user['time_L'], dict_user['max_y_front_L'])
    ax1.set_xlabel('Time (-)')
    ax1.set_ylabel('Minn-Max y coordinate of the front (-)')
    fig.tight_layout()
    fig.savefig('plot/evol_min_max_front_t.png')
    plt.close(fig)

#------------------------------------------------------------------------------------------------------------------------------------------ #

def plot_m_c_well(dict_user, dict_sample):
    '''
    Plot the evolution of the mean concentration of the solute in the well.
    '''
    # compute the solute concentration at the equilibrium considering the solid activity
    R_gas = 82.06e5 # cm3 Pa K-1 mol-1
    Temp = 25+278   # K
    V_m = 27.1      # cm3 mol-1
    a_s = math.exp(dict_user['pressure_applied']*V_m/(R_gas*Temp))
    c_eq_as = dict_user['C_eq']*a_s

    # plot
    fig, (ax1) = plt.subplots(1,1,figsize=(16,9))
    ax1.plot(dict_user['m_c_well_L'])
    ax1.plot([0, len(dict_user['m_c_well_L'])-1], [c_eq_as, c_eq_as], linestyle='dotted')
    ax1.set_xlabel('Iteration (-)')
    ax1.set_ylabel('mean concentration in the well (-)')
    fig.tight_layout()
    fig.savefig('plot/evol_m_c_weel_ite.png')
    plt.close(fig)

    # pp data
    L_sat = []
    for m_c in dict_user['m_c_well_L']:
        sat = (m_c-dict_user['C_eq'])/(c_eq_as-dict_user['C_eq'])*100
        L_sat.append(sat)

    # plot
    fig, (ax1) = plt.subplots(1,1,figsize=(16,9))
    ax1.plot(L_sat)
    ax1.set_xlabel('Iteration (-)')
    ax1.set_ylabel('saturation in the well (%)')
    fig.tight_layout()
    fig.savefig('plot/evol_sat_weel_ite.png')
    plt.close(fig)

#------------------------------------------------------------------------------------------------------------------------------------------ #

def plot_fit(dict_user, dict_sample):
    '''
    Plot the evolution of the log(disp) - log(Times) and compute a fit (y = ax + b).
    '''
    # compute the pp data
    log_disp = []
    log_times = []
    for i in range(1, len(dict_user['y_front_L'])):
        log_times.append(math.log(dict_user['time_L'][i]))
        disp = (dict_user['y_front_L'][0]-dict_user['y_front_L'][i])
        log_disp.append(math.log(disp))

    # compute the fit
    coeff, const, corr = lsm_linear(log_disp, log_times)
    L_fit = []
    # compute fitted Andrade creep law
    for i in range(len(log_times)):
        L_fit.append(const + coeff*log_times[i])

    # plot
    fig, (ax1) = plt.subplots(1,1,figsize=(16,9))
    ax1.plot(log_times, log_disp, label='Data')
    ax1.plot(log_times, L_fit, linestyle='dotted', color='k',\
             label='Fit : y = '+str(round(coeff,2))+'x + '+str(round(const,2))+' ('+str(round(corr,2))+')')
    ax1.set_ylabel(r'log(displacement) (-)')
    ax1.set_xlabel('log(Times) (-)')   
    ax1.legend()
    fig.tight_layout()
    fig.savefig('plot/evol_log_disp_log_t.png')
    plt.close(fig)

# ------------------------------------------------------------------------------------------------------------------------------------------ #

def lsm_linear(L_y, L_x):
    '''
    Least square method to determine y = ax + b
    '''
    # compute sums
    s_1 = 0
    s_2 = 0
    s_3 = 0
    s_4 = 0
    s_5 = 0
    for i in range(len(L_y)):
        s_1 = s_1 + 1*L_x[i]*L_x[i]
        s_2 = s_2 + 1*L_x[i]
        s_3 = s_3 + 1
        s_4 = s_4 + 1*L_x[i]*L_y[i]
        s_5 = s_5 + 1*L_y[i]
    # solve problem
    M = np.array([[s_1, s_2],[s_2, s_3]])
    V = np.array([s_4, s_5])
    result = np.linalg.solve(M, V)
    a = result[0]
    b = result[1]
    # correlation linear
    cov = 0
    vx = 0
    vy = 0
    for i in range(len(L_y)):
        cov = cov + (L_x[i]-np.mean(L_x))*(L_y[i]-np.mean(L_y))
        vx = vx + (L_x[i]-np.mean(L_x))*(L_x[i]-np.mean(L_x))
        vy = vy + (L_y[i]-np.mean(L_y))*(L_y[i]-np.mean(L_y))
    corr = cov/(math.sqrt(vx*vy))
    return a, b, corr

#------------------------------------------------------------------------------------------------------------------------------------------ #

def reduce_n_vtk_files(dict_user, dict_sample):
    '''
    Reduce the number of vtk files for phase-field and dem.

    Warning ! The pf and dem files are not synchronized...
    '''
    if dict_user['n_max_vtk_files'] != None:
        # Phase Field files

        # compute the frequency
        if dict_user['j_total']-1 > dict_user['n_max_vtk_files']:
            f_save = (dict_user['j_total']-1)/(dict_user['n_max_vtk_files']-1)
        else :
            f_save = 1
        # post proccess index
        i_save = 0

        # iterate on time 
        for iteration in range(dict_user['j_total']):
            iteration_str = index_to_str(iteration) # from pf_to_dem.py 
            if iteration >= f_save*i_save:
                i_save_str = index_to_str(i_save) # from pf_to_dem.py
                # rename .pvtu
                os.rename('vtk/pf_'+iteration_str+'.pvtu','vtk/pf_'+i_save_str+'.pvtu')
                # write .pvtu to save all vtk
                file = open('vtk/pf_'+i_save_str+'.pvtu','w')
                file.write('''<?xml version="1.0"?>
                <VTKFile type="PUnstructuredGrid" version="0.1" byte_order="LittleEndian" header_type="UInt32" compressor="vtkZLibDataCompressor">
                \t<PUnstructuredGrid GhostLevel="1">
                \t\t<PPointData>
                \t\t\t<PDataArray type="Float64" Name="as"/>
                \t\t\t<PDataArray type="Float64" Name="kc"/>
                \t\t\t<PDataArray type="Float64" Name="eta"/>
                \t\t\t<PDataArray type="Float64" Name="c"/>
                \t\t</PPointData>
                \t\t<PCellData>
                \t\t\t<PDataArray type="Int32" Name="libmesh_elem_id"/>
                \t\t\t<PDataArray type="Int32" Name="subdomain_id"/>
                \t\t\t<PDataArray type="Int32" Name="processor_id"/>
                \t\t</PCellData>
                \t\t<PPoints>
                \t\t\t<PDataArray type="Float64" Name="Points" NumberOfComponents="3"/>
                \t\t</PPoints>''')
                line = ''
                for i_proc in range(dict_user['n_proc']):
                    line = line + '''\t\t<Piece Source="pf_'''+i_save_str+'''_'''+str(i_proc)+'''.vtu"/>\n'''
                file.write(line)
                file.write('''\t</PUnstructuredGrid>
                </VTKFile>''')
                file.close()
                # rename .vtk
                for i_proc in range(dict_user['n_proc']):
                    os.rename('vtk/pf_'+iteration_str+'_'+str(i_proc)+'.vtu','vtk/pf_'+i_save_str+'_'+str(i_proc)+'.vtu')
                i_save = i_save + 1 
            else:
                # delete files
                os.remove('vtk/pf_'+iteration_str+'.pvtu')
                for i_proc in range(dict_user['n_proc']):
                    os.remove('vtk/pf_'+iteration_str+'_'+str(i_proc)+'.vtu')
        # .e file
        os.remove('vtk/pf_out.e')
        # other files
        j = 0
        j_str = index_to_str(j)
        filepath = Path('vtk/pf_other_'+j_str+'.pvtu')
        while filepath.exists():
            for i_proc in range(dict_user['n_proc']):
                os.remove('vtk/pf_other_'+j_str+'_'+str(i_proc)+'.vtu')
            os.remove('vtk/pf_other_'+j_str+'.pvtu')
            j = j + 1
            j_str = index_to_str(j)
            filepath = Path('vtk/pf_other_'+j_str+'.pvtu')

# -----------------------------------------------------------------------------#

def sort_files(dict_user, dict_sample):
     '''
     Sort files generated by MOOSE to different directories
     '''
     os.rename('pf_out.e','vtk/pf_out.e')
     os.rename('pf.i','input/pf.i')
     j = 0
     j_str = index_to_str(j)
     j_total_str = index_to_str(dict_user['j_total'])
     filepath = Path('pf_other_'+j_str+'.pvtu')
     while filepath.exists():
         for i_proc in range(dict_user['n_proc']):
            os.rename('pf_other_'+j_str+'_'+str(i_proc)+'.vtu','vtk/pf_other_'+j_str+'_'+str(i_proc)+'.vtu')
            shutil.copyfile('vtk/pf_other_'+j_str+'_'+str(i_proc)+'.vtu','vtk/pf_'+j_total_str+'_'+str(i_proc)+'.vtu')
         os.rename('pf_other_'+j_str+'.pvtu','vtk/pf_other_'+j_str+'.pvtu')
         # write .pvtu to save all vtk
         file = open('vtk/pf_'+j_total_str+'.pvtu','w')
         file.write('''<?xml version="1.0"?>
         <VTKFile type="PUnstructuredGrid" version="0.1" byte_order="LittleEndian" header_type="UInt32" compressor="vtkZLibDataCompressor">
         \t<PUnstructuredGrid GhostLevel="1">
         \t\t<PPointData>
         \t\t\t<PDataArray type="Float64" Name="as"/>
         \t\t\t<PDataArray type="Float64" Name="kc"/>
         \t\t\t<PDataArray type="Float64" Name="eta"/>
         \t\t\t<PDataArray type="Float64" Name="c"/>
         \t\t</PPointData>
         \t\t<PCellData>
         \t\t\t<PDataArray type="Int32" Name="libmesh_elem_id"/>
         \t\t\t<PDataArray type="Int32" Name="subdomain_id"/>
         \t\t\t<PDataArray type="Int32" Name="processor_id"/>
         \t\t</PCellData>
         \t\t<PPoints>
         \t\t\t<PDataArray type="Float64" Name="Points" NumberOfComponents="3"/>
         \t\t</PPoints>''')
         line = ''
         for i_proc in range(dict_user['n_proc']):
             line = line + '''\t\t<Piece Source="pf_'''+j_total_str+'''_'''+str(i_proc)+'''.vtu"/>\n'''
         file.write(line)
         file.write('''\t</PUnstructuredGrid>
         </VTKFile>''')
         file.close()
         j = j + 1
         j_str = index_to_str(j)
         filepath = Path('pf_other_'+j_str+'.pvtu')
         dict_user['j_total'] = dict_user['j_total'] + 1
         j_total_str = index_to_str(dict_user['j_total'])
     return index_to_str(j-1)

# -----------------------------------------------------------------------------#

def index_to_str(j):
    '''
    Convert a integer into a string with the format XXX.
    '''
    if j < 10:
        return '00'+str(j)
    elif 10<=j and j<100:
        return '0'+str(j)
    else :
        return str(j)

#------------------------------------------------------------------------------------------------------------------------------------------ #

def save_mesh_database(dict_user, dict_sample):
    '''
    Save mesh database.
    '''
    # creating a database
    if not Path('mesh_map.database').exists():
        dict_data = {
        'n_proc': dict_user['n_proc'],
        'x_min': min(dict_sample['x_L']),
        'x_max': max(dict_sample['x_L']),
        'y_min': min(dict_sample['y_L']),
        'y_max': max(dict_sample['y_L']),
        'n_mesh_x': len(dict_sample['x_L']),
        'n_mesh_y': len(dict_sample['y_L']),
        'L_L_i_XYZ_used': dict_sample['L_L_i_XYZ_used'],
        'L_XYZ': dict_sample['L_XYZ']
        }
        dict_database = {'Run_1': dict_data}
        with open('mesh_map.database', 'wb') as handle:
                pickle.dump(dict_database, handle, protocol=pickle.HIGHEST_PROTOCOL)
    # updating a database
    else :
        with open('mesh_map.database', 'rb') as handle:
            dict_database = pickle.load(handle)
        dict_data = {
        'n_proc': dict_user['n_proc'],
        'x_min': min(dict_sample['x_L']),
        'x_max': max(dict_sample['x_L']),
        'y_min': min(dict_sample['y_L']),
        'y_max': max(dict_sample['y_L']),
        'n_mesh_x': len(dict_sample['x_L']),
        'n_mesh_y': len(dict_sample['y_L']),
        'L_L_i_XYZ_used': dict_sample['L_L_i_XYZ_used'],
        'L_XYZ': dict_sample['L_XYZ']
        }   
        mesh_map_known = False
        for i_run in range(1,len(dict_database.keys())+1):
            if dict_database['Run_'+str(int(i_run))] == dict_data:
                mesh_map_known = True
        # new entry
        if not mesh_map_known: 
            key_entry = 'Run_'+str(int(len(dict_database.keys())+1))
            dict_database[key_entry] = dict_data
            with open('mesh_map.database', 'wb') as handle:
                pickle.dump(dict_database, handle, protocol=pickle.HIGHEST_PROTOCOL)

#------------------------------------------------------------------------------------------------------------------------------------------ #

def check_mesh_database(dict_user, dict_sample):
    '''
    Check mesh database.
    '''
    if Path('mesh_map.database').exists():
        with open('mesh_map.database', 'rb') as handle:
            dict_database = pickle.load(handle)
        dict_data = {
        'n_proc': dict_user['n_proc'],
        'x_min': min(dict_sample['x_L']),
        'x_max': max(dict_sample['x_L']),
        'y_min': min(dict_sample['y_L']),
        'y_max': max(dict_sample['y_L']),
        'n_mesh_x': len(dict_sample['x_L']),
        'n_mesh_y': len(dict_sample['y_L'])
        }   
        mesh_map_known = False
        for i_run in range(1,len(dict_database.keys())+1):
            if dict_database['Run_'+str(int(i_run))]['n_proc'] == dict_user['n_proc'] and\
            dict_database['Run_'+str(int(i_run))]['x_min'] == min(dict_sample['x_L']) and\
            dict_database['Run_'+str(int(i_run))]['x_max'] == max(dict_sample['x_L']) and\
            dict_database['Run_'+str(int(i_run))]['y_min'] == min(dict_sample['y_L']) and\
            dict_database['Run_'+str(int(i_run))]['y_max'] == max(dict_sample['y_L']) and\
            dict_database['Run_'+str(int(i_run))]['n_mesh_x'] == len(dict_sample['x_L']) and\
            dict_database['Run_'+str(int(i_run))]['n_mesh_y'] == len(dict_sample['y_L']) :
                mesh_map_known = True
                i_known = i_run
        if mesh_map_known :
            dict_sample['Map_known'] = True
            dict_sample['L_L_i_XYZ_used'] = dict_database['Run_'+str(int(i_known))]['L_L_i_XYZ_used']
            dict_sample['L_XYZ'] = dict_database['Run_'+str(int(i_known))]['L_XYZ']
        else :
            dict_sample['Map_known'] = False
    else :
        dict_sample['Map_known'] = False

# -----------------------------------------------------------------------------#

def read_vtk(dict_user, dict_sample, j_str):
    '''
    Read the last vtk files to obtain data from MOOSE.

    Do not work calling yade.
    '''
    eta_map_old = dict_sample['eta_map'].copy()
    c_map_old = dict_sample['c_map'].copy()
    L_eta = []
    L_c = []
    if not dict_sample['Map_known']:
        L_limits = []
        L_XYZ = []
        L_L_i_XYZ_used = []
    else :
        L_XYZ = dict_sample['L_XYZ']

    # iterate on the proccessors used
    for i_proc in range(dict_user['n_proc']):

        # name of the file to load
        namefile = 'vtk/pf_other_'+j_str+'_'+str(i_proc)+'.vtu'

        # load a vtk file as input
        reader = vtk.vtkXMLUnstructuredGridReader()
        reader.SetFileName(namefile)
        reader.Update()

        # Grab a scalar from the vtk file
        nodes_vtk_array = reader.GetOutput().GetPoints().GetData()
        eta_vtk_array = reader.GetOutput().GetPointData().GetArray("eta")
        c_vtk_array = reader.GetOutput().GetPointData().GetArray("c")

        #Get the coordinates of the nodes and the scalar values
        nodes_array = vtk_to_numpy(nodes_vtk_array)
        eta_array = vtk_to_numpy(eta_vtk_array)
        c_array = vtk_to_numpy(c_vtk_array)

        # map is not know
        if not dict_sample['Map_known']:
            # look for limits
            x_min = None
            x_max = None
            y_min = None
            y_max = None
            # save the map
            L_i_XYZ_used = []
            # Must detect common zones between processors
            for i_XYZ in range(len(nodes_array)) :
                XYZ = nodes_array[i_XYZ]
                # Do not consider twice a point
                if list(XYZ) not in L_XYZ :
                    L_XYZ.append(list(XYZ))
                    L_eta.append(eta_array[i_XYZ])
                    L_c.append(c_array[i_XYZ])
                    L_i_XYZ_used.append(i_XYZ)
                    # set first point
                    if x_min == None :
                        x_min = list(XYZ)[0]
                        x_max = list(XYZ)[0]
                        y_min = list(XYZ)[1]
                        y_max = list(XYZ)[1]
                    # look for limits of the processor
                    else :
                        if list(XYZ)[0] < x_min:
                            x_min = list(XYZ)[0]
                        if list(XYZ)[0] > x_max:
                            x_max = list(XYZ)[0]
                        if list(XYZ)[1] < y_min:
                            y_min = list(XYZ)[1]
                        if list(XYZ)[1] > y_max:
                            y_max = list(XYZ)[1]
            # Here the algorithm can be help as the mapping is known
            L_L_i_XYZ_used.append(L_i_XYZ_used)
            # save limits
            L_limits.append([x_min,x_max,y_min,y_max])

        # map is known
        else :
            # the last term considered is at the end of the list
            if dict_sample['L_L_i_XYZ_used'][i_proc][-1] == len(nodes_array)-1:
                L_eta = list(L_eta) + list(eta_array)
                L_c = list(L_c) + list(c_array)
            # the last term considered is not at the end of the list
            else :
                L_eta = list(L_eta) + list(eta_array[dict_sample['L_L_i_XYZ_used'][i_proc][0]: dict_sample['L_L_i_XYZ_used'][i_proc][-1]+1])
                L_c = list(L_c) + list(c_array[dict_sample['L_L_i_XYZ_used'][i_proc][0]: dict_sample['L_L_i_XYZ_used'][i_proc][-1]+1])

    if not dict_sample['Map_known']:
        # plot processors distribution
        if 'processor' in dict_user['L_figures']:
            fig, (ax1) = plt.subplots(1,1,figsize=(16,9))
            # parameters
            title_fontsize = 20
            for i_proc in range(len(L_limits)):
                limits = L_limits[i_proc]
                ax1.plot([limits[0],limits[1],limits[1],limits[0],limits[0]],[limits[2],limits[2],limits[3],limits[3],limits[2]], label='proc '+str(i_proc))
            ax1.legend()
            ax1.set_xlabel('X (m)')
            ax1.set_ylabel('Y (m)')
            ax1.set_title('Processor i has the priority on i+1',fontsize = title_fontsize)
            fig.suptitle('Processors ditribution',fontsize = 1.2*title_fontsize)    
            fig.tight_layout()
            fig.savefig('plot/processors_distribution.png')
            plt.close(fig)
        # the map is known
        dict_sample['Map_known'] = True
        dict_sample['L_L_i_XYZ_used'] = L_L_i_XYZ_used
        dict_sample['L_XYZ'] = L_XYZ

    # rebuild map from lists
    for i_XYZ in range(len(L_XYZ)):
        # find nearest node
        L_search = list(abs(np.array(dict_sample['x_L']-L_XYZ[i_XYZ][0])))
        i_x = L_search.index(min(L_search))
        L_search = list(abs(np.array(dict_sample['y_L']-L_XYZ[i_XYZ][1])))
        i_y = L_search.index(min(L_search))
        # rewrite map
        dict_sample['eta_map'][-1-i_y, i_x] = L_eta[i_XYZ]
        dict_sample['c_map'][-1-i_y, i_x] = L_c[i_XYZ]

#------------------------------------------------------------------------------------------------------------------------------------------ #

def compute_mass(dict_user, dict_sample):
    '''
    Compute the mass at a certain time.
     
    Mass is sum of eta and c.
    '''
    # sum of masses
    dict_user['sum_eta_tempo'] = np.sum(dict_sample['eta_map'])
    dict_user['sum_c_tempo'] = np.sum(dict_sample['c_map'])
    dict_user['sum_mass_tempo'] = np.sum(dict_sample['eta_map'])+np.sum(dict_sample['c_map'])
    
#------------------------------------------------------------------------------------------------------------------------------------------ #

def compute_mass_loss(dict_user, dict_sample, tracker_key):
    '''
    Compute the mass loss from the previous compute_mass() call.
     
    Plot in the given tracker.
    Mass is sum of etai and c.
    '''
    # delta masses
    deta = np.sum(dict_sample['eta_map']) - dict_user['sum_eta_tempo']
    dc = np.sum(dict_sample['c_map']) - dict_user['sum_c_tempo']
    dm = np.sum(dict_sample['eta_map'])+np.sum(dict_sample['c_map']) - dict_user['sum_mass_tempo']
    
    # save
    dict_user[tracker_key+'_eta'].append(deta)
    dict_user[tracker_key+'_c'].append(dc)
    dict_user[tracker_key+'_m'].append(dm)

    # percentage
    dict_user[tracker_key+'_eta_p'].append(deta/dict_user['sum_eta_tempo']*100)
    dict_user[tracker_key+'_c_p'].append(dc/dict_user['sum_c_tempo']*100)
    dict_user[tracker_key+'_m_p'].append(dm/dict_user['sum_mass_tempo']*100)

    # plot
    if 'mass_loss' in dict_user['L_figures']:
        fig, (ax1,ax2,ax3) = plt.subplots(nrows=3,ncols=1,figsize=(16,9))
        ax1.plot(dict_user[tracker_key+'_eta'])
        ax1.set_title(r'$\eta$ loss (-)')
        ax2.plot(dict_user[tracker_key+'_c'])
        ax2.set_title(r'$c$ loss (-)')
        ax3.plot(dict_user[tracker_key+'_m'])
        ax3.set_title(r'$\eta$ + $c$ loss (-)')
        fig.tight_layout()
        fig.savefig('plot/evol_mass_loss_'+tracker_key+'_ite.png')
        plt.close(fig)

        fig, (ax1,ax2,ax3) = plt.subplots(nrows=3,ncols=1,figsize=(16,9))
        ax1.plot(dict_user[tracker_key+'_eta_p'])
        ax1.set_title(r'$\eta$ loss (%)')
        ax2.plot(dict_user[tracker_key+'_c_p'])
        ax2.set_title(r'$c$ loss (%)')
        ax3.plot(dict_user[tracker_key+'_m_p'])
        ax3.set_title(r'$\eta$ + $c$ loss (%)')
        fig.tight_layout()
        fig.savefig('plot/evol_mass_loss_'+tracker_key+'_p_ite.png')
        plt.close(fig)

#------------------------------------------------------------------------------------------------------------------------------------------ #

def plot_performances(dict_user, dict_sample):
    '''
    Plot figure illustrating the time performances of the algorithm.
    '''
    if 'performances' in dict_user['L_figures']:
        fig, (ax1) = plt.subplots(nrows=1,ncols=1,figsize=(16,9))
        ax1.plot(dict_user['L_t_dem'], label='DEM')
        ax1.plot(dict_user['L_t_pf'], label='PF')
        ax1.plot(dict_user['L_t_dem_to_pf'], label='DEM to PF')
        ax1.plot(dict_user['L_t_pf_to_dem_1'], label='PF to DEM 1')
        ax1.plot(dict_user['L_t_pf_to_dem_2'], label='PF to DEM 2')
        ax1.legend()
        ax1.set_title('Performances (s)')
        ax1.set_xlabel('Iterations (-)')
        fig.tight_layout()
        fig.savefig('plot/performances.png')
        plt.close(fig)

#------------------------------------------------------------------------------------------------------------------------------------------ #

def remesh(dict_user, dict_sample):
    '''
    Remesh the problem.
    
    Eta1, Eta2, c maps are updated
    x_L, n_mesh_x, y_L, n_mesh_y are updated.
    '''
    # search the grain boundaries
    y_min = dict_sample['y_L'][-1]
    y_max = dict_sample['y_L'][0]
    x_min = dict_sample['x_L'][-1]
    x_max = dict_sample['x_L'][0]
    # iterate on y
    for i_y in range(len(dict_sample['y_L'])):
        if max(dict_sample['eta_1_map'][-1-i_y, :]) > 0.5 or\
           max(dict_sample['eta_2_map'][-1-i_y, :]) > 0.5:
            if dict_sample['y_L'][i_y] < y_min : 
                y_min = dict_sample['y_L'][i_y]
            if dict_sample['y_L'][i_y] > y_max :
                y_max = dict_sample['y_L'][i_y]
    # iterate on x
    for i_x in range(len(dict_sample['x_L'])):
        if max(dict_sample['eta_1_map'][:, i_x]) > 0.5 or\
           max(dict_sample['eta_2_map'][:, i_x]) > 0.5:
            if dict_sample['x_L'][i_x] < x_min : 
                x_min = dict_sample['x_L'][i_x]
            if dict_sample['x_L'][i_x] > x_max :
                x_max = dict_sample['x_L'][i_x]
    # compute the domain boundaries (grain boundaries + margins)
    x_min_dom = x_min - dict_user['margin_mesh_domain']
    x_max_dom = x_max + dict_user['margin_mesh_domain']
    y_min_dom = y_min - dict_user['margin_mesh_domain']
    y_max_dom = y_max + dict_user['margin_mesh_domain']
    # compute the new x_L and y_L
    x_L = np.arange(x_min_dom, x_max_dom, dict_user['size_x_mesh'])
    n_mesh_x = len(x_L)
    y_L = np.arange(y_min_dom, y_max_dom, dict_user['size_y_mesh'])
    n_mesh_y = len(y_L)
    delta_x_max = 0
    delta_y_max = 0
    # compute the new maps
    eta_1_map = np.zeros((n_mesh_y, n_mesh_x))
    eta_2_map = np.zeros((n_mesh_y, n_mesh_x))
    c_map = np.ones((n_mesh_y, n_mesh_x))
    # iterate on lines
    for i_y in range(len(y_L)):
        # addition
        if y_L[i_y] < dict_sample['y_L'][0] or \
           dict_sample['y_L'][-1] < y_L[i_y]:
            # iterate on columns
            for i_x in range(len(x_L)):
                eta_1_map[-1-i_y, i_x] = 0
                eta_2_map[-1-i_y, i_x] = 0
                c_map[-1-i_y, i_x] = 1
        # extraction
        else :     
            # iterate on columns
            for i_x in range(len(x_L)):
                # addition
                if x_L[i_x] < dict_sample['x_L'][0] or \
                   dict_sample['x_L'][-1] < x_L[i_x]: 
                    eta_1_map[-1-i_y, i_x] = 0
                    eta_2_map[-1-i_y, i_x] = 0
                    c_map[-1-i_y, i_x] = 1
                # extraction
                else :
                    # find nearest node to old node
                    L_search = list(abs(np.array(dict_sample['x_L']-x_L[i_x])))
                    i_x_old = L_search.index(min(L_search))
                    delta_x = min(L_search)
                    L_search = list(abs(np.array(dict_sample['y_L']-y_L[i_y])))
                    i_y_old = L_search.index(min(L_search))
                    delta_y = min(L_search)
                    # track
                    if delta_x > delta_x_max:
                        delta_x_max = delta_x
                    if delta_y > delta_y_max:
                        delta_y_max = delta_y
                    # update
                    eta_1_map[-1-i_y, i_x] = dict_sample['eta_1_map'][-1-i_y_old, i_x_old]  
                    eta_2_map[-1-i_y, i_x] = dict_sample['eta_2_map'][-1-i_y_old, i_x_old]  
                    c_map[-1-i_y, i_x] = dict_sample['c_map'][-1-i_y_old, i_x_old]  
    # tracking
    dict_user['L_x_min_dom'].append(min(x_L))
    dict_user['L_x_max_dom'].append(max(x_L))
    dict_user['L_y_min_dom'].append(min(y_L))
    dict_user['L_y_max_dom'].append(max(y_L))
    dict_user['L_delta_x_max'].append(delta_x_max)
    dict_user['L_delta_y_max'].append(delta_y_max)
    # plot 
    if 'dim_dom' in dict_user['L_figures']:
        fig, (ax1) = plt.subplots(1,1,figsize=(16,9))
        ax1.plot(dict_user['L_x_min_dom'], label='x_min')
        ax1.plot(dict_user['L_x_max_dom'], label='x_max')
        ax1.plot(dict_user['L_y_min_dom'], label='y_min')
        ax1.plot(dict_user['L_y_max_dom'], label='y_max')
        ax1.legend(fontsize=20)
        ax1.set_title(r'Domain Dimensions',fontsize = 30)
        fig.tight_layout()
        fig.savefig('plot/dim_dom.png')
        plt.close(fig)

        fig, (ax1) = plt.subplots(1,1,figsize=(16,9))
        ax1.plot(dict_user['L_delta_x_max'], label=r'max $\Delta$x')
        ax1.plot(dict_user['L_delta_y_max'], label=r'max $\Delta$y')
        ax1.legend(fontsize=20)
        ax1.set_title(r'Interpolation errors',fontsize = 30)
        fig.tight_layout()
        fig.savefig('plot/dim_dom_delta.png')
        plt.close(fig)
    # save 
    dict_sample['x_L'] = x_L
    dict_user['n_mesh_x'] = n_mesh_x
    dict_sample['y_L'] = y_L
    dict_user['n_mesh_y'] = n_mesh_y
    dict_sample['eta_1_map'] = eta_1_map
    dict_sample['eta_2_map'] = eta_2_map
    dict_sample['c_map'] = c_map

                