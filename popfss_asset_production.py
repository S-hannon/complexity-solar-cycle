from __future__ import division
import glob
import os
import numpy as np
import pandas as pd
import matplotlib as mpl
import PIL.Image as Image
import datetime
import asset_production_tools as apt
import hi_processing as hip


def load_comp_events():
    """
    Function to load in the csv file containing the HELCATS CMEs.
    Import as a pandas dataframe.
    :return:
    """
    proj_dirs = apt.project_info()
    # Get converters for making sure times imported correctly
    convert = dict(start_time=pd.to_datetime, mid_time=pd.to_datetime,
                   end_time=pd.to_datetime)
    helcats_cmes = pd.read_csv(proj_dirs['comp_times'], converters=convert)
    return helcats_cmes
    

def make_output_directory_structure_comp():
    """
    Creates output directory structure.
    """
    # Get project directories
    proj_dirs = apt.project_info()
    # Loop over each event, and create a directory in outdata with a unique event name, from HELCATS id and ssw id       
    path = os.path.join(proj_dirs['out_data'], 'comp_assets')
    # If this directory doesnt exist, make it
    if not os.path.exists(path):
        os.makedirs(path)
    else:
        print(path + " exists already.")
    return


def find_nearest(array, value):
    """
    Finds the index of the array value nearest to input value.
    :param: array: array in which to search for nearest value
    :paran: value: value which wish to find nearest in array
    :return: index of nearest value in array
    """
    ds = []
    for i in range(len(array)):
        ds.append(array[i] - value)
    idx = (np.abs(ds)).argmin()
    return idx


def make_ssw_comp_index_assets():
    """
    Function to loop over the HELCATS CMEs, find all relevant HI1A and HI1B 1-day background images, and produce
    plain, differenced and relative difference images.
    :return:
    """
    # Get project directories and CME times
    proj_dirs = apt.project_info()
    helcats_cmes = load_comp_events()

    # Set values
    img_type_list = ['diff']

    # Loop over the HELCATS_cmes and estimate HI1A\B start and stop times
    for idx, cme in helcats_cmes.iterrows():
        # Get event label
        event_label = "ssw_{0:03d}_helcats_{1}".format(idx+1214, cme['event_id'])
        print(event_label)
        # Set correct craft value
        craft = cme.event_id[5]
        if craft == 'A':
            craft = 'sta'
        elif craft == 'B':
            craft = 'stb'

        # Make sure CME date is within data on hard drive
        if cme['start_time'].year > datetime.date(2007, 1, 1).year:
#            if cme['end_time'].year < datetime.date(2014, 1, 1).year:
            if cme['start_time'].year > datetime.date(2007, 1, 1).year:
                
                # Search hard drive for matching HI image
                hi_files = hip.find_hi_files(cme['start_time'],
                                             cme['end_time'],
                                             craft=craft,
                                             camera='hi1',
                                             background_type=1)
                if len(hi_files) > 1:
                    # Make images listed in img_type_list
                    for img_type in img_type_list:
                        # Loop over the hi_files, make each image type
                        files_c = hi_files[1:]  # current frame files
                        files_p = hi_files[0:-1]  # previous frame files

                        # Loop over images found and store times
                        ts = []
                        for i in range(len(files_c)):
                            time_string = files_c[i][-25:-10]
                            ts.append(pd.datetime.strptime(time_string, '%Y%m%d_%H%M%S'))
                        
                        # Find nearest HI frame to the time the CME reaches
                        # half-way through the HI1 FOV
                        idx = find_nearest(ts, cme['mid_time'])
                        print (ts[idx] - cme['mid_time'])
                        fc = files_c[idx]  # current frame
                        fp = files_p[idx]  # previous frame to fc (for differencing)
                        
                        # Make the differenced image
                        hi_map = hip.get_image_diff(fc, fp, align=True, smoothing=True)
                        # Make it grey
                        diff_normalise = mpl.colors.Normalize(vmin=-0.05, vmax=0.05)
                        out_img = mpl.cm.gray(diff_normalise(hi_map.data), bytes=True)
                        # Get the image and flip upside down as there is no "origin=lower" option with PIL
                        out_img = Image.fromarray(np.flipud(out_img))                      
                        
                        # Save the image
                        out_name = "_".join([event_label, craft, img_type, hi_map.date.strftime('%Y%m%d_%H%M%S')]) + '.jpg'
                        out_path = os.path.join(proj_dirs['out_data'], 'comp_assets', craft, out_name)
                        out_img = out_img.convert(mode='RGB')
                        out_img.save(out_path)
            else:
                print("data not found")
        else:
            print("data not found")



def find_comps(n, cycles='all', rounds=1, rn=0):
    """Finds pairwise comparisons to use in the manifest file.
    
    :param: n: number of objects to compare
    :param: cycles: number of cycles with n comparisons, must be between 1 and 
        ceil(n/2)
        
    :param: rounds: number of files to split total comparisons over
    :param: rn: current round number, adds an offset to the cycle numbers run
    
    :returns: lists containing indexes of each asset to compare
    """ 
    # calculate maximum values
    max_cycles = np.int(np.ceil(n/2))-1
    max_ccs = n*max_cycles
    max_comps = np.int((n/2)*(n-1))  
    
    # if no number of cycles chosen, set at maximum
    if cycles == 'all':
        cycles = max_cycles
    if cycles != np.int(cycles):
        raise ValueError("number of cycles must be an integer")
    if cycles < 1:
        raise ValueError("must be at least one cycle")
    if cycles > max_cycles:
        raise ValueError("number of cycles cannot be greater than ceil(n/2)-1")
    if (cycles * rounds) > max_cycles:
        raise ValueError("cycles*rounds must be less than %s" %(max_cycles))
    if rn > rounds:
        raise ValueError("round number cannot exceed number of rounds")
        
    # build nxn matrix
    matrix = np.zeros((n,n), dtype=int)

    spacing = np.int(np.floor(max_cycles/cycles))
    cycle_nos = range(1, np.int(np.ceil(n/2)), spacing)[0:cycles]
    # change dependant on round number
    for c in range(len(cycle_nos)):
        cycle_nos[c] = cycle_nos[c] + rn - 1

    # each s is a loop with n comparisons
    # starts at diagonal under 1, as the 0 diagonal is the origin
    for s in cycle_nos:
        print(s)
        
        # change 0s to 1s for comparisons in this loop
        for i in range(0, n):
            j = np.mod(s+i, n)
            # Check this hasn't been compared already...
            if matrix[j, i] == 0:
                # Do this comparison
                matrix[i, j] = 1

    print('cycles run: %s out of %s' %(cycles, max_cycles))
    print('comparisons generated: %s out of %s' %(np.sum(matrix), max_ccs))
    m = matrix_to_list(matrix)
    return m


def matrix_to_list(matrix):
    """
    Takes a matrix and returns a list of the rows/columns of the non-zero 
    values.
    """
    first = []
    second = []
    n = len(matrix)
    # loop over rows
    for i in range(0, n):
        # loop over values in row
        for j in range(0, n):
            if matrix[i, j] == 1:
                first.append(i)
                second.append(j)
    return first, second
        

def make_manifest_comp(cycles=16, m_files=30):
    """
    This function produces the manifest to serve the ssw assets. This has the format of a CSV file with:
    asset_name,file1,file2,...fileN.
    Asset names will be given the form of sswN_helcatsM_craft_type_t1_t3, where t1 and t3 correspond to the times of the
    first and third image in sets of three.
    
    This works by searching the 'out_data/comp_assets' folder for assets and
    creating a manifest file for these assets. 
    
    :param: m_files: number of manifest files to split comparisons into
    :return: Outputs a "manifest.csv" file in the event/craft/type directory of these images, or multiple files
    """
    proj_dirs = apt.project_info()
    
    # Get the images to compare
    sta_data_dir = os.path.join(proj_dirs['out_data'], 'comp_assets', 'sta')
    stb_data_dir = os.path.join(proj_dirs['out_data'], 'comp_assets', 'stb')
    if not os.path.exists(sta_data_dir):
        print("Error: sta_data_dir path does not exist.")
    if not os.path.exists(stb_data_dir):
        print("Error: stb_data_dir path does not exist.")       
    sta_files = glob.glob(os.path.join(sta_data_dir, '*.jpg'))
    sta_files2 = glob.glob(os.path.join(sta_data_dir, '*.png'))
    stb_files = glob.glob(os.path.join(stb_data_dir, '*.jpg'))
    stb_files2 = glob.glob(os.path.join(stb_data_dir, '*.png'))
    # get only filename not full path, exclude extension
    sta_files = [os.path.basename(f) for f in sta_files]
    sta_files2 = [os.path.basename(f) for f in sta_files2]
    stb_files = [os.path.basename(f) for f in stb_files]
    stb_files2 = [os.path.basename(f) for f in stb_files2]
    images1 = np.append(sta_files, stb_files)
    images1.sort()
    images2 = np.append(sta_files2, stb_files2)
    images2.sort()
    images = np.append(images1, images2)
    print("found %s images, generating comparisons..." %(len(images)))

    # Get place to save manifest files
    data_dir = os.path.join(proj_dirs['out_data'], 'comp_assets')
    if not os.path.exists(data_dir):
        print("Error: data_dir path does not exist.")
    
    # Create manifest files
    for r in range(m_files):
        # Make the manifest file
        manifest_path = os.path.join(data_dir, 'manifest'+str(r+1)+'.csv')
        with open(manifest_path, 'w') as manifest:
            # Add in manifest headers
            manifest.write("subject_id,asset_0,asset_1\n")

            # Get comparisons list for this manifest file
            comps = find_comps(len(images), cycles=cycles, rounds=m_files, rn=r+1)
            # returns lists of left and right images to compare

            # Write comparisons list into correct columns
            # loop over each comparison
            i = 0
            # give each comparison a subject id
            sub_id = 0
            while i < len(comps[0]):
                manifest_elements = [str(sub_id), images[comps[0][i]], images[comps[1][i]]]
                i = i + 1
                sub_id += 1
                # Write out as comma sep list
                manifest.write(",".join(manifest_elements) + "\n")


def make_manifest_comp_add_on(m_files=30):
    """
    This function produces the manifest to serve the ssw assets. This has the format of a CSV file with:
    asset_name,file1,file2,...fileN.
    Asset names will be given the form of sswN_helcatsM_craft_type_t1_t3, where t1 and t3 correspond to the times of the
    first and third image in sets of three.
    
    This works by searching the 'out_data/comp_assets' folder for assets and
    creating a manifest file for these assets. 
    
    :param: m_files: number of manifest files to split comparisons into
    :return: Outputs a "manifest.csv" file in the event/craft/type directory of these images, or multiple files
    """
    proj_dirs = apt.project_info()
    
    # Get the images to compare
    sta_data_dir = os.path.join(proj_dirs['out_data'], 'comp_assets', 'sta')
    stb_data_dir = os.path.join(proj_dirs['out_data'], 'comp_assets', 'stb')
    if not os.path.exists(sta_data_dir):
        print("Error: sta_data_dir path does not exist.")
    if not os.path.exists(stb_data_dir):
        print("Error: stb_data_dir path does not exist.")       
    sta_files = glob.glob(os.path.join(sta_data_dir, '*.jpg'))
    stb_files = glob.glob(os.path.join(stb_data_dir, '*.jpg'))
    # get only filename not full path, exclude extension
    sta_files = [os.path.basename(f) for f in sta_files]
    stb_files = [os.path.basename(f) for f in stb_files]
    images = np.append(sta_files, stb_files)
    images.sort()
    print("found %s images, generating comparisons..." %(len(images)))

    # Get place to save manifest files
    data_dir = os.path.join(proj_dirs['out_data'], 'comp_assets')
    if not os.path.exists(data_dir):
        print("Error: data_dir path does not exist.")
    
    # Create manifest files
    for r in range(m_files):
        # Make the manifest file
        manifest_path = os.path.join(data_dir, 'manifest'+str(r+1)+'.csv')
        with open(manifest_path, 'w') as manifest:
            # Add in manifest headers
            manifest.write("subject_id,asset_0,asset_1\n")

            # Get comparisons list for this manifest file
            comps = find_comps(len(images), cycles=18, rounds=m_files, rn=r+1)
            # returns lists of left and right images to compare

            # Write comparisons list into correct columns
            # loop over each comparison
            i = 0
            # give each comparison a subject id
            sub_id = 0
            while i < len(comps[0]):                          
                if np.int(images[comps[0][i]][len(images[comps[0][i]])-19:len(images[comps[0][i]])-15]) < 2014:
                    if np.int(images[comps[1][i]][len(images[comps[1][i]])-19:len(images[comps[1][i]])-15]) < 2014:
                        i = i + 1
                    else:
                        manifest_elements = [str(sub_id), images[comps[0][i]], images[comps[1][i]]]
                        i = i + 1
                        sub_id += 1
                        # Write out as comma sep list
                        manifest.write(",".join(manifest_elements) + "\n")
                        
                else:
                    manifest_elements = [str(sub_id), images[comps[0][i]], images[comps[1][i]]]
                    i = i + 1
                    sub_id += 1
                    # Write out as comma sep list
                    manifest.write(",".join(manifest_elements) + "\n")