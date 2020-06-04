from __future__ import division
import glob
import os
import sys
import numpy as np
import pandas as pd
import simplejson as json
#sys.path.insert(0, r'N:\\Documents\\Code\\useful_code')
sys.path.insert(0, r'C:\\Users\\shann\\OneDrive\\Documents\\Research\\Workspace\\Code\\useful_code')
from misc import get_project_dirs


def import_classifications(latest=True, version=None):
    """
    Function to load in the classifications and do some initial selection on them. Defaults to keeping only the
    classifications from the latest version of the workflow.
    :param latest: Bool, True or False depending on whether the classifications from the latest version of the workflow
                   should be returned.
    :param version: Float, version number of the workflow to extract the classifications. Defaults to None. Overrides
                     the "latest" input.
    :return data: A pandas dataframe containing the requested classifications.
    """
    if not isinstance(latest, bool):
        print("Error: latest should be type bool. Defaulting to True")
        latest = True

    if version is not None:
        # Version not none, so should be float.
        if not isinstance(version, float):
            print("Error, version ({}) should be None or a float. Defaulting to None.".format(version))
            version = None

    project_dirs = get_project_dirs()
    # Load in classifications:
    # Get converters for each column:
    converters = dict(classification_id=int, user_name=str, user_ip=str, workflow_id=int,
                      workflow_name=str, workflow_version=float, created_at=str, gold_standard=str, expert=str,
                      metadata=json.loads, annotations=json.loads, subject_data=json.loads, subject_ids=str)
    # Load the classifications into DataFrame and get latest subset if needed.
    data = pd.read_csv(project_dirs['classifications'], converters=converters)

    # Correct empty user_ids and convert to integers.
    data['user_id'].replace(to_replace=[np.NaN], value=-1, inplace=True)
    data['user_id'].astype(int)

    # Convert the subject ids, which are a variable length list of subject_ids. Because of setup of workflow
    # these numbers just repeat. Only need first.
    # Also, the json annotations are buried in a list, but only the dictionary in element 0 is needed. So pull this out
    # too
    subject_ids = []
    annos = []
    for idx, row in data.iterrows():
        subject_ids.append(row['subject_data'].keys()[0])
        annos.append(row['annotations'][0])

    # Set the new columns
    data['subject_id'] = pd.Series(subject_ids, dtype=int, index=data.index)
    data['annotations'] = pd.Series(annos, dtype=object, index=data.index)

    # Drop the old subject_ids, as not needed
    data.drop('subject_ids', axis=1, inplace=True)

    if version is None:
        if latest:
            data = data[data['workflow_version'] == data['workflow_version'].tail(1).values[0]]
    else:
        # Check the version exists.
        all_versions = set(np.unique(data['workflow_version']))
        if version in all_versions:
            data = data[data['workflow_version'] == version]
        else:
            print("Error: requested version ({}) doesn't exist. Returning all classifications".format(version))

    # Correct the index in case any have been removed
    data.set_index(np.arange(data.shape[0]), inplace=True)
    return data


def import_subjects(active=False):
    """
    Function to load in the classifications and do some initial selection on
    them. Defaults to retrieving all subjects, but can optionally retrieve only
    subjects associated with workflows.
    :param active: Bool, True or False depending on whether the only subjects
    assigned to a workflow should be selected.
    :return data: A pandas DataFrame containing the requested classifications.
    """
    if not isinstance(active, bool):
        print("Error: active should be type bool. Defaulting to False")
        active = False

    project_dirs = get_project_dirs()
    # Load in classifications:
    # Get converters for each column:
    converters = dict(subject_id=int, project_id=int, workflow_ids=json.loads, subject_set_id=int, metadata=json.loads,
                      locations=str, classifications_by_workflow=str, retired_in_workflow=str)
    # Load the classifications into DataFrame and get latest subset if needed.
    data = pd.read_csv(project_dirs['subjects'], converters=converters)
    # Sort out the workflow ids which are in an awkward format

    if active:
        ids = np.zeros(len(data), dtype=bool)
        for idx, d in data.iterrows():
            if not d['workflow_id'] == []:
                ids[idx] = 1
        data = data[ids]

    # Correct the index if subset selected
    data.set_index(np.arange(data.shape[0]), inplace=True)
    return data


def get_df_subset(data, col_name, values, get_range=False):
    """
    Function to return a copy of a subset of a data frame:
    :param data: pd.DataFrame, Data from which to take a subset
    :param col_name: Column name to index the DataFrame
    :param values: A list of values to compare the column given by col_name against. If get_range=True, values should be
                   a list of length 2, giving the lower and upper limits of the range to select between. Else, values
                   should be a list of items, where data['col_name'].isin(values) method is used to select data subset.
    :param get_range: Bool, If true data is returned between the two limits provided by items 0 and 1 of values. If
                      False data returned is any row where col_name matches an element in values.
    :return: df_subset: A copy of the subset of values in the DataFrame
    """
    if not isinstance(data, pd.DataFrame):
        print("Error: data should be a DataFrame")

    if not (col_name in data.columns):
        print("Error: Requested col_name not in data columns")

    # Check the values were a list, if not convert
    if not isinstance(values, list):
        print("Error: Values should be a list. Converting to list")
        values = [values]

    if get_range:
        # Range of values requested. In this case values should be a list of length two. Check
        if len(values) != 2:
            print("Error: Range set True, so values should have two items. ")

    # TODO: Add in a check for the dtype of value.

    if get_range:
        # Get the subset between the range limits
        data_subset = data[(data[col_name] >= values[0]) & (data[col_name] <= values[1])].copy()
    else:
        # Get subset from list of values
        data_subset = data[data[col_name].isin(values)].copy()

    return data_subset


#def match_classifications_to_subjects(active=True, latest=True):
#    """
#    A function to process the Solar Stormwatch subjects to create a HDF5 data
#    structure linking each classification with it's corresponding frame.
#    :param do_kde: Bool, If true finds consensus storm front coords
#    :param active: Bool, If true only processes subjects currently assigned to
#        a workflow.
#    :param latest: Bool, If true only processes classifications from the latest
#        version of the workflow
#    :return: Saves a csv file to 'out_data' containing the following columns:
#        index: asset number from 0 to n
#        subject_id: Zooniverse number of asset
#        left_subject: name of subject on left
#        right_subject: name of subject on right
#        left_wins: number of votes saying left subject more complex
#        right_wins: number of votes saying right subject more complex
#    """
#    project_dirs = get_project_dirs()
#
#    # Get classifications from latest workflow
#    all_classifications = import_classifications(latest=latest)
#    # Only import currently active subjects
#    all_subjects = import_subjects(active=active)
#    
#    # Initialise output lists
#    sub_ids = []
#    left_sub = []
#    right_sub = []
#    left_n = []
#    right_n = []
#    
#    # Loop over the subjects
#    for s in range(len(all_subjects)):
#        print "subject %s out of %s" %(s, len(all_subjects))
#        # Get subject details
#        sid = all_subjects.subject_id[s]
#        sub_ids.append(sid)
#        left_sub.append(all_subjects.metadata[s]['asset_0'])
#        right_sub.append(all_subjects.metadata[s]['asset_1'])
#        
#        # Initialse counters
#        left = 0
#        right = 0
#    
#        # Rows to drop
#        used_rows = []
#        
#        # Loop over classifications and find for this subject
#        for c in range(len(all_classifications)):
##            print "classification %s out of %s" %(c, len(all_classifications))
#            if all_classifications.subject_id[c] == sid:
#                result = all_classifications.annotations[c]['value']
#                # Add result to left or right score
#                if result == "Image on the left":
#                    left = left + 1
#                elif result == "Image on the right":
#                    right = right + 1
#                # This classification has been appended, drop from df
#                used_rows.append(c)
#        
#        # Tidy up df; remove used rows and reindex
#        all_classifications = all_classifications.drop(used_rows)
#        all_classifications.index = pd.RangeIndex(len(all_classifications.index))
#        print '%s classifications left to sort' %(len(all_classifications))
#
#        # Append total scores for this subject
#        left_n.append(left)
#        right_n.append(right)
#    
#    # Convert to pandas df
#    df = pd.DataFrame({'subject_id' : sub_ids,
#                       'left_subject' : left_sub,
#                       'right_subject' : right_sub,
#                       'left_wins' : left_n,
#                       'right_wins' : right_n})
#    
#    name = os.path.join(project_dirs['out_data'], 'popfss_comparison_results.csv')
#    df.to_csv(name, sep=',')
#    return 


def match_classifications_to_subjects(suffix="", active=True, latest=True):
    """
    A function to process the Solar Stormwatch subjects to create a HDF5 data
    structure linking each classification with it's corresponding frame.
    :param do_kde: Bool, If true finds consensus storm front coords
    :param active: Bool, If true only processes subjects currently assigned to
        a workflow.
    :param latest: Bool, If true only processes classifications from the latest
        version of the workflow
    :return: Saves a csv file to 'out_data' containing the following columns:
        index: asset number from 0 to n
        subject_id: Zooniverse number of asset
        left_subject: name of subject on left
        right_subject: name of subject on right
        left_wins: number of votes saying left subject more complex
        right_wins: number of votes saying right subject more complex
    """
    project_dirs = get_project_dirs()

    # Get classifications from latest workflow
    all_classifications = import_classifications(latest=latest)
    # Only import currently active subjects
    all_subjects = import_subjects(active=active)
    # Sort into ascending order and reindex
    all_classifications = all_classifications.sort_values(by='subject_id')
    all_subjects = all_subjects.sort_values(by='subject_id')
    all_classifications.index = pd.RangeIndex(len(all_classifications.index))
    all_subjects.index = pd.RangeIndex(len(all_subjects.index))
   
    # Initialise output lists
    sub_ids = []
    left_sub = []
    right_sub = []
    left_n = []
    right_n = []
    
    # Initialise classification counter
    c = 0
    
    # Loop over the subjects
    for s in range(len(all_subjects)):
        print("subject %s out of %s" %(s+1, len(all_subjects)))
        # Get subject details
        sid = all_subjects.subject_id[s]
        
        # Initialse counters
        left = 0
        right = 0

        while (c < len(all_classifications)) and (all_classifications.subject_id[c] <= sid):
            if all_classifications.subject_id[c] == sid:
                result = all_classifications.annotations[c]['value']
                # Add result to left or right score
                if result == "Image on the left":
                    left = left + 1
                elif result == "Image on the right":
                    right = right + 1
            # This classification has been appended, move on
            # Or this classification has not been matched to a subject... oops
            c = c + 1

        # Append total scores for this subject
        # if left == right:
        #     if left != 0:
        #         sub_ids.append(sid)
        #         left_sub.append(all_subjects.metadata[s]['asset_0'])
        #         right_sub.append(all_subjects.metadata[s]['asset_1'])
        #         left_n.append(left)
        #         right_n.append(right)
        else:
            sub_ids.append(sid)
            this_left_sub = all_subjects.metadata[s]['asset_0']
            this_right_sub = all_subjects.metadata[s]['asset_1']
            # remove any trailing spaces
            this_left_sub = this_left_sub.replace(' ', '')
            this_right_sub = this_right_sub.replace(' ', '')
            left_sub.append(this_left_sub)
            right_sub.append(this_right_sub)
            left_n.append(left)
            right_n.append(right)
    
    # Convert to pandas df
    df = pd.DataFrame({'subject_id' : sub_ids,
                       'left_subject' : left_sub,
                       'right_subject' : right_sub,
                       'left_wins' : left_n,
                       'right_wins' : right_n})
    
    name = os.path.join(project_dirs['out_data'], 'popfss_comparison_results' + suffix + '.csv')
    df.to_csv(name, sep=',')
    split_scores_by_spacecraft(df, suffix)
    return df


def remove_images_only_one_side(df):
    to_drop = []
    left_unique = np.unique(df.left_subject.values)
    right_unique = np.unique(df.right_subject.values)
    merged = np.append(left_unique, right_unique)
    unique, counts = np.unique(merged, return_counts=True)
    for j in range(len(counts)):
        if counts[j] == 1:
            to_drop.append(unique[j])
    id_drop = []
    for k in range(len(to_drop)):
        for l in range(len(df)):
            if df.left_subject.values[l] == to_drop[k]:
                id_drop.append(df.index.values[l])
            elif df.right_subject.values[l] == to_drop[k]:
                id_drop.append(df.index.values[l])
    df = df.drop(id_drop)
    return df


def split_scores_by_spacecraft(df, suffix=""):
    project_dirs = get_project_dirs()
    a_rows = []
    b_rows = []
    for i in range(len(df)):
        craft_l = df.left_subject[i][len(df.left_subject[i])-28:len(df.left_subject[i])-25]
        craft_r = df.right_subject[i][len(df.right_subject[i])-28:len(df.right_subject[i])-25]
        if craft_l == 'sta':
            if craft_r == 'sta':
                a_rows.append(i)
        if craft_l == 'stb':
            if craft_r == 'stb':
                b_rows.append(i)
    dfa = df.loc[a_rows]
    dfb = df.loc[b_rows]
    # Check all values exist in left and right cols, remove if not
    dfa = remove_images_only_one_side(dfa)
    dfb = remove_images_only_one_side(dfb)
    # Save
    name_a = os.path.join(project_dirs['out_data'], 'popfss_comparison_results_sta' + suffix + '.csv')
    name_b = os.path.join(project_dirs['out_data'], 'popfss_comparison_results_stb' + suffix + '.csv')
    dfa.to_csv(name_a, sep=',')
    dfb.to_csv(name_b, sep=',')
    return    
    
