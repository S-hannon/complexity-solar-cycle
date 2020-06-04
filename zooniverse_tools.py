from __future__ import division
import os
import numpy as np
import pandas as pd
import simplejson as json
from misc import get_project_dirs


class Workflow:
    def __init__(self, workflow_name, project_name):
        self.name = workflow_name
        # project data comes saved with '-' instead of ' ' in the name
        self.project = ('-').join(project_name.split(' '))
        self.folder = get_project_dirs()['data']
    def get_classification_data(self):
        """Loads in the project-name-classifications and project-name-subjects
        files from the Zooniverse project, and converts to pd dfs.
        """
        converters = dict(classification_id=int,
                          user_name=str,
                          user_id=str,
                          user_ip=str,
                          workflow_id=int,
                          workflow_name=str,
                          workflow_version=float,
                          created_at=pd.to_datetime,
                          gold_standard=str,
                          expert=str,
                          metadata=json.loads,
                          annotations=json.loads,
                          subject_data=json.loads,
                          subject_ids=int)
        clas_path = os.path.join(self.folder,
                                 self.project + '-classifications.csv')
        classifications = pd.read_csv(clas_path, converters=converters)
        # drop classifications from other workflows         
        classifications = classifications[classifications.workflow_name == self.name]
        # make nice list of assets in new column
        assets = []
        for c in classifications.index:
            sub_id = str(classifications.subject_ids[c])
            thing = classifications.subject_data[c][sub_id]
            asset_list = [value for key, value in thing.items() if 'asset' in key]
            assets.append(asset_list)
        classifications['assets'] = pd.Series(assets,
                                              index=classifications.index)
        # drop stuff for simplicity
        cols = ['user_id', 'workflow_id', 'workflow_name', 'workflow_version',
                'gold_standard', 'expert', 'metadata', 'subject_data',
                'subject_ids']
        classifications = classifications.drop(columns=cols)
        return classifications
