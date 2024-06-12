import math
import numpy as np
import pandas as pd
from dotenv import load_dotenv
load_dotenv()
from src.utility.load import load_csv
from src.utility.video import get_video_properties
from src.utility.file import check_file_path
from src.utility.coordinates import get_intersection_coordinates


class GenerateRegionProposals:
    def __init__(self, config) -> None:
        self.config = config
        self.dur = config['base']['duration']
        self.fps = config['base']['fps']

    def generate_proposals(self) -> None:
        """ Calculates typing region proposals using ROI and keyboard
        detections.

        It write the output to `tyrp_only_roi.csv`

        Parameters
        ----------
        dur : int, optional
            Duraion of each typing proposal
        fps : Frames per second, optional
            Framerate of 
        """

        # Check if file already exists
        if not self.config['generate_proposal']['overwrite']:
            is_proposals_loaded = self.load_region_proposals(f"{self.config['generate_proposal']['output_dir']}/{self.config['generate_proposal']['output_file']}")

            if is_proposals_loaded:
                print(f"Reading loaded region proposals from: {self.config['generate_proposal']['output_dir']}/{self.config['generate_proposal']['output_file']}")
                return True

        print(f"Creating region proposals: {self.config['generate_proposal']['output_dir']}/{self.config['generate_proposal']['output_file']}")

        # Load session properties dataframe
        session_props = load_csv(self.config['files']['session_props'])

        # Load ROI dataframe
        roi = load_csv(self.config['files']['roi'])

        
        # Loop over each video every 3 secons
        typrop_lst = []
        for vid_name in self.config['files']['videos']:

            
            # Properties of current Video
            T = int(session_props[session_props['name'] == vid_name]['dur'].item())
            W = int(session_props[session_props['name'] == vid_name]['width'].item())
            H = int(session_props[session_props['name'] == vid_name]['height'].item())
            FPS = int(session_props[session_props['name'] == vid_name]['FPS'].item())



            # ROI and Keyboard detection for current video
            roi_df_vid = roi[roi['video_names'] == vid_name].copy()

            if check_file_path(f"{self.config['directory']['video']}/{vid_name}"):
                video_props = get_video_properties(f"{self.config['directory']['video']}/{vid_name}")
                f0_last = video_props['num_frames'] - self.dur*self.fps
            else:
                print(f"Video not found: {self.config['directory']['video']}/{vid_name}")
                raise FileNotFoundError()
                


            # Loop over each 3 second instance
            for i, f0 in enumerate(range(0, f0_last, self.dur*self.fps)):
                f = self.dur*self.fps
                f1 = f0 + f - 1

                # 3 second dataframe instances
                roi_interval = roi_df_vid[roi_df_vid['f0'] >= f0].copy()
                roi_interval = roi_interval[roi_interval['f0'] <= f1].copy()


                # Check if skipping
                if not self.check_interval_region(roi_interval):
                    
                    # Get 3 second proposal regions
                    proposal_regions = self.get_region_proposal(roi_interval.copy())

                    # Adding to prop_lst
                    for region_interval in proposal_regions:
                        typrop_lst_temp = [vid_name, W, H, FPS, T, f0, f, f1] + region_interval
                        typrop_lst += [typrop_lst_temp]

        # Creating typing proposal dataframe
        region_proposals = pd.DataFrame(
            typrop_lst,
            columns=['name', 'W', 'H', 'FPS', 'T', 'f0', 'f', 'f1', 'pseudonym', 'w0', 'h0', 'w', 'h']
        )
        region_proposals.to_csv(f"{self.config['generate_proposal']['output_dir']}/{self.config['generate_proposal']['output_file']}", index=False)

          
    def load_region_proposals(self, path: str):
        """
        FUNCTION RENAMED FROM: _read_from_disk()
        ORIGINAL FILE: framework3_roi_kb.py

        Load region proposals from the given path
        
        Returns pandas dataframe

        Parameters
        ----------
        path : str
            Path to the region proposals file
        """
        proposals = load_csv(path)
        if proposals is None:
            return False, proposals
            
        return True, proposals

    def check_interval_region(self, region: pd.DataFrame) -> bool:
        """
        FUNCTION RENAMED FROM: _skip_this_3sec_roi_only()
        ORIGINAL FILE: framework3_roi_kb.py

        Skip the current 3 second interval if
        1. We do not have table region of interest
        2. ROI should be available for more than half of the duration.

        Parameters
        ----------
        region : Pandas DataFrame instance
            ROI dataframe

        Returns
        -------
        bool
            True  = skip the current 3 seconds
            False = Do not skip the current 3 seconds.
        """
        region_temp = region.copy()

        # Return True if there are no ROI regions
        region_temp = region_temp.drop(['Time', 'f0', 'f', 'video_names'], axis=1)
        if region_temp.isnull().values.all():
            return True

        # There should be atleast one Table ROI  for 2 seconds or more.
        # If not we will skip analyzing the current 3 seconds
        for col_name in region_temp.columns.tolist():
            cur_col = [str(x) for x in region_temp[col_name].tolist()]
            num_nan = np.sum([1*(x == 'nan') for x in cur_col])
            if num_nan > math.floor(self.dur/2):
                continue
            else:
                return False

        return True
    
    def get_region_proposal(self, region: pd.DataFrame):
        """
        FUNCTION RENAMED FROM : _get_3sec_proposal_df_roi_only()
        ORIGINAL FILE: framework3_roi_kb.py

        Returns a dataframe with typing region proposals using
        1. Table ROI

        Parameters
        ----------
        region : Pandas Dataframe
            Table ROI for 3 seconds
        """

        # ROI column names (persons sitting around the table)
        region_cp = region.copy()
        region_cp = region_cp.drop(['Time', 'f0', 'f', 'video_names'], axis=1)
        persons_list = region_cp.columns.tolist()
        
        # Loop over each person ROI and check for keyboard detection
        prop_lst = []
        for person in persons_list:
            
            # Process further only if ROIs are available for more than
            # 1/2 the duration of the time
            roi_lst = [str(x) for x in region[person].tolist()]
            num_roi = len(roi_lst) - np.sum([1*(x == 'nan') for x in roi_lst])
            
            if num_roi > math.floor(self.dur/2):

                # Get ROIs intersection bounding box
                roi_coords = self.get_region_intersection(region, person)
                prop_lst += [
                    [person, roi_coords[0], roi_coords[1], roi_coords[2], roi_coords[3]]
                ]
                
        return prop_lst
    
    def get_region_intersection(self, region_df: pd.DataFrame, person: str):
        """
        FUNCTION RENAMED FROM: _get_roi_intersection
        ORIGINAL FILE: framework3_roi_kb.py

        Returns intersection region coordinates

        Parameters
        ----------
        region_df : pandas DataFrame
            region dataframe
        person : Str
            The name of the person currently under consideration
        """
        regions = [str(x) for x in region_df[person].tolist()]

        region_intersection = []
        for region in regions:
            
            if not region == 'nan':
                region = [int(x) for x in region.split('-')]
                
                if len(region_intersection) < 1:
                    region_intersection = region

                else:
                    # Get intersection coordinates of two boxes
                    overlap_flag, region_intersection = get_intersection_coordinates(region_intersection, region)
                    
                    # If not intersecting update to current coordinates
                    if not overlap_flag:
                        region_intersection = region
                    else:
                        region_intersection = region_intersection
                        
        return region_intersection
    
    
    