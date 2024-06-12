import os
import pandas as pd
from tqdm import tqdm
from dotenv import load_dotenv
load_dotenv()
from src.utility.load import load_csv
from src.utility.directory import check_directory_path
from src.utility.video import get_video_properties, save_spatiotemporal_trim
from src.utility.coordinates import get_intersection_coordinates

class ExtractRegionProposals:
    def __init__(self, config) -> None:
        self.config = config

    

    def extract_proposals(self) -> None:
        """Extracts typing region proposals using ROI to a directory (`odir`).

        It write the output to `tyrp_only_roi.csv` adding an extra
        column for names of extracted videos.

        It also creates a text file at the output location of proposals
        in the format our validation dataloader and mmaction2 validation
        dataloader expects called, `proposals_list_kbdet.txt`

        Parameters
        ----------
        overwrite : Bool, optional
            Overwrite existing typing proposals. Defaults to True.
        model_fps : int, optional
            The input FPS expected by the testing model. The proposals extracted from then
            video have to be sampled at this frame rate. Defaults to the FPS of the session
            video.
        """
        
 
        # Check if output directory exists
        if not check_directory_path(self.config['extract_proposal']['output_dir']):
            print(f"Creating output directory structure: {self.config['extract_proposal']['output_dir']}")
            os.makedirs(self.config['extract_proposal']['output_dir'])

        # Loop through each video
        prop_name_lst = []
        prop_rel_paths = []

        region_proposals = load_csv(f"{self.config['generate_proposal']['output_dir']}/{self.config['generate_proposal']['output_file']}")
        video_names = region_proposals['name'].unique().tolist()

        for i, video_name in enumerate(video_names):

            # Loading keyboard detection dataframe
            video_name_no_ext = os.path.splitext(video_name)[0]
            kb_det = pd.read_csv(f"{self.config['directory']['keyboard_detection']}/{video_name_no_ext}_60_det_per_min.csv")

            # Typing proposals for current dataframe
            print(f"Extracting typing region proposals from: {video_name}")

            # Region proposal per video
            tyrp_video = region_proposals[region_proposals['name'] == video_name].copy()

            
            video_props = get_video_properties(f"{self.config['directory']['video']}/{video_name}")
            # Loop through each instance in the video
            for ii, row in tqdm(tyrp_video.iterrows(), total=tyrp_video.shape[0], desc="Extracting: "):

                # Spatio temporal trim coordinates
                bbox = [row['w0'],row['h0'], row['w'], row['h']]
                sfrm = row['f0']
                efrm = row['f1']

                # Get keyboard detection intersection bounding box
                kb_bbox = self.get_detection_intersection(kb_det, sfrm, efrm)

                # Did they overlap?
                iflag, icoords = get_intersection_coordinates(bbox, kb_bbox)

                # Trimming video
                if iflag:
                    prop_name = f"{video_props['name']}_{row['pseudonym']}_{sfrm}_to_{efrm}.mp4"
                    prop_name_lst += [prop_name]
                    opth_rel = f"proposals_kbdet/{prop_name}"
                    prop_rel_paths += [opth_rel]
                    opth = f"{self.config['extract_proposal']['output_dir']}/{opth_rel}"

                    if not check_directory_path(f"{self.config['extract_proposal']['output_dir']}/proposals_kbdet"):
                        os.makedirs(f"{self.config['extract_proposal']['output_dir']}/proposals_kbdet")



                    # Check if the file already exists
                    if self.config['extract_proposal']["overwrite"]:
                        save_spatiotemporal_trim(video_props, sfrm, efrm, bbox, opth)
                    else:
                        if not os.path.isfile(opth):
                            save_spatiotemporal_trim(video_props, sfrm, efrm, bbox, opth)
                else:
                    prop_name_lst += ["dummy_name.mp4"]
                    opth_rel = f"proposals_kbdet/dummy_name.mp4"
                    prop_rel_paths += [opth_rel]
                            

        # Saving the proposal dataframe with new column
        if "proposal_name" in region_proposals.columns:
            region_proposals.drop("proposal_name", inplace=True, axis=1)
            region_proposals['proposal_name'] = prop_name_lst
        else:
            region_proposals['proposal_name'] = prop_name_lst

        tyrp_roi_only_loc = f"{self.config['generate_proposal']['output_dir']}/{self.config['generate_proposal']['output_file']}"

        print(f"Rewriting {tyrp_roi_only_loc}")
        region_proposals.to_csv(tyrp_roi_only_loc, index=False)

        # Saving the proposals list text files
        text_file_path = f"{self.config['extract_proposal']['output_dir']}/proposals_list_kbdet.txt"
        print(f"Writing {text_file_path}")

        f = open(text_file_path, "w")
        for prop_rel_path in prop_rel_paths:
            if prop_rel_path != "proposals_kbdet/dummy_name.mp4":
                f.write(f"{prop_rel_path} 100\n")
        f.close()


    def get_detection_intersection(self, kb_det, sfrm, efrm):
        """Determines keyboard detection intersectin bouhnding box between
        sfrm and efrm"""

        # Snipping keyboard detection dataframe between sfrm and efrm
        kdf = kb_det.copy()
        kdf = kdf[kdf['f0'] >= sfrm].copy()
        kdf = kdf[kdf['f0'] <= efrm].copy()

        # If we do not have any detection we will send [0, 0, 0, 0]
        if len(kdf) == 0:
            return [0, 0, 0, 0]
        
        kdf['w1'] = kdf['w0'] + kdf['w']
        kdf['h1'] = kdf['h0'] + kdf['h']

        # Top left intersection coordinates
        tl_w = max(kdf['w0'].tolist())
        tl_h = max(kdf['h0'].tolist())

        # Bottom right intersection coordinates
        br_w = min(kdf['w1'].tolist())
        br_h = min(kdf['h1'].tolist())
        w = br_w - tl_w
        h = br_h - tl_h

        return [tl_w, tl_h, w, h]
