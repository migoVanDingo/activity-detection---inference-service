import os
import time
import pandas as pd
from barbar import Bar
import torch
from torchsummary import summary
from src.classes.data_loader import AOLMEValTrmsDLoader
from src.classes.dyadic_cnn_3d import DyadicCnn3d
from src.utility.load import load_csv
from src.utility.memory_monitor import MemoryMonitor
from torch.utils.data import DataLoader
import nvidia_smi
#

class ClassifyTypingProposalsFast:

    def __init__(self, config) -> None:
        self.config = config
        self.enable_gpu()

    def enable_gpu(self):
      if self.config['neural_network']['gpu']:
        nvidia_smi.nvmlInit()

    def classify_proposals(self):
            """Classify each proposed region as typing / no-typing. This method needs the
            dataset in the format validation AOLMETrmsDLoader expects.

            Parameters
            ----------
            overwrite : Bool, optional
                Overwrites existing excel file 
            """

            # Output excel file with predictions
            if self.config['base']['keyboard_detection']:
                out_file = f"{self.config['classify_proposal']['output_dir']}/{self.config['classify_proposal']['output_file_detection']}"
            else:
                out_file = f"{self.config['classify_proposal']['output_dir']}/{self.config['classify_proposal']['output_file_no_detection']}"

            # Creating default columns for activity, class_idx and class_prob
            region_proposals = load_csv(f"{self.config['generate_proposal']['output_dir']}/{self.config['generate_proposal']['output_file']}")
            region_proposals["activity"] = "notyping"
            region_proposals["class_idx"] = 0
            region_proposals["class_prob"] = 0
            
            # If the file already exists and overwrite argument is false
            # we load the file as dataframe
            if not self.config["neural_network"]["overwrite"]:
                if os.path.isfile(out_file):
                    print(f"Reading {out_file}")
                    tydf = load_csv(out_file)
                    return tydf

            # creating tydf by copying self.tyrp_only_roi
            tydf = region_proposals.copy()
            
            # Loading list of videos from the proposals_list.txt file
            proposal_names = []
            if self.config['base']['keyboard_detection']:
                proposal_list_file =f"{self.config['extract_proposal']['output_dir']}/proposals_list_kbdet.txt"
            else:
                proposal_list_file =f"{self.config['extract_proposal']['output_dir']}/proposals_list.txt"
            with open(proposal_list_file) as f:
                lines = f.readlines()
                for line in lines:
                    proposal_rel_loc = line.split(" ")[0]
                    proposal_name = os.path.basename(proposal_rel_loc)
                    proposal_names += [proposal_name]
            
            # Loading the network
            net = self.load_net()


            # Initializing AOLME Validaiton data loader
            tst_data = AOLMEValTrmsDLoader(
                self.config['extract_proposal']['output_dir'], proposal_list_file, oshape=(224, 224)
            )
            tst_loader = DataLoader(
                tst_data,
                shuffle=False,
                batch_size=self.config['base']['batch_size'],
                num_workers=self.config['base']['num_workers']
            )

            # Resetting maximum memory usage and starting the clock
            if self.config['neural_network']['gpu']:
                torch.cuda.reset_peak_memory_stats(device=0)
            pred_prob_lst = []

            # Starting inference
            start_time = time.time()
            for idx, data in enumerate(Bar(tst_loader)):
                try:
                    if self.config['neural_network']['gpu']:
                        dummy_labels, inputs = (
                            data[0].to("cuda:0", non_blocking=True),
                            data[1].to("cuda:0", non_blocking=True)
                        )
                    else: 
                        dummy_labels, inputs = (
                            data[0],
                            data[1]
                        )
                except Exception as e:
                    print(f"Error processing batch {idx} with error: {e}")
                    raise

                with torch.no_grad():
                    outputs = net(inputs)
                    ipred = outputs.data.clone()
                    ipred = ipred.to("cpu").numpy().flatten().tolist()
                    pred_prob_lst += ipred
            # End of inference

            # Collecting and printing statistics
            end_time = time.time()
            if self.config['neural_network']['gpu']:      
                max_memory_MB = torch.cuda.max_memory_allocated(device=0)/1000000
            else:
                monitor = MemoryMonitor()
                monitor.update()
                max_memory_MB = monitor.get_max_memory_allocated()

            print(f"INFO: Total time for batch size of       {self.config['base']['batch_size']} = {round(end_time - start_time)} sec.")
            print(f"INFO: Max memory usage for batch size of {self.config['base']['batch_size']} = {round(max_memory_MB, 2)} MB")
            
            # Edit information in the data frame
            for i, proposal_name in enumerate(proposal_names):

                # Calculating class details
                pred = pred_prob_lst[i]
                pred_class_idx = round(pred)
                if pred_class_idx == 1:
                    pred_class = "typing"
                else:
                    pred_class = "notyping"
                pred_class_prob = round(pred, 2)

                # This is because for 0.5 I am having problems in ROC curve
                if pred_class_prob == 0.5:
                    if pred_class_idx == 1:
                        pred_class_prob = 0.51
                    else:
                        pred_class_prob = 0.49
                        
                # Adding the class details to to the dataframe
                loc = tydf[tydf['proposal_name']==proposal_name].index.tolist()
                if len(loc) > 1:
                    print(f"Multiple rows having same proposal name! {loc}")
                    import pdb; pdb.set_trace()

                tydf.loc[loc[0], "activity"] = pred_class
                tydf.loc[loc[0], "class_idx"] = pred_class_idx
                tydf.loc[loc[0], "class_prob"] = float(pred_class_prob)


            # Saving the csv file
            print(f"Saving {out_file}")
            tydf.to_csv(out_file, index=False) 


    def load_net(self):
        """ Load neural network to GPU or CPU. """
        print("INFO: Loading Trained network to GPU/CPU ...")

        # Determine the device
        device = torch.device("cuda:0" if self.config['neural_network']['gpu'] else "cpu")

        # Creating an instance of Dyadic 3D-CNN
        net = DyadicCnn3d(self.config['neural_network']['depth'], self.config['neural_network']['input_shape'].copy())
        net.to(device)  # Move model to the correct device

        # Print summary of the network
        summary(net, tuple(self.config['neural_network']['input_shape']), device="cuda" if self.config['neural_network']['gpu'] else "cpu")

        # Load the checkpoint weights
        ckpt_weights = torch.load(self.config['neural_network']['checkpoint'], map_location=device)
        net.load_state_dict(ckpt_weights['model_state_dict'])
        net.eval()  # Set model to evaluation mode

        return net