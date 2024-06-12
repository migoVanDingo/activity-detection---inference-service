import os
from src.utility.load import load_yaml
from src.classes.generate_region_proposals import GenerateRegionProposals

# STAGE: GENERATE REGION PROPOSALS
if __name__ == "__main__":
    print(f"Generating region proposals")
    config = load_yaml(os.environ['PARAMS_PATH'])
    print(f"config: {config}")
    rp = GenerateRegionProposals(config)
    gp = rp.generate_proposals()


    