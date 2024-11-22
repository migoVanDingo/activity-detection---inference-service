import os
from src.utility.load import load_yaml
from src.classes.extract_region_proposals import ExtractRegionProposals
from dotenv import load_dotenv
load_dotenv()

# STAGE: EXTRACT PROPOSALS
if __name__ == "__main__":
    config = load_yaml(os.environ['PARAMS_PATH'])
    rp = ExtractRegionProposals(config)
    gp = rp.extract_proposals()


    