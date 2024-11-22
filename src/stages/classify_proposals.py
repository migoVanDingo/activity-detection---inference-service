import os
from src.utility.load import load_yaml
from src.classes.classify_typing_proposals import ClassifyTypingProposalsFast
from dotenv import load_dotenv
load_dotenv()


# STAGE: EXTRACT PROPOSALS
if __name__ == "__main__":
    config = load_yaml(os.environ['PARAMS_PATH'])
    cp = ClassifyTypingProposalsFast(config)
    gp = cp.classify_proposals()


    