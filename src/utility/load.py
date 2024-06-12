import json
import os
import yaml
import pandas as pd

def load_json(config_path: str):
    try:
        print(f"Loading config data from: {config_path}")

        with open(config_path, "r") as file:
            config = json.load(file)
        return config
    except FileNotFoundError:
        print(f"ERROR: --------------> File not found: {config_path}")

def load_yaml(config_path: str):
    try:
        print(f"Loading config data from: {config_path}")
        with open(config_path, "r") as file:
            config = yaml.safe_load(file)
        return config
    except FileNotFoundError:
        print(f"ERROR: --------------> File not found: {config_path}")


def load_csv(path: str):
        try:
            """Loads CSV returns pandas dataframe if exists else returns None"""
            print(f"Loading CSV from: {path}")
            if os.path.isfile(path):
                return pd.read_csv(path)
                
            return None
        except FileNotFoundError:
            print(f"ERROR: --------------> File not found: {path}")
        


    