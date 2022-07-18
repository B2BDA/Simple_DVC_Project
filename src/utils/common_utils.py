import yaml
import os
import shutil
import logging
import json

def read_params(config_path: str) -> dict:
    with open(config_path) as yaml_file:
        config = yaml.safe_load(yaml_file)
    logging.info(f"read the parameters")
    return config

def clean_prev_dirs_if_exists(dir_path: str):
    if os.path.isdir(dir_path):
        shutil.rmtree(dir_path)
    logging.info(f"cleanign exisiting artifacts directory at {dir_path}")


def create_dir(dirs: list):
    for dir_path in dirs:
        os.makedirs(dir_path,exist_ok= True)
        logging.info(f"Created directory at: {dir_path}")

def save_local_df(df, df_path, header = False):
    if header == True:
        new_cols = [col.replace(" ","_") for col in df.columns]
        df.to_csv(df_path, index = False, header = new_cols)
        logging.info(f"Data saved at with new header: {df_path}")
        
    else:
        df.to_csv(df_path, index = False)
        logging.info(f"Data saved at: {df_path}")

def save_reports(filepath: str, report: dict):
    with open(filepath, "w") as f:
        json.dump(report, f, indent=4)
    f.close()
    logging.info(f"Report(s) saved at : {filepath}")




