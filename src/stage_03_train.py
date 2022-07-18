import pandas as pd
import argparse
from src.utils.common_utils import read_params,clean_prev_dirs_if_exists,create_dir,save_reports
import logging
from sklearn.model_selection import train_test_split
from sklearn.linear_model import ElasticNet
import joblib

logging_str = "[%(asctime)s: %(levelname)s: %(module)s]: %(message)s"
logging.basicConfig(level= logging.DEBUG, format=logging_str)

def train(config_path):
    config = read_params(config_path)
    
    artifacts = config["artifacts"]
    raw_local_data = artifacts['raw_local_data']
    split_data = artifacts['split_date']
    processed_data_dir = split_data['processed_data_dir']
    test_data_path = split_data['train_path']
    train_data_path = split_data['test_path']

    
    base = config['base']
    random_state = base['random_state']
    target = base['target_col']


    reports = artifacts["reports"]
    reports_dir = reports['reports_dir']
    params_file = reports['params']

    elasticnet_params = config['estimators']['ElasticNet']['params']
    alpha = elasticnet_params['alpha']
    l1_ratio = elasticnet_params['l1_ratio']

    train = pd.read_csv(train_data_path, sep=',')
    train_y = train[target]
    train_x = train.drop(target,axis = 1)
    
    lr = ElasticNet(alpha= alpha, l1_ratio= l1_ratio, random_state= random_state)

    lr.fit(train_x, train_y)
    model_dir = artifacts['model_dir']
    model_path = artifacts['model_path']
    create_dir([reports_dir, model_dir])

    params = {
        "alpha": alpha,
        "l1_ratio": l1_ratio
    }
    save_reports(params_file, params)

    joblib.dump(lr, model_path)

    logging.info(f"Model Saved at {model_path}")

if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--config", default="params.yaml")
    parsed_args = args.parse_args()

    try:
        data = train(config_path = parsed_args.config)
        logging.info("Train data stage is completed")
    except Exception as e:
        logging.error(e)
        raise e