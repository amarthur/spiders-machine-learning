import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
import yaml
from database import Database
from dataset import Dataset
from directory_structure import DirectoryStructure
from species_model import SpeciesModel
from torch.optim import lr_scheduler


def create_database(db_config):
    db = Database(**db_config['init_parameters'])
    db.create_database(**db_config['creation_parameters'])


def create_dataset(ds_config):
    ds = Dataset(**ds_config['init_parameters'])
    ds.create_dataset(**ds_config['creation_parameters'])


def train_model(sm_config):
    # Configs
    pretr_params = sm_config['pretrained_parameters']
    train_params = sm_config['train_parameters']
    model_params = sm_config['model_parameters']
    sched_config = model_params['scheduler']

    # Configure
    criterion_class = getattr(nn, model_params['criterion'])
    optimizer_class = getattr(optim, model_params['optimizer'])
    scheduler_class = getattr(lr_scheduler, sched_config['sched_name'])

    # Model
    sm = SpeciesModel(**sm_config['init_parameters'])
    model = sm.load_pretrained_model(**pretr_params)

    # Model parameters
    model_params['criterion'] = criterion_class()
    model_params['optimizer'] = optimizer_class(model.parameters())
    model_params['scheduler'] = scheduler_class(model_params['optimizer'], **sched_config['params'])

    # Training
    sm.train_model(model=model, **model_params, **train_params)


def get_yaml_config_file():
    dirs = DirectoryStructure()
    yaml_default_name = "default_config.yaml"

    yaml_file = yaml_default_name if len(sys.argv) < 2 else sys.argv[1]
    yaml_file_path = dirs.configs_dir / Path(yaml_file).name
    valid_yaml_file = yaml_file_path.is_file() and yaml_file_path.suffix == ".yaml"

    return yaml_file_path if valid_yaml_file else None


def main():
    yaml_file = get_yaml_config_file()
    if yaml_file == None:
        print("Error: Invalid YAML config file.")
        return

    with open(yaml_file) as config_file:
        config = yaml.safe_load(config_file.read())

    qk_config = config['quick_settings']
    db_config = config['database']
    ds_config = config['dataset']
    sm_config = config['model']

    if qk_config['do_database']:
        create_database(db_config)

    if qk_config['do_dataset']:
        create_dataset(ds_config)

    if qk_config['do_model']:
        if not torch.cuda.is_available():
            raise RuntimeError("Cuda not available")
        train_model(sm_config)


if __name__ == "__main__":
    main()
