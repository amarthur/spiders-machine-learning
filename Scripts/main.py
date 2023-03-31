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
    sched_config = sm_config['lr_scheduler']
    criterion_class = getattr(nn, sm_config['criterion'])
    optimizer_class = getattr(optim, sm_config['optimizer'])
    scheduler_class = getattr(lr_scheduler, sched_config['sched_name'])

    # Model
    sm = SpeciesModel(**sm_config['init_parameters'])
    model = sm.load_pretrained_model(**sm_config['pretrained_parameters'])

    # Other parameters
    criterion = criterion_class()
    optimizer_ft = optimizer_class(model.parameters())
    lr_sched = scheduler_class(optimizer_ft, **sched_config['params'])

    # Training
    sm.train_model(model=model,
                   criterion=criterion,
                   optimizer=optimizer_ft,
                   scheduler=lr_sched,
                   **sm_config['train_parameters'])


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

    db_config = config['database']
    ds_config = config['dataset']
    sm_config = config['model']

    if db_config['do'] == True:
        create_database(db_config)

    if ds_config['do'] == True:
        create_dataset(ds_config)

    if sm_config['do'] == True:
        if torch.cuda.is_available():
            train_model(sm_config)
        else:
            print("Cuda not available")


if __name__ == "__main__":
    main()
