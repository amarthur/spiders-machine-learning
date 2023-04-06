import argparse
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
    if train_params['weights_file_name'] == True:
        train_params['weights_file_name'] = get_config_file_name(sm_config, ext='.pt')

    # Model
    sm = SpeciesModel(**sm_config['init_parameters'])
    model = sm.load_pretrained_model(**pretr_params)

    # Model parameters
    model_params['criterion'] = criterion_class()
    model_params['optimizer'] = optimizer_class(model.parameters())
    model_params['scheduler'] = scheduler_class(model_params['optimizer'], **sched_config['params'])

    # Training
    sm.train_model(model=model, **model_params, **train_params)


def get_config_name(data, exclude_keys):
    is_dict = lambda v: isinstance(v, dict)
    name = ""

    if is_dict(data):
        for key, value in data.items():
            if value is None or key in exclude_keys:
                continue
            if is_dict(value):
                name += f"{get_config_name(value, exclude_keys)}"
            else:
                name += f"-{key[0]}_{str(value)}-"
    return name

def get_config_file_name(config_data, ext='.yaml', exclude_keys={"device", "weights_file_name"}):
    config_name = get_config_name(config_data, exclude_keys)
    config_file_name = config_name[1:-1] + ext
    return config_file_name

def get_config_file_path(config_file_name):
    dirs = DirectoryStructure()
    config_file_path = dirs.configs_dir / config_file_name
    file_is_valid = config_file_path.is_file() and config_file_path.suffix == ".yaml"
    return config_file_path if file_is_valid else None


def main():
    # Define parser flags
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--filename', type=str, help='Input config filename (Filename/Path)')
    parser.add_argument('-r', '--rename', action=argparse.BooleanOptionalAction, help='Rename config file (True/False)')

    # Add defaults
    defaults = {"filename": "default_config.yaml", "rename": False}
    parser.set_defaults(**defaults)

    # Parse
    args = parser.parse_args()
    config_file_name = Path(args.filename).name
    config_file_path = get_config_file_path(config_file_name)

    # Get config file
    if config_file_path is None:
        raise FileNotFoundError(f"File '{config_file_name}' not found.")

    with open(config_file_path) as config_file:
        config = yaml.safe_load(config_file.read())

    qk_config = config['quick_settings']
    db_config = config['database']
    ds_config = config['dataset']
    sm_config = config['model']

    # Rename
    if args.rename and config_file_name != defaults['filename']:
        new_config_file_name = get_config_file_name(sm_config)
        new_config_file_path = config_file_path.with_name(new_config_file_name)
        if new_config_file_path.exists():
            print("Skipping: File with same configurations already exists.")
        else:
            config_file_path.rename(new_config_file_path)

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
