from dynaconf import Dynaconf
import argparse
from pathlib import Path

def parse_args():
    parser = argparse.ArgumentParser(description="Configuration loader")
    parser.add_argument("--config", type=str, default="go1.yaml", help="Configuration file name")
    return parser.parse_args()

def get_config_path(config_name):
    return Path.cwd() / config_name

def create_config(config_path):
    return Dynaconf(
        envvar_prefix="DYNACONF",
        settings_files=[config_path],
    )

def init_config():
    args = parse_args()
    config_path = get_config_path(args.config)
    return create_config(config_path)

cfg = init_config()
