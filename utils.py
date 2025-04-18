import pathlib, yaml

cfg = yaml.safe_load(pathlib.Path("config.yaml").open())

def get(key, default=None):
    return cfg.get(key, default)

