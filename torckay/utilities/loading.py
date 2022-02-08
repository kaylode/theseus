import yaml

def load_yaml(path):
    with open(path, 'rt') as f:
        return yaml.safe_load(f)