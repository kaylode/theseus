import yaml

class Config():
    def __init__(self, yaml_path):
        yaml_file = open(yaml_path)
        self._attr = yaml.load(yaml_file, Loader=yaml.FullLoader)['settings']

    def __setattr__(self, name, value):
        self.__dict__[name] = value

    def __getattr__(self, attr):
        try:
            return self._attr[attr]
        except KeyError:
            try:
                return self.__dict__[attr]
            except KeyError:
                return None

    def __str__(self):
        print("##########   CONFIGURATION INFO   ##########")
        pretty(self._attr)
        return '\n'
    
    def to_dict(self):
        out_dict = {}
        for k,v in self._attr.items():
            if v is not None:
                out_dict[k] = v
        return out_dict
    

def config_from_dict(_dict):
    config = Config('./configs/configs.yaml')
    for k,v in _dict.items():
        config[k] = v
    return config
        
def pretty(d, indent=0):
  for key, value in d.items():
    print('    ' * indent + str(key) + ':', end='')
    if isinstance(value, dict):
      print()
      pretty(value, indent+1)
    else:
      print('\t' * (indent+1) + str(value))