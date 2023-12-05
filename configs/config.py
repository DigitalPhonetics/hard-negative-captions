import json

class Config:
    def __init__(self, config_path) -> None:
        self.__conf = json.load(open(config_path))

    def set(self, name, value) -> None:
        if name in self.__conf['setters']:
            self.__conf[name] = value
        else:
            raise NameError("Property not listed in config setters. Please add the respective property, if ")
    
    def get(self, name):
        return self.__conf.get(name)
    
    def __str__(self):
        return str(self.__conf)
