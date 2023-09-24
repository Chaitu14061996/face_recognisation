import yaml

data={
    'base' :{
        'data_path':'Data_base',
        'BASE_MODEL':'resnet50',
        'include_top': 'False',
        'pooling':'avg'
    }
}

with open('params.yaml',mode='w') as config_file:
    yaml.dump(data,config_file,indent=2)