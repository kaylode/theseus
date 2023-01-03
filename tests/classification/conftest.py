import pytest
from theseus.opt import Config

@pytest.fixture(scope="session")
def override_config():
    config = Config('./configs/classification/pipeline.yaml')
    config['global']['exp_name'] = 'pytest_clf'
    config['global']['exist_ok'] = True
    config['global']['save_dir'] = 'runs'
    config['global']['device'] = 'cpu'
    config['trainer']['args']['print_interval'] = 1
    config['trainer']['args']['save_interval'] = 2
    config['trainer']['args']['use_fp16'] = False
    config['trainer']['args']['num_iterations'] = 10
    config['data']['dataloader']['train']['args']['batch_size'] = 1
    config['data']['dataloader']['val']['args']['batch_size'] = 1
    return config

@pytest.fixture(scope="session")
def override_test_config():
    config = Config('./configs/classification/test.yaml')
    config['global']['exp_name'] = 'pytest_clf'
    config['global']['exist_ok'] = True
    config['global']['save_dir'] = 'runs'
    config['global']['device'] = 'cpu'
    config['data']['dataloader']['args']['batch_size'] = 1
    return config