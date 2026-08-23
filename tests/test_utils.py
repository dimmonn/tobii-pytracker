from pathlib import Path

def get_test_resources_dir() -> Path:
    return Path(__file__).parent / 'resources'


def get_test_config_path() -> Path:
    return get_test_resources_dir() / 'test_config.yaml'