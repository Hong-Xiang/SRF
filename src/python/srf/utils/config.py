from dxl.core.config import ConfigProxy, CNode, Configuration

DEFAULT_CONFIGURATION_NAME = 'SRF_CONFIGURATION'


def config_with_name(name):
    if ConfigProxy().get(DEFAULT_CONFIGURATION_NAME) is None:
        ConfigProxy()[DEFAULT_CONFIGURATION_NAME] = Configuration(CNode({}))
    return ConfigProxy().get_view(DEFAULT_CONFIGURATION_NAME, name)


def clear_config():
    ConfigProxy.reset()
