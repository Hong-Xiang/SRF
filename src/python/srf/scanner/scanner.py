from typing import Dict
from dxl.learn.core.config import ConfigurableWithName


import numpy as np


class Scanner(ConfigurableWithName):
    """A Scanner is a general object.

    """

    def __init__(self, name, config: Dict):
        name = 'scanner_config'
        super.__init__(name, config)
        self._scanner = self._make_scanner()

    def _make_scanner(self):
        pass
