from dxl.learn.test import TestCase as TestCaseBase
import os
from pathlib import Path


class TestCase(TestCaseBase):
    @property
    def resource_path(self):
        return Path(os.getenv('HOME')) / 'UnitTestResource' / 'SRF'
