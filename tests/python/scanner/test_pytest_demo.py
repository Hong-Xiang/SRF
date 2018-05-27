import unittest

class TestStandard(unittest.TestCase):
    def test_1(self):
        self.assertEqual(1, 1)
    
    def test_2(self):
        self.assertTrue(True)

def test_3():
    a = 1
    assert a == 1
