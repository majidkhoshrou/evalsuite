import unittest
import evalsuite

class BasicTests(unittest.TestCase):
    def test_import(self):
        self.assertIsNotNone(evalsuite)

if __name__ == '__main__':
    unittest.main()
