import os
import sys
import unittest

def run_tests():
    test_loader = unittest.TestLoader()
    test_suite = test_loader.discover('tests', pattern='test_*.py')
    test_runner = unittest.TextTestRunner()
    result = test_runner.run(test_suite)
    return result.wasSuccessful()

if __name__ == "__main__":
    if run_tests():
        print("All tests passed!")
    else:
        print("Some tests failed.")
        sys.exit(1)

