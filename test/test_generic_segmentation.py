import unittest
from data.generic.segmentation import GenericSegmentation
from pathlib import Path


class TestGenericSegmentation(unittest.TestCase):

    def test(self):
        root = Path(__file__).parent.joinpath('datasets').joinpath('lip')
        dataset = GenericSegmentation(root)
