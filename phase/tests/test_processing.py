from phase import image_process
from phase.tests.base_test import BaseTest
import cv2


class TestProcess(BaseTest):
    """
    Unit Tests for image processing
    """

    def test_convolve(self, homo, kernels):
        image_process.convolveImage(homo, kernels)
    
    def test_threshold(self, homo):
        gray = homo[:,:,0]
        image_process.apply_otsu(gray,dom_color=0)
    
    def test_border(self, homo):
        image_process.remove_border(homo)

    def test_dominant_color(self, homo):
        image_process.get_dominant_color(homo, 1)
