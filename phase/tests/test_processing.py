from phase import image_process
from phase.tests.base_test import BaseTest
import cv2


class TestProcess(BaseTest):
    """
    Unit Tests for image processing
    """

    def TestConvolve(self, homo):
        image_process.convolveImage(homo)
    
    def TestThreshold(self, homo):
        gray = homo[:,:,0]
        image_process.apply_otsu(gray,dom_color=0)
    
    def TestBorder(self, homo):
        image_process.remove_border(homo)

    def get_dominant_color(self, homo):
        image_process.get_dominant_color(homo)
