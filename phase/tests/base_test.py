import pytest
import cv2
import os
import numpy as np
from pkg_resources import resource_filename


TESTFILE_DIR = resource_filename('phase', 'tests/test_images')


class BaseTest:
    @pytest.fixture(autouse=True)
    def initdir(self, tmpdir):
        tmpdir.chdir()


    @pytest.fixture
    def homo(self):
        image = cv2.imread(os.path.join(TESTFILE_DIR, 'homo.png'))
        
        return image


    @pytest.fixture
    def hetero(self):
        image = cv2.imread(os.path.join(TESTFILE_DIR, 'hetero.png'))
        
        return image

    @pytest.fixture
    def kernels(self):
        kernel_size = 10
        kernel = np.ones((kernel_size, kernel_size)) / kernel_size
        kernel[0,:]

        return kernel
