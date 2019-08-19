import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import scale
from scipy import misc
from torch.utils import data
import os
import copy

def adjust_gamma(image, gamma=1.0):
    # build a lookup table mapping the pixel values [0, 255] to
    # their adjusted gamma values
    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) * 255
                      for i in np.arange(0, 256)]).astype("uint8")

    # apply gamma correction using the lookup table
    return cv2.LUT(image, table)

def de_adjust_gamma(image, invGamma):
    table = np.array([((i / 255.0) ** invGamma) * 255
                      for i in np.arange(0, 256)]).astype("uint8")

    # apply gamma correction using the lookup table
    return cv2.LUT(image, table)

def add_noise(img, scale):

    img = img / 255.0

    m = (0, 0, 0)
    s = (scale[0], scale[1], scale[2])
    noise = np.zeros_like(img)
    noise = cv2.randn(noise, m, s) / 255.0

    img = np.clip((img + noise)*255 - np.mean(noise), 0, 255).astype(np.uint8)

    return img

def homography_alignment(target, im, number_of_iterations = 10):

    # Convert images to grayscale
    im1_gray = cv2.cvtColor(target, cv2.COLOR_RGB2GRAY)
    im2_gray = cv2.cvtColor(target, cv2.COLOR_RGB2GRAY)

    # Find size of image1
    sz = im.shape

    # Define the motion model
    warp_mode = cv2.MOTION_HOMOGRAPHY  # cv2.MOTION_TRANSLATION

    # Define 3x3 matrices and initialize the matrix to identity
    warp_matrix = np.eye(3, 3, dtype=np.float32)

    # Specify the threshold of the increment
    # in the correlation coefficient between two iterations
    termination_eps = 1e-10

    # Define termination criteria
    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, number_of_iterations, termination_eps)

    # Run the ECC algorithm. The results are stored in warp_matrix.
    (cc, warp_matrix) = cv2.findTransformECC(im1_gray, im2_gray, warp_matrix, warp_mode, criteria, None, 1)

    # Use warpPerspective for Homography
    im2_aligned = cv2.warpPerspective(im, warp_matrix, (sz[1], sz[0]), flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP)

    return im2_aligned

def motion_blur(damping=0.01, size=160):
    # generating the kernel
    kernel_motion_blur = np.zeros((size, size))
    x0 = int((size - 1) / 2)
    y0 = int((size - 1) / 2)
    dx = 0
    dy = 0

    x = x0
    y = y0
    dist = 0
    while dist < damping:
        ddx = np.random.normal(0, 1, 1)
        ddy = np.random.normal(0, 1, 1)

        dx += ddx
        dy += ddy
        x += (dx + 0.5 * ddx)
        y +=  (dy + 0.5 * ddy)
        if x >= 0 and y >= 0 and x < size and y < size:

            kernel_motion_blur[int(x), int(y)] = 1
            dist = np.linalg.norm(np.asarray([x0,y0]) - np.asarray([x,y]), 2)

    kernel_motion_blur = kernel_motion_blur / np.sum(kernel_motion_blur)  # normalizing

    return kernel_motion_blur

def make_burst_seq(img, length, kernel_lvl = 1, noise_lvl = 1, motion_blur_boolean = True, homo_align = True):
    burst = []

    for i in range(length):
        # add noise
        noise_lvl = np.random.uniform(2*noise_lvl, 5 * noise_lvl, 3)
        im_noisy = add_noise(copy.deepcopy(img), noise_lvl)

        # pick random gamma_coef between 1.5 and 2.5
        gamma_coef = np.random.uniform(1.5, 2.5)
        im_lin = de_adjust_gamma(im_noisy, gamma_coef)

        # apply motion blur
        if motion_blur_boolean:
            kernel = motion_blur(damping=kernel_lvl)
            output = cv2.filter2D(im_lin, -1, kernel)
        else:
            sigma = np.random.randint(3, 10*kernel_lvl)
            kernel_size = np.random.randint(4, 10)
            kernel = cv2.getGaussianKernel(ksize = kernel_size, sigma = sigma)
            output = cv2.filter2D(im_lin, -1, kernel)
            output = cv2.medianBlur(output, 5)

        # homography alignment
        if homo_align:
            output = homography_alignment(img, output, number_of_iterations = 10)

        # transform back to gamma space
        im_gam = adjust_gamma(output, gamma_coef)

        # add noise
        noise_lvl = np.random.uniform(0.1*noise_lvl, 0.5*noise_lvl, 3)
        im_noisy = add_noise(im_gam, noise_lvl)

        burst.append(im_noisy)

    return burst


class Dataset(data.Dataset):

    def __init__(self, path, max_images = None, kernel_lvl = 1, noise_lvl = 1, motion_blur_boolean = True, homo_align = False):
        self.path = path
        self.images = os.listdir(path)
        if max_images != None:
            self.images = self.images[:max_images]
        self.size = 160
        self.burst_length = 8
        self.kernel_lvl = kernel_lvl
        self.noise_lvl = noise_lvl
        self.motion_blur_boolean = motion_blur_boolean
        self.homo_align = homo_align

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):

        target = cv2.cvtColor(cv2.imread(os.path.join(self.path, self.images[index])), cv2.COLOR_BGR2RGB)
        target = cv2.resize(target, (self.size, self.size))

        bursts = make_burst_seq(target, self.burst_length, kernel_lvl = self.kernel_lvl,
                                noise_lvl = self.noise_lvl, motion_blur_boolean = self.motion_blur_boolean, homo_align= self.homo_align)

        # transpose to get (number of burst, c, w, h)
        return np.transpose(np.asarray(bursts), (0, 3, 1, 2)), np.transpose(target, (2, 0, 1))

