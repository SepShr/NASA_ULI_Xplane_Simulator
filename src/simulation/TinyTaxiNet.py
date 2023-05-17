from nnet import *
from PIL import Image

import numpy as np
import time

import mss
import cv2
import os

class TinyTaxiNet:
    def __init__(self, network=None) -> None:
        pass
    
        # Read in the network
        if network is None:
            filename = "../../models/TinyTaxiNet.nnet"
            self.network = NNet(filename)
        else:
            self.network = network

        ### IMPORTANT PARAMETERS FOR IMAGE PROCESSING ###
        self.stride = 16             # Size of square of pixels downsampled to one grayscale value
        # During downsampling, average the numPix brightest pixels in each square
        self.numPix = 16
        self.width = 256//self.stride    # Width of downsampled grayscale image
        self.height = 128//self.stride    # Height of downsampled grayscale image

        self.screenShot = mss.mss()
        self.monitor = {'top': 100, 'left': 100, 'width': 1720, 'height': 960}
        self.screen_width = 360  # For cropping
        self.screen_height = 200  # For cropping

    def getCurrentImage(self):
        """ Returns a downsampled image of the current X-Plane 11 image
            compatible with the TinyTaxiNet neural network state estimator

            NOTE: this is designed for screens with 1920x1080 resolution
            operating X-Plane 11 in full screen mode - it will need to be adjusted
            for other resolutions
        """
        # Get current screenshot
        img = cv2.cvtColor(np.array(self.screenShot.grab(self.monitor)),
                        cv2.COLOR_BGRA2BGR)[230:, :, :]
        img = cv2.resize(img, (self.screen_width, self.screen_height))
        img = img[:, :, ::-1]
        img = np.array(img)

        # Convert to grayscale, crop out nose, sky, bottom of image, resize to 256x128, scale so
        # values range between 0 and 1
        img = np.array(Image.fromarray(img).convert('L').crop(
            (55, 5, 360, 135)).resize((256, 128)))/255.0

        # Downsample image
        # Split image into stride x stride boxes, average numPix brightest pixels in that box
        # As a result, img2 has one value for every box
        img2 = np.zeros((self.height, self.width))
        for i in range(self.height):
            for j in range(self.width):
                img2[i, j] = np.mean(np.sort(
                    img[self.stride*i:self.stride*(i+1), self.stride*j:self.stride*(j+1)].reshape(-1))[-self.numPix:])

        # Ensure that the mean of the image is 0.5 and that values range between 0 and 1
        # The training data only contains images from sunny, 9am conditions.
        # Biasing the image helps the network generalize to different lighting conditions (cloudy, noon, etc)
        img2 -= img2.mean()
        img2 += 0.5
        img2[img2 > 1] = 1
        img2[img2 < 0] = 0
        return img2.flatten()

    def getStateTinyTaxiNet(self, client):
        """ Returns an estimate of the crosstrack error (meters)
            and heading error (degrees) by passing the current
            image through TinyTaxiNet

            Args:
                client: XPlane Client
        """
        image = self.getCurrentImage()
        pred = self.network.evaluate_network(image)
        return pred[0], pred[1], image
