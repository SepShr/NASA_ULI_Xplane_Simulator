import torch
import torch.nn as nn
from PIL import Image

import numpy as np
import mss
import cv2

class FullyConnectedReLU(nn.Module):
    def __init__(self):
        super(FullyConnectedReLU, self).__init__()

        self.fc1 = nn.Linear(128, 16)
        self.relu1 = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(16, 8)
        self.relu2 = nn.ReLU(inplace=True)
        self.fc3 = nn.Linear(8, 8)
        self.relu3 = nn.ReLU(inplace=True)
        self.fc4 = nn.Linear(8, 2)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.fc2(x)
        x = self.relu2(x)
        x = self.fc3(x)
        x = self.relu3(x)
        x = self.fc4(x)
        return x
    

class TinyTaxiNetTorch:
    def __init__(self, model_path=None) -> None:
        if model_path is None:
            raise NotImplementedError
        else:
            # ordered_dict = torch.load(model_path, map_location=torch.device('cuda') if torch.cuda.is_available() else 'cpu')
            ordered_dict = torch.load(model_path, map_location=torch.device('cpu'))
            self.model = FullyConnectedReLU()
            self.model.load_state_dict(ordered_dict)
            self.model.eval()
        
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
        
    def predict(self, state):
        return self.model(torch.from_numpy(state).float()).detach().numpy()
    
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
        pred = self.predict(image)
        return pred[0], pred[1], image
