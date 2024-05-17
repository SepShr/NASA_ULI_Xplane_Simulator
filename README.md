# NASA ULI Xplane Simulator
This repository contains the instructions and code for setting up the NASA ULI Xplane Simulator to generate data using controller and DNN (i.e., system)-in-the-loop simulation for the paper titled "System Safety Monitoring of Learned Components Using Temporal Metric Forecasting", by Sepehr Sharifi, Andrea Stocco, and Lionel C. Briand.

This work is done at [Nanda Lab](https://www.nanda-lab.ca/), EECS Department, University of Ottawa, Canada.

This project is modified fork of the [NASA ULI Project](https://github.com/StanfordASL/NASA_ULI_Xplane_Simulator). We are thankful to the authors for their work.

(NOTE: if you do not have a proper markdown viewer, you can use [this online viewer](https://dillinger.io))

# System Requirements
First, export a system (bash) variable corresponding to where you have cloned this repo named `NASA_ULI_ROOT_DIR`. For example, in your bashrc:

`export NASA_ULI_ROOT_DIR={path to this directory on your machine}`

This code was tested using Python 3.9. In general, any version of Python 3 should work.
See src/requirements.txt for specific packages.

# Quick Links
* [X-Plane 11 Set Up Instructions](src/README.md)
* [Simulation Instructions](src/simulation/README.md)

# Repository Structure
- `src`
    - Has the main code. See `src/examples` for a tutorial.

- `scratch`
    - Create this folder on your machine, where results will be saved. Do not check this into the main GIT repository to keep it compact.

- `models`
    - has a Tiny TaxiNet model that takes in a flattened 8 x 16 image
    - predicts two scalars
        - estimate of the crosstrack error (meters)
        - heading error (degrees) 
        - these are un-normalized outputs
    - see the code in `simulation` for controller-in-the-loop training

- `pretrained_DNN`
    - has ResNet-18 with 2 final linear regression outputs pre-trained on morning condition data
    - was trained used the scripts in `train_DNN`
