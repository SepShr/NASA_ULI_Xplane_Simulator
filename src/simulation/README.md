# Simulation
**NOTE:** the image-based controllers in this repository are designed for a monitor with resolution 1920x1080 running in full screen mode. You can change back and forth from full screen mode is X-Plane 11 by opening the settings (second from the left in the upper right toolbar), opening the Graphics tab, and modifying the Monitor usage option under MONITER CONFIGURATION. For any other monitor configuration, the code would need to be modified. To navigate between terminals and X-Plane 11 when it is in full screen mode, use the `ALT + TAB` keyboard shortcut. 

In this folder, we provide an implementation of a proportional controller in for the taxinet problem in X-Plane 11.

## X-Plane 11 Dynamics
More information about the X-Plane 11 dynamics model can be found [here](https://www.x-plane.com/desktop/how-x-plane-works/). Control is performed using a rudder command that ranges between -1 and 1. Positive values create right turns, while negative values create left turns.

## Code Overview
The simulation code consists of the `citl_data_generation.py` python file which allows you to generate simulation jobs and run them sequntially. The file also allows you to also load previously generated simulation jobs. The parameters for the simulation should can be modified in the same file.

The file `controllers.py` contains functions for proportional control, and the file `tiny_taxinet.py` contains state estimation functions. The `nnet.py` file has some necessary functions for working with neural networks in the .nnet format (more info [here](https://github.com/sisl/NNet)).

## Quick Start Tutorial
This tutorial assumes that you are already followed the steps [here](..) and that X-Plane 11 is currently running in the proper configuration for taxinet.

1. Modify the parameters in `citl_data_generation.py` based on your desired simulation parameters

    Main parameters:

    `TIME_OF_DAY`
    * Time of day in local time, e.g. 8.0 = 8AM, 17.0 = 5PM

    `CLOUD_COVER`
    * Cloud cover (higher numbers are cloudier/darker)
    * 0 = Clear, 1 = Cirrus, 2 = Scattered, 3 = Broken, 4 = Overcast

    `START_CTE`
    * Starting crosstrack error in meters (for initial position on runway)

    `START_HE`
    * Starting heading error in degrees (for initial position on runway)

    `START_DTP`
    * Starting downtrack position in meters (for initial position on runway)

    `END_DTP`
    * Used to determine when to stop the simulation

Alternatively, you can run the previously generated simulation jobs by commenting out the data generation section of the code and uncommenting the data loading section in the `citl_data_generation.py` file.

**NOTE:** the previously generated jobs are included in `~/generated_jobs/` folder. However, you can generate new jobs by running the `citl_data_generation.py` file which will be recoded in the `~/scratch/` folder.

2. Open a terminal and navigate to `NASA_ULI_Xplane_Simulator/src/simulation`.

3. In the terminal first create a virtual environment, install the requirements and run the data generation code by running the following commands:
```bash
 # For Linux-based systems
#  For Windows-based systems use python instead of python3
 python3 -m venv ./venv
 source venv/bin/activate
 python -m pip install --upgrade pip
 pip install -r requirements.txt
 
 cd ~/src/simulation
 python citl_data_generation.py
```

1. Quickly minimize the terminal (if using the image-based controller) so that it does not get in the way of the screenshots. There should be a five second buffer since starting the `citl_data_generation.py` script.
