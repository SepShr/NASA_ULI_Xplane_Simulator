"""
Generate and record data using CITL simulations.
"""
import gc
import os
import pickle
import sys
import time
from datetime import datetime
from pathlib import Path

# make sure this is a system variable in your bashrc
NASA_ULI_ROOT_DIR = os.environ['NASA_ULI_ROOT_DIR']
XPC3_DIR = NASA_ULI_ROOT_DIR + '/src/'
sys.path.append(XPC3_DIR)
NNET_DIR = NASA_ULI_ROOT_DIR + '/models/TinyTaxiNet.nnet'
sys.path.append(NNET_DIR)

import cv2
import numpy as np
import pandas as pd
from controllers import getProportionalControl
from mss import mss
from nnet import NNet
from pyDOE import lhs
from TinyTaxiNet import TinyTaxiNet
from tqdm import tqdm

from xpc3 import XPlaneConnect
from xpc3_helper import (getHomeState, getSpeed, reset, save_state_append,
                         sendBrake)

MONITOR={
    'screenshot_params': {
        'top': 100,
        'left': 100,
        'width': 1720,
        'height': 960
    },
    'width': 360,
    'height': 200
}

CLOUD_COVER_MAP = {
    0: 'clear',
    1: 'cirrus',
    2: 'scattered',
    3: 'broken',
    4: 'overcast'
}

RUNWAY_END_DTP_VALUE = 2982.0
START_DTP = RUNWAY_END_DTP_VALUE * 0.05
CTE_LIMIT = 15
HE_LIMIT = 90
NUM_SECTIONS = 4

MSS = mss()


def generate_jobs(
        start_CTE_range: tuple,
        start_HE_range: tuple,
        time_of_day_range: tuple,
        cloud_cover_range: tuple,
        end_DTP_perc: float,
        max_jobs: int
) -> list:
    """
    Generate a list of jobs to be run using controller-in-the-loop
    (CITL) simulations.
    """
    # Gather all parameter ranges into a dictionary
    param_ranges = {
        'time_of_day':time_of_day_range,
        'start_CTE': start_CTE_range,
        'start_HE': start_HE_range,
        'cloud_cover': cloud_cover_range
    }

    # Generate samples using Latin Hypercube Sampling (LHS)
    lhs_samples = lhs(4, samples=max_jobs, criterion='center')

    # Convert LHS samples to parameter values
    jobs = []
    for job_id, sample in enumerate(lhs_samples):
        job_params = {}
        job_params['job_id'] = job_id
        for key, param_range in param_ranges.items():
            param_value = param_range[0] + sample[list(param_ranges.keys()).index(key)] * (param_range[1] - param_range[0])
            job_params[key] = param_value
            if key == 'cloud_cover':
                job_params[key] = int(job_params[key])
            else:
                job_params[key] = round(job_params[key], 4)
        job_params['end_DTP'] = end_DTP_perc*RUNWAY_END_DTP_VALUE
        jobs.append(job_params)

    return jobs


def run_job(
        job_params: dict,
        model,
        model_type: str = 'nnet',
        out_dir: str = None,
        data_df : pd.DataFrame = None,
        csv_file: Path = None,
        sim_speed: float = 1.0
):
    """
    Run a single job using CITL simulations.
    """
    # Initialize TinyTaxiNet
    dnn_model = model
    dataframe = data_df

    with XPlaneConnect() as client:
        # Set weather and time of day
        client.sendDREF("sim/time/zulu_time_sec", job_params['time_of_day'] * 3600 + 8 * 3600)
        client.sendDREF("sim/weather/cloud_type[0]", job_params['cloud_cover'])

        # Run simulation
        client.sendDREF("sim/time/sim_speed", sim_speed)
        reset(
            client=client,
            cteInit=job_params['start_CTE'],
            heInit=job_params['start_HE'],
            dtpInit = START_DTP
            )
        sendBrake(client, 0)

        time.sleep(5)  # 5 seconds to get terminal window out of the way
        client.pauseSim(False)

        episode_start_time = client.getDREF("sim/time/zulu_time_sec")[0]
        step_start_time = client.getDREF("sim/time/zulu_time_sec")[0]
        step_end_time = step_start_time
        _, dtp, _ = getHomeState(client)
        current_step = 0  # current step in simulation
        dtp_range = job_params['end_DTP'] - START_DTP
        with tqdm(total=job_params['end_DTP']) as pbar:
            while dtp < job_params['end_DTP']:
                # Set proper throttle value based on speed
                speed = getSpeed(client)
                throttle = 0.1
                if speed > 5:
                    throttle = 0.0
                elif speed < 3:
                    throttle = 0.2
                
                # Environmental conditions
                local_time = client.getDREF("sim/time/local_time_sec")[0]
                if local_time < 5 * 3600 or local_time > 17 * 3600:
                    period_of_day = 2
                    time_period = 'night'
                elif local_time > 12 * 3600 and local_time < 17 * 3600:
                    period_of_day = 1
                    time_period = 'afternoon'
                else:
                    period_of_day = 0
                    time_period = 'morning'

                # Save screenshot
                img_name, img = save_screenshot(
                    time_period=time_period,
                    cloud_cover = job_params['cloud_cover'],
                    episode_num=job_params['job_id'],
                    step_num=current_step,
                    out_dir=out_dir
                )
                
                # Get state, i.e., CTE and HE
                cte_est, he_est, ttn_input_img = model.getStateTinyTaxiNet(client)

                # Get control
                rudder = getProportionalControl(client, cte_est, he_est)

                # Send control
                client.sendCTRL([0, rudder, rudder, throttle])

                # Get additional data to record
                cte_act, dtp, he_act = getHomeState(client)
                absolute_time = client.getDREF("sim/time/zulu_time_sec")[0]
                
                # Calculate section based on dtp
                # The number of sections is 4
                # 0: 0 - 25% of dtp_range
                # 1: 25% - 50% of dtp_range
                # 2: 50% - 75% of dtp_range
                # 3: 75% - 100% of dtp_range
                dtp_section = int((dtp - START_DTP) / (dtp_range / 4))

                # Record data
                dataframe = record_data(
                    img_name=img_name.split('/')[-1].split('.')[0],
                    img=img,
                    ttn_input_img=ttn_input_img,
                    absolute_time=absolute_time,
                    relative_time= absolute_time - episode_start_time,
                    cte_act=cte_act,
                    cte_act_norm=cte_act / 10.0,
                    cte_est=cte_est,
                    he_act=he_act,
                    he_act_norm=he_act / 30.0,
                    he_est=he_est,
                    dtp=dtp,
                    dtp_norm=dtp / 2982.0,
                    period_of_day=period_of_day,
                    cloud_cover=job_params['cloud_cover'],
                    episode_num=job_params['job_id'],
                    step_num=current_step,
                    section_num=dtp_section,
                    model_type=model_type,
                    dataframe=dataframe,
                    out_dir=out_dir,
                    csv_file=csv_file
                )
                save_state_append(client, str(out_dir), 'extra_params.csv')

                # Wait for next timestep
                while step_end_time - step_start_time < 1:
                    step_end_time = client.getDREF("sim/time/zulu_time_sec")[0]
                    time.sleep(0.001)
                
                # Set things for next round
                step_start_time = client.getDREF("sim/time/zulu_time_sec")[0]
                step_end_time = step_start_time
                _, dtp, _ = getHomeState(client)

                # Update progress bar
                delta_dtp = dtp - pbar.n
                pbar.update(delta_dtp)

                # Stop simulation if dtp is not changing
                if delta_dtp < 0.01:
                    break
                
                time.sleep(0.001)

                # stop simlation if CTE or HE are too large
                if cte_act > CTE_LIMIT or he_act > HE_LIMIT:
                    break
                else:
                    current_step += 1

        client.pauseSim(True)
    
    return dataframe


def setup_csv_file(out_dir: Path, csv_filename: str):
    """
    Setup CSV file for recording data.
    """
    # Make data folder if it doesn't exist
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    
    # Create CSV file
    csv_file_path = out_dir.joinpath(csv_filename)

    # Write header
    with open(csv_file_path, 'w') as csv_file:
        csv_file.write('image_filename,absolute_time_GMT_seconds,relative_time_seconds,distance_to_centerline_meters,')
        csv_file.write('distance_to_centerline_NORMALIZED,downtrack_position_meters,downtrack_position_NORMALIZED,')
        csv_file.write('heading_error_degrees,heading_error_NORMALIZED,period_of_day,cloud_type,episode_num,step_num,\n')


    return csv_file_path


def record_data(
        img_name: str,
        img: np.ndarray,
        ttn_input_img: np.ndarray,
        absolute_time: float,
        relative_time: float,
        cte_act: float,
        cte_act_norm: float,
        cte_est: float,
        he_act: float,
        he_act_norm: float,
        he_est: float,
        dtp: float,
        dtp_norm: float,
        period_of_day: int,
        cloud_cover: int,
        episode_num: int,
        step_num: int,
        section_num: int,
        model_type: str,
        dataframe: pd.DataFrame = None,
        out_dir: Path = None,
        csv_file: Path = None,
        # save_file: bool = True
        save_file: bool = False
):
    """
    Record CITL data.
    """
    data = {
        'image_filename': [img_name],
        'image': [img.tolist()],
        'ttn_input_img': [ttn_input_img.tolist()],
        'absolute_time_GMT_seconds': [absolute_time],
        'relative_time_seconds': [relative_time],
        'estimated_distance_to_centerline_meters': [cte_est],
        'actual_distance_to_centerline_meters': [cte_act],
        'actual_distance_to_centerline_NORMALIZED': [cte_act_norm],
        'estimated_heading_error_degrees': [he_est],
        'actual_heading_error_degrees': [he_act],
        'actual_heading_error_NORMALIZED': [he_act_norm],
        'downtrack_position_meters': [dtp],
        'downtrack_position_NORMALIZED': [dtp_norm],
        'period_of_day': [period_of_day],
        'cloud_type': [cloud_cover],
        'episode_num': [episode_num],
        'step_num': [step_num],
        'section_num': [section_num],
        'model_type': [model_type]
    }

    if dataframe is None:
        df = pd.DataFrame(data, index=[0])
    else:
        df = dataframe.reset_index(drop=True)  # Reset index of the existing dataframe
        df = pd.concat([df, pd.DataFrame(data, index=[0])], ignore_index=True)

    if csv_file is not None:
        df.to_csv(csv_file, mode='a', index=False, index_label=False, header=not os.path.exists(csv_file))

    if save_file:
        # Save dataframe
        df = df.reset_index(drop=True)  # Reset index of the existing dataframe
        df_path = out_dir.joinpath('data_df.pkl')
        df.to_pickle(df_path)

    return df


def save_df(
        dataframe: pd.DataFrame,
        out_dir: Path,
        df_suffix: str = '',
        save_file: bool = True
):
    """
    Save dataframe.
    """
    if save_file:
        # Save dataframe
        df = dataframe.reset_index(drop=True)  # Reset index of the existing dataframe
        df_path = out_dir.joinpath(f'data_df_{df_suffix}.pkl')
        df.to_pickle(df_path)

    # return df

def save_screenshot(
        time_period: str,
        cloud_cover: int,
        episode_num: int,
        step_num: int,
        out_dir: str,
        monitor: dict = MONITOR,
        save_file: bool = False
):
    """
    Save screenshot of current X-Plane window.
    """
    # Image information
    img = cv2.cvtColor(np.array(MSS.grab(monitor['screenshot_params'])),
                                        cv2.COLOR_BGRA2BGR)[230:, :, :]
    img = cv2.resize(img, (monitor['width'], monitor['height']))

    img_name = f'{out_dir}/MWH_Runway04_{time_period}_{CLOUD_COVER_MAP[cloud_cover]}_{str(episode_num)}_{str(step_num)}.png'
    
    if save_file:
        cv2.imwrite(img_name, img)

    img_array = np.array(img)
    img_array = img_array.flatten()

    return img_name, img_array

def initialize_ttn_model(model_type : str = 'nnet'):
    """
    Initialize TinyTaxiNet model.
    """
    if model_type == 'nnet':
        # NNet TinyTaxiNet
        tinytaxinet = TinyTaxiNet(NNet(NNET_DIR))
    else:
        raise ValueError('tinytaxinet_model_type must be "nnet"')
    
    return tinytaxinet


def main():

    # Create a new output directory
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    OUT_DIR = Path(NASA_ULI_ROOT_DIR).joinpath('scratch', 'data', 'citl', str(timestamp))
    os.makedirs(OUT_DIR, exist_ok=True)
    sys.path.append(str(OUT_DIR))

    # ====================Generate jobs==========================
    jobs = generate_jobs(
        start_CTE_range=(-3, 3),
        start_HE_range=(-10, 10),
        time_of_day_range=(0.0, 24.0),
        cloud_cover_range=(0, 4),
        end_DTP_perc=0.45,
        max_jobs=1000  # the number of jobs to generate (adjust as needed)
    )

    # Pickle jobs
    with open(OUT_DIR.joinpath('jobs.pkl'), 'wb') as f:
        pickle.dump(jobs, f)

    # NOTE: Comment out the above code and uncomment the below code to load jobs from a file
    # # ====================Load jobs==========================
    # # Load already generated jobs from the files
    # jobs_path = Path(NASA_ULI_ROOT_DIR).joinpath('generated_jobs').joinpath('jobs_p1.pkl')
    # with open(jobs_path, 'rb') as f:
    #     jobs = pickle.load(f)
    
    # print(f'loaded jobs: {jobs}')

    # ====================Run jobs==========================

    # Initialize csv file or keep it None
    # csv_file = setup_csv_file(OUT_DIR, 'data.csv')
    csv_file = None

    for model_type in ['nnet']:
        # Initialize TinyTaxiNet ('nnet')
        tinytaxinet = initialize_ttn_model(model_type=model_type)
        print(f'initialized {model_type} TinyTaxiNet')
    
        # Run jobs
        for job_counter, job in enumerate(jobs):
            data_df = pd.DataFrame()

            job_id = job['job_id']
            print(f'running job {job_counter+1} of {len(jobs)} (job_id={job_id})')
            data_df = run_job(
                job_params=job,
                model=tinytaxinet,
                model_type=model_type,
                out_dir=OUT_DIR,
                data_df=data_df,
                csv_file=csv_file
            )
            print(f'finished job {job_counter+1} of {len(jobs)} (job_id={job_id})')

            save_df(data_df, OUT_DIR, df_suffix=f'{model_type}_job_{job_id}')
            del data_df
            gc.collect()

if __name__ == "__main__":
    main()