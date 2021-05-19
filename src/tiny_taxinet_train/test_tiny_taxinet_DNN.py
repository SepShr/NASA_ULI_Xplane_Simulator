"""
    Tests a pre-trained DNN model
"""

import time
import copy
import torch
import numpy as np
from tqdm.autonotebook import tqdm as tqdm
from torch.utils.tensorboard import SummaryWriter

from model_taxinet import TaxiNetDNN, freeze_model
from taxinet_dataloader import *
from plot_utils import *

# make sure this is a system variable in your bashrc
NASA_ULI_ROOT_DIR=os.environ['NASA_ULI_ROOT_DIR']

DATA_DIR = os.environ['NASA_DATA_DIR'] 

# where intermediate results are saved
# never save this to the main git repo
SCRATCH_DIR = NASA_ULI_ROOT_DIR + '/scratch/'

UTILS_DIR = NASA_ULI_ROOT_DIR + '/src/utils/'
sys.path.append(UTILS_DIR)


TRAIN_DNN_DIR = NASA_ULI_ROOT_DIR + '/src/train_DNN/'
sys.path.append(TRAIN_DNN_DIR)

from textfile_utils import *
from tiny_taxinet_dataloader import *
from model_tiny_taxinet import TinyTaxiNetDNN
from test_final_DNN import test_model


if __name__=='__main__':

    torch.cuda.empty_cache()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    print('found device: ', device)

    # where the training results should go
    results_dir = remove_and_create_dir(SCRATCH_DIR + '/test_tiny_taxinet_DNN/')

    # tests the pre-trained model
    model_dir = NASA_ULI_ROOT_DIR + '/pretrained_DNN/' 
    #'epoch_30_resnet18/'

    evaluation_condition_list = ['afternoon', 'morning', 'night', 'overcast']

    # larger images require a resnet, downsampled can have a small custom DNN
    dataset_type = 'tiny_images'

    train_test_val = 'test'

    with open(results_dir + '/results.txt', 'w') as f:
        header = '\t'.join(['evaluation_condition', 'train_condition', 'train_test_val', 'dataset_type', 'model_type', 'loss', 'mean_inference_time_sec', 'std_inference_time_sec']) 
        f.write(header + '\n')

        for evaluation_condition in evaluation_condition_list:
            # where raw images and csvs are saved
            BASE_DATALOADER_DIR = DATA_DIR + '/' + dataset_type  + '/' + condition

            test_dir = BASE_DATALOADER_DIR + '/' + condition + '_' + train_test_val

            dataloader_params = {'batch_size': 128,
                                 'shuffle': False,
                                 'num_workers': 12,
                                 'drop_last': False}

            # MODEL
            # instantiate the model 
            model = TinyTaxiNetDNN()

            # load the pre-trained model
            if device.type == 'cpu':
                model.load_state_dict(torch.load(model_dir + '/best_model.pt', map_location=torch.device('cpu')))
            else:
                model.load_state_dict(torch.load(model_dir + '/best_model.pt'))
            
            model = model.to(device)
            model.eval()

            # DATALOADERS
            # instantiate the model and freeze all but penultimate layers
            test_dataset = TaxiNetDataset(test_dir)

            test_loader = DataLoader(test_dataset, **dataloader_params)

            # LOSS FUNCTION
            loss_func = torch.nn.MSELoss().to(device)

            # DATASET INFO
            datasets = {}
            datasets['test'] = test_dataset

            dataloaders = {}
            dataloaders['test'] = test_loader

            # TEST THE TRAINED DNN
            test_results = test_model(model, test_dataset, test_loader, device, loss_func)
            test_results['model_type'] = 'trained'

            out_str = '\t'.join([condition, train_test_val, dataset_type, test_results['model_type'], str(test_results['losses']), str(test_results['time_per_item']), str(test_results['time_per_item_std'])]) 
            f.write(out_str + '\n')

            # COMPARE WITH A RANDOM, UNTRAINED DNN
            untrained_model = TaxiNetDNN()
            untrained_model = untrained_model.to(device)

            test_results = test_model(untrained_model, test_dataset, test_loader, device, loss_func)
            test_results['model_type'] = 'untrained'

            out_str = '\t'.join([condition, train_test_val, dataset_type, test_results['model_type'], str(test_results['losses']), str(test_results['time_per_item']), str(test_results['time_per_item_std'])]) 
            f.write(out_str + '\n')
