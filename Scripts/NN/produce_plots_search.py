import numpy as np
import matplotlib.pyplot as plt
import uproot
import ROOT as R
import argparse
from scipy.interpolate import make_interp_spline
from pathlib import Path
import time
import json

def main(
    channel: str,
    classification: str,
):
    
    cl_dir = 'BC' if classification == 'binary' else 'MC'
    path = Path(f'Hyperparameter_Optimization/{channel}/{cl_dir}')
    
    with open(path / f'{channel}.json', 'r') as f:
        config = json.load(f)

    keys = list(config.keys())
    keys.sort()
    
    data = {}
    with uproot.open(path / f'{channel}.root') as f:
        for key in keys:
            data[key] = f[key].arrays(library='np')
            
    weights = []
    auc = []
    loss = []

    for key in keys:
        train_loss, train_auc, test_loss, test_auc = list(data[key].values())
        weight = config[key]['class_weights']
        
        if weight == 'balanced':
            weight_b = weight
            auc_b = np.max(test_auc)
            loss_b = np.min(test_loss)
        elif weight == 'none':
            weight_n = weight
            auc_n = np.max(test_auc)
            loss_n = np.min(test_loss)
        else:
            weights.append(weight)
            auc.append(np.max(test_auc))
            loss.append(np.min(test_loss))
            
    idx_sort = np.argsort(weights)
    weights = np.array(weights)[idx_sort]
    auc = np.array(auc)[idx_sort]
    
    # Plot AUC vs Class Weights
    plt.figure()
    plt.plot(weights, auc, '--ob')
    plt.axline([0, auc_b], [1, auc_b], color='g', ls='--', label=f'Balanced: {auc_b:.3f}')
    plt.axline([0, auc_n], [1, auc_n], color='r', ls='--', label=f'None: {auc_n:.3f}')

    plt.grid()
    plt.legend()

    plt.xlabel('Class Weights')
    plt.ylabel('AUC')
    
    if classification == 'binary':
        plt.title(f'{channel} Binary Classification: Class weights', fontdict={'fontsize': 14})
    else:
        plt.title(f'{channel} Multiclass Classification: Class weights', fontdict={'fontsize': 14})
    
    plt.savefig(f'Plots/{channel}/{cl_dir}/class_weights.png')
    
    # Plot loss vs Class Weights
    plt.figure()
    plt.plot(weights, loss, '--ob')
    plt.axline([0, loss_b], [1, loss_b], color='g', ls='--', label=f'Balanced: {loss_b:.3f}')
    plt.axline([0, loss_n], [1, loss_n], color='r', ls='--', label=f'None: {loss_n:.3f}')
    
    plt.grid()
    plt.legend()
    
    plt.xlabel('Class Weights')
    plt.ylabel('Loss')
    
    if classification == 'binary':
        plt.title(f'{channel} Binary Classification: Class weights', fontdict={'fontsize': 14})
    else:
        plt.title(f'{channel} Multiclass Classification: Class weights', fontdict={'fontsize': 14})
        
    plt.savefig(f'Plots/{channel}/{cl_dir}/class_weights_loss.png')
    
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    
    parser.add_argument(
        '-c', '--channel', type=str, default='1l0tau', help='Name of channel, default: 1l0tau'
    )
    parser.add_argument(
        '-m', '--multiclass', action='store_true', help='Multiclass classification'
    )
    
    args = parser.parse_args()
    
    classification = 'multiclass' if args.multiclass else 'binary'
    
    main(
        channel=args.channel,
        classification=classification,
    )