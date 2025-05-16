from copy import deepcopy
import numpy as np
import matplotlib.pyplot as plt
import uproot
import ROOT as R
import argparse
from scipy.interpolate import make_interp_spline
from pathlib import Path
import mpltern


def cut(true: np.ndarray, events: np.ndarray, pred: np.ndarray, **cut_values):
    labels = {'sig': 0, 'sh': 1, 'bkg': 2}
    keys = list(cut_values.keys())
    sig_sum = np.sum(events[true == 0])
    sh_sum = np.sum(events[true == 1])
    bkg_sum = np.sum(events[true == 2])

    rms = 1e10
    optimal_cut = {}
    def recursive(cuts: list, rms: float, optimal_cut: dict, i: int = 0, **cut_values):
        if i == len(cut_values):
            return cuts, rms, optimal_cut

        key = keys[i]
        for cut in np.arange(*cut_values[key]):
            cuts[i] = cut
            cuts, rms, optimal_cut = recursive(cuts, rms, optimal_cut, i + 1, **cut_values)

            if i == len(cut_values) - 1:
                # cut = (pred[:, 0] > cuts[0]) & (pred[:, 1] < cuts[1]) & (pred[:, 2] < cuts[2])

                cut = pred[:, labels[keys[0]]] > cuts[0] if keys[0] == 'sig' else pred[:, labels[keys[0]]] < cuts[0]
                for key, value in zip(keys[1:], cuts[1:]):
                    cut &= pred[:, labels[key]] > value if key == 'sig' else pred[:, labels[key]] < value

                sig = np.sum(events[(true == 0) & cut])/sig_sum
                sh = np.sum(events[(true == 1) & cut])/sh_sum
                bkg = np.sum(events[(true == 2) & cut])/bkg_sum

                total_bkg = (np.sum(events[(true == 1) & cut]) + np.sum(events[(true == 2) & cut])) / (sh_sum + bkg_sum)

                if np.abs(sig - 0.8) < 0.01:
                    rms_new = (1 - sig)**2 + sh**2 + bkg**2
                    if rms_new < rms:
                        rms = rms_new
                        optimal_cut = {'cuts': {k: v for k, v in zip(keys, cuts)}, 'sig': sig, 'sh': sh, 'bkg': bkg, 'total_bkg': total_bkg}
                        
        return cuts, rms, optimal_cut

    _, _, optimal_cut = recursive(cuts=[0] * len(cut_values), rms=rms, optimal_cut=optimal_cut, **cut_values)
    
    return optimal_cut


def plot_regions(ax, *splits: float):
    for split in splits:
        ax.plot((split, split), (0, 1e6), ls='--', color='red')
        

def main(channel: str, input_path: Path):
    processes = ['signal_ggF', 'signal_VBF', 'SH', 'Sherpa', 'Vyy']

    pred = np.empty((0, 3))
    true = np.empty(0, dtype=np.int64)
    weights = np.empty(0)

    for process in processes:
        with uproot.open(input_path / f'{process}_{channel}.root') as f:
            output = f['output'].arrays(library='np')
            size = len(output['DNN_MC_Signal'])

            print(f'Processing {process}_{channel}.root with {size} events')
            
            weights = np.concatenate((weights, output['weight']))
            
            dh = output['DNN_MC_Signal']
            sh = output['DNN_MC_Single_Higgs']
            cb = output['DNN_MC_Background']
            
            pred = np.concatenate((pred, np.column_stack((dh, sh, cb))))
            
            if 'signal' in process:
                true = np.concatenate((true, np.zeros(size)))
            elif 'SH' in process:
                true = np.concatenate((true, np.ones(size)))
            else:
                true = np.concatenate((true, np.full(size, 2)))
    
    
    optimal_cuts = cut(true, weights, pred, sig=(0, 1, 0.05), sh=(0, 1, 0.05), bkg=(0, 1, 0.05))
    
    print(optimal_cuts['cuts'])
    
    sig = optimal_cuts['sig']
    sh = optimal_cuts['sh']
    bkg = optimal_cuts['bkg']
    total_bkg = optimal_cuts['total_bkg']
    
    sig_cut, sh_cut, bkg_cut = list(optimal_cuts['cuts'].values())
    
    print(f'Signal events: {sig:.3f}')                
    print(f'Single Higgs events: {sh:.3f}')                
    print(f'Continuum background events: {bkg:.3f}')   
    print(f'Total background events: {total_bkg:.3f}')             
                
    # +-----------------------------------+
    # |          Ternary plots            |
    # +-----------------------------------+
    cmap = ['Blues', 'Greens', 'Reds']
    labels = ['Signal', 'Single Higgs', 'Continuum Background']

    # fig = plt.figure(figsize=(12, 2.8))
    fig = plt.figure(figsize=(6, 12))
    # fig.subplots_adjust(wspace=1)
    fig.subplots_adjust(hspace=0.6)
    for i in range(3):
        ax = fig.add_subplot(3, 1, i+1, projection='ternary')

        # Plot the data with the mc weights
        hist = ax.hexbin(pred[true==i, 0], pred[true==i, 1], pred[true==i, 2], gridsize=20, cmap='plasma', C=weights[true==i], reduce_C_function=np.sum)
        
        # Set labels
        ax.set_tlabel('Signal')
        ax.set_llabel('Single Higgs')
        ax.set_rlabel('Continuum Background')
        
        ax.taxis.set_label_position('tick1')
        ax.laxis.set_label_position('tick1')
        ax.raxis.set_label_position('tick1')
        
        # Plot the cuts
        
        # Cut on the Signal events
        ax.axline(
            (sig_cut, 1 - sig_cut, 0),
            (sig_cut, 0, 1 - sig_cut),
            color = 'blue',
            ls = '--',
        )
        
        # Cut on the Single Higgs events
        ax.axline(
            (1 - sh_cut, sh_cut, 0),
            (0, sh_cut, 1 - sh_cut),
            color = 'green',
            ls = '--',
        )
        
        # Cut on the Continuum Background events
        ax.axline(
            (1 - bkg_cut, 0, bkg_cut),
            (0, 1 - bkg_cut, bkg_cut),
            color = 'red',
            ls = '--',
        )
        
        # Create a colorbar
        cax = ax.inset_axes([1.2, -0.1, 0.05, 1.1], transform=ax.transAxes)
        cbar = fig.colorbar(hist, cax=cax)
        cbar.set_label(f'Predicted {labels[i]}', rotation=270, va='baseline')  
        
    # fig.suptitle(f'Ternary plot ({channel})')
    fig.savefig(f'Plots/{channel}/ternary_{channel}.png')          
    

if __name__=="__main__":
    parser = argparse.ArgumentParser(
        description='Producing plots generated by the DNN and BDTG.'
    )
    
    parser.add_argument(
        '-c', '--channel', type=str, default='1l0tau', help='Name of channel, default: 1l0tau'
    )
    parser.add_argument(
        '-i', '--input_path', type=Path, default=Path('../Files'), help='Input path for the root files, these files must already exist. Default: ../Files'
    )
    
    args = parser.parse_args()
    
    main(args.channel, args.input_path)