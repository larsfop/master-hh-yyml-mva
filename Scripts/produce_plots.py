import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import uproot
import pandas as pd
from pathlib import Path
import argparse
import ROOT as R
import sys
import mpltern

sys.path.append(str(Path(__file__).resolve().parents[1]))
from utils.timer import Timer
from utils.metrics import roc_auc, Z_score


def plot_weights(
    weights: np.ndarray,
    name: str|Path,
    process: str,
    channel: str,
):
    
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.ticklabel_format(axis='x', style='sci', scilimits=(-3, 1), useMathText=True)
    
    ax.hist(weights, bins=100, histtype='step', color='black')
    ax.set_yscale('log')
    
    ax.set_xlabel('Weights', fontsize=26)
    ax.set_ylabel('Number of events', fontsize=26)
    ax.set_title(f'Weights distribution {process} - {channel}', fontsize=32)
    
    ax.tick_params(axis='both', which='major', labelsize=24)
    ax.xaxis.offsetText.set_fontsize(20)
    
    fig.savefig(name, bbox_inches='tight')
    
    print(f'{process} - negative weights: {weights[weights < 0].size/weights.size:.2%}')
    

def heatmap(
    df: pd.DataFrame, 
    x: str, 
    y: str, 
    z: str, 
    name: str|Path,
    fig: plt.Figure = None, 
    ax: plt.Axes = None
) -> tuple[plt.Figure, plt.Axes]:
    
    labels = {
        'eta': r'$\eta$',
        'lambda': r'$\lambda$',
        'batch_size': 'Batch size',
        'num_hidden_layers': 'Hidden layers',
        'num_hidden_neurons': 'Neurons',
        'test_auc_mean': 'AUC',
        'test_loss_mean': 'Loss',
    }
    
    ticks = {
        'eta': lambda x: [f'$\mathregular{{10^{{{np.log10(i):.0f}}}}}$' for i in np.unique(x)],
        'lambda': lambda x: [f'$\mathregular{{10^{{{np.log10(i):.0f}}}}}$' for i in np.unique(x)],
        'batch_size': lambda x: [f'{i:.0f}' for i in x],
        'num_hidden_layers': lambda x: [f'{i:.0f}' for i in x],
        'num_hidden_neurons': lambda x: [f'{i:.0f}' for i in x],
    }
    
    x_array = df[x].unique()
    y_array = df[y].unique()
        
    if fig is None or ax is None:
        fig, ax = plt.subplots(figsize=(x_array.size + 4, 8))
        
    data = df.pivot(y, x, z)
    
    sns.heatmap(
        data, 
        annot=True,
        fmt='.3f', 
        cmap='viridis', 
        cbar_kws={'label': labels[z]}, 
        ax=ax,
        xticklabels=ticks[x](x_array),
        yticklabels=ticks[y](y_array),
    )

    ax.set_xlabel(labels[x])
    ax.set_ylabel(labels[y])
    ax.set_title(f'Hyperparameter Optimisation: {labels[x]} vs {labels[y]}')
    
    fig.savefig(name)
    
    return fig, ax


def plot(
    data: pd.DataFrame,
    x: str,
    y: str,
    name: str|Path,
    channel: str,
    fig: plt.Figure = None,
    ax: plt.Axes = None,
) -> tuple[plt.Figure, plt.Axes]:
    
    if fig is None or ax is None:
        fig, ax = plt.subplots(figsize=(10, 8))
        ax.grid(which='both')
        
    ax.plot(data[x], data[y], label=y.split('_')[0].capitalize())
    
    ax.set_xlabel('Epochs', fontsize=26)
    ax.set_ylabel(y.split('_')[1].capitalize(), fontsize=26)
    ax.set_title(f'{y.split("_")[1].capitalize()} vs Epochs - {channel}', fontsize=32)
    
    ax.tick_params(axis='both', which='major', labelsize=24)
    ax.tick_params(axis='both', which='minor', labelsize=24)
    
    if 'loss' in y:
        ax.set_yscale('log')
    
    ax.legend(fontsize=24)
    
    fig.savefig(name, bbox_inches='tight')
    
    return fig, ax


def fill_histograms(
    datasets: dict[str, pd.DataFrame],
    bins: int,
    predID: str,
) -> dict[str, R.THStack]:
    
    def normalise_histogram(h: R.TH1F):
        dx = ( h.GetXaxis().GetXmax() - h.GetXaxis().GetXmin() ) / h.GetNbinsX()
        h.Scale(1.0 / h.GetSumOfWeights() / dx)
    
    
    h_stacks = {
        'train': R.THStack('hs_train', 'Train samples'),
        'test': R.THStack('hs_test', 'Test samples'),
    }
    
    for id, ds in datasets.items():
        pred = ds[predID].values
        weights = ds['weight'].values
        target = ds['classID'].values
        for classID in np.unique(target):
            h = R.TH1F(f'h_{id}_{classID}', f'h_{id}_{classID}', bins, 0, 1)
            h.FillN(
                len(pred[target == classID]),
                np.asarray(pred[target == classID], dtype=np.float64),
                np.asarray(weights[target == classID], dtype=np.float64)
            )
            
            normalise_histogram(h)
            
            if classID == 0:
                h.SetLineColor(R.kBlue)
            elif classID == 1:
                h.SetLineColor(R.kRed)
            else:
                h.SetLineColor(R.kGreen)
                
            if id == 'test':
                if classID == 0:
                    h.SetFillColorAlpha(R.kBlue, 0.2)
                elif classID == 1:
                    h.SetFillColorAlpha(R.kRed, 0.2)
                else:
                    h.SetFillColorAlpha(R.kGreen, 0.2)
            
            else:
                h.SetMarkerStyle(20)
                h.SetMarkerSize(0.5)
                h.SetMarkerColor(h.GetLineColor())
            
            h_stacks[id].Add(h)
    
    return h_stacks
    
    
def fill_canvas(
    h_stacks: dict[str, R.THStack],
    name: str|Path,
    channel: str,
    mva: str = 'DNN',
):
    
    name = str(name)
    
    C = R.TCanvas('c', 'c', 800, 600)
    
    hs_train = h_stacks['train']
    hs_test = h_stacks['test']
    
    mc = True if hs_train.GetNhists() > 2 else False
    
    frame = R.TH2F('frame', '', 100, 0, 1, 10, 0, 10)
    frame.SetTitle(f'Overtraining Check - {channel}')
    frame.GetXaxis().SetTitle(f'{mva} score')
    frame.GetYaxis().SetTitle('Events')
    frame.GetYaxis().SetTitleOffset(1.2)
    frame.GetYaxis().SetTitleSize(0.04)
    frame.GetXaxis().SetTitleSize(0.04)
    frame.GetXaxis().SetLabelSize(0.04)
    frame.GetYaxis().SetLabelSize(0.04)
    
    # R.TMVA.TMVAGlob.SetFrameStyle(frame)
    
    frame.Draw()
    
    hs_train.Draw('e1same nostack')
    hs_test.Draw('samehist nostack')
    
    ymax = max(hs_train.GetMaximum(), hs_test.GetMaximum()) + 1
    frame.GetYaxis().SetLimits(0, ymax)
    
    if mc:
        classIDS = ['Signal', 'Single Higgs', 'Continuum background']
    else:
        classIDS = ['Signal', 'Background']
    
    ks_scores = []
    
    legend_test = R.TLegend(0.1, 0.8, 0.5, 0.9)
    legend_train = R.TLegend(0.5, 0.8, 0.9, 0.9)
    for i, (h_train, h_test) in enumerate(zip(hs_train.GetHists(), hs_test.GetHists())):
        legend_train.AddEntry(h_train, f'{classIDS[i]} (Training sample)', 'ep')
        legend_test.AddEntry(h_test, f'{classIDS[i]} (Test sample)', 'f')
        
        ks_scores.append(h_train.KolmogorovTest(h_test, 'X'))
        
    legend_train.Draw()
    legend_test.Draw()
        
    if mc:
        pt = R.TPaveText(
            0.08,
            0.68,
            0.9,
            0.78,
            'nb NDC'
        )
        tt = pt.AddText(f'Kolmogorov-Smirnov test: Signal probability = {ks_scores[0]:.3f}')
        tt.SetTextSize(0.03)
        tt.SetTextAlign(12)
        
        tt = pt.AddText(f'                                              Single Higgs = {ks_scores[1]:.3f}')
        tt.SetTextSize(0.03)
        tt.SetTextAlign(12)
        
        tt = pt.AddText(f'                                              Continuum Background probability = {ks_scores[2]:.3f}')
        tt.SetTextSize(0.03)
        tt.SetTextAlign(12)
        
        pt.SetFillStyle(4000)
        
        pt.Draw()
        
    else:
        tt = R.TText(
            0.12, 
            0.76, 
            f'Kolmogorov-Smirnov test: Signal (Background) probability = {ks_scores[0]:.3f} ({ks_scores[1]:.3f})'
        )
        
        tt.SetNDC()
        tt.SetTextSize(0.03)
        tt.Draw()

    C.SetGrid()
    C.Draw()
    
    C.Print(name)
    

def overfitting_test(
    datasets: dict[str, pd.DataFrame],
    name: str|Path,
    fold: int|None,
    channel: str,
    mva: str = 'DNN',
    classification: str = 'binary',
    bins: int = 40,
):
    
    
    if classification == 'binary':
        h_stacks = fill_histograms(
            datasets,
            bins,
            predID=mva if fold is None else f'{mva}_fold{fold}'
        )
        
        fill_canvas(
            h_stacks,
            name=name,
            channel=channel,
            mva=mva,
        )
    else:
        for predID in ['DNN_signal', 'DNN_SH', 'DNN_CB']:
            h_stacks = fill_histograms(
                datasets,
                bins,
                predID if fold is None else f'{predID}_fold{fold}'
            )
            
            if fold is None:
                mc_name = name.with_stem(f'{name.stem}_{predID.split("_")[1]}')
            else:
                name = str(name)
                mc_name = name.split('_')
                mc_name.insert(-1, predID.split('_')[1])
                mc_name = '_'.join(mc_name)
            
            fill_canvas(
                h_stacks,
                name=mc_name,
                channel=channel,
                mva=mva,
            )
        
        
def plot_roc(
    data: pd.DataFrame,
    mva: str,
    channel: str,
    name: str|Path,
    fold: int|None = None,
    classification: str = 'binary',
    fig: plt.Figure = None,
    ax: plt.Axes = None,
) -> tuple[plt.Figure, plt.Axes]:
    
    if mva == 'DNN':
        label_name = 'DNN - BC' if classification == 'binary' else 'DNN - MC'
    else:
        label_name = 'BDTG      '
    
    if classification == 'binary':
        if mva == 'BDTG':
            ls = '-'
        else:
            ls = '-' if fig is None else '--'
    else:
        ls = '-' if fig is None else ':'
    
    if fig is None or ax is None:
        fig, ax = plt.subplots(figsize=(10, 8))
        
        ax.grid(which='both')
        ax.set_box_aspect(1)
        
        ax.set_xlabel('Background efficiency', fontsize=26)
        ax.set_ylabel('Signal efficiency', fontsize=26)
        
        if fold is None:
            ax.set_title(f'ROC curves - {channel}', fontsize=32)
        else:
            ax.set_title(f'ROC curves fold {fold} - {channel}', fontsize=32)
        
        ax.tick_params(axis='both', which='major', labelsize=24)
        ax.tick_params(axis='both', which='minor', labelsize=24)
        
        ax.axline((0, 0), slope=1, color='black', linewidth=0.8)
        
    if classification == 'binary':
        if mva == 'DNN':
            key = 'DNN' if fold is None else f'DNN_fold{fold}'
            pred = data[key].values
        else:
            key = 'BDTG' if fold is None else f'BDTG_fold{fold}'
            pred = data[key].values
    else:
        if fig is None or ax is None:
            keys = ['DNN_signal', 'DNN_SH', 'DNN_CB'] if fold is None else [f'DNN_signal_fold{fold}', f'DNN_SH_fold{fold}', f'DNN_CB_fold{fold}']
            pred = data[keys].values
            # pred = data.iloc[:, 2:5].values
        else:
            key = 'DNN_signal' if fold is None else f'DNN_signal_fold{fold}'
            pred = data[key].values      
        
    true = data['classID'].values
    weights = data['weight'].values
    
    if classification == 'binary':
        true = np.where(true == 2, 1, true)
    
    roc, auc = roc_auc(pred, true, weights)
    
    ax.plot(roc[0], roc[1], ls=ls, label=f'{label_name} (AUC = {auc:.3f})')
    
    ax.legend(loc='lower right', fontsize=16)
    
    fig.savefig(name, bbox_inches='tight')
        
    return fig, ax


def plot_ternary(
    data: pd.DataFrame,
    channel: str,
    name: str|Path,
    fold: int|None,
) -> None:
    
    if fold is None:
        pred = data.loc[:, ['DNN_signal', 'DNN_SH', 'DNN_CB']].values
    else:
        pred = data.loc[:, [f'DNN_signal_fold{fold}', f'DNN_SH_fold{fold}', f'DNN_CB_fold{fold}']].values
    true = data['classID'].values
    weights = data['weight'].values
    
    for i, (label, id) in enumerate(zip(['Signal', 'Single Higgs', 'Continuum background'], ['signal', 'SH', 'CB'])):
        sig = pred[true == i, 0]
        sh = pred[true == i, 1]
        cb = pred[true == i, 2]
        
        fig, ax = plt.subplots(figsize=(12, 8), subplot_kw=dict(projection='ternary'))

        if fold is None:
            title = f'Predicted {label} - {channel}'
        else:
            title = f'Predicted {label} fold {fold} - {channel}'
        ax.set_title(title, fontsize=32)
        ax.set_tlabel('Signal', fontsize=26)
        ax.set_llabel('Single Higgs', fontsize=26)
        ax.set_rlabel('Continuum background', fontsize=26)
        
        ax.taxis.set_label_position('tick1')
        ax.laxis.set_label_position('tick1')
        ax.raxis.set_label_position('tick1')
        
        ax.taxis.set_tick_params(labelsize=24)
        ax.laxis.set_tick_params(labelsize=24)
        ax.raxis.set_tick_params(labelsize=24)
        
        hist = ax.hexbin(
            sig, 
            sh, 
            cb, 
            gridsize=20, 
            cmap='jet',
            C=weights[true == i],
            reduce_C_function=np.sum,
        )
        
        cax = ax.inset_axes([1.1, -0.1, 0.05, 1.1], transform=ax.transAxes)
        cbar = fig.colorbar(hist, cax=cax)
        cbar.set_label(f'Predicted {label} samples', rotation=270, va='baseline', fontsize=26, labelpad=15)
        cbar.ax.tick_params(labelsize=24)
        
        ax.set_aspect('equal', anchor='SW')
        
        if fold is None:
            mc_name = name.with_stem(f'{name.stem}_{id}')
        else:
            name = str(name)
            mc_name = name.split('_')
            mc_name.insert(-1, id)
            mc_name = '_'.join(mc_name)
            
        ax.plot(
            (0.4, 0.4),
            (0.6, 0.4),
            (0, 0.2),
            color='red',
            linewidth=2,
        )
        
        ax.plot(
            (0.4, 0.4),
            (0.4, 0),
            (0.2, 0.6),
            color='red',
            linewidth=2,
            linestyle='--',
        )
        
        ax.plot(
            (0.8, 0.4),
            (0, 0.4),
            (0.2, 0.2),
            linewidth=2,
            color='red',
        )
        
        ax.plot(
            (0.4, 0),
            (0.4, 0.8),
            (0.2, 0.2),
            color='red',
            linewidth=2,
            linestyle='--',
        )

        ax.axline(
            (0.3, 0.7, 0),
            (0.3, 0, 0.7),
            color='green',
            linewidth=2,
        )
        
        fig.savefig(mc_name, bbox_inches='tight')
        
        
def plot_significance(
    data: pd.DataFrame,
    channel: str,
    name: str|Path,
    fold: int|None = None,
    mva: str = 'DNN',
    classification: str = 'binary',
    legend: list[str]|None = None,
    fig: plt.Figure = None,
    ax: plt.Axes = None,
) -> tuple[plt.Figure, plt.Axes]:
    
    if fig is None or ax is None:
        fig, ax = plt.subplots(figsize=(10, 8))
        ax.set_xlim(0, 1)
        
        ax.grid(which='both')
        
        ax.set_xlabel(f'MVA score', fontsize=26)
        ax.set_ylabel('Significance', fontsize=26)
        if fold is None:
            ax.set_title(f'Signal significance - {channel}', fontsize=32)
        else:
            ax.set_title(f'Signal significance fold {fold} - {channel}', fontsize=32)
        ax.tick_params(axis='both', which='major', labelsize=24)
        ax.tick_params(axis='both', which='minor', labelsize=24)
    
    if classification == 'binary':
        pred = data[mva].values if fold is None else data[f'{mva}_fold{fold}'].values
    else:
        if fold is None:
            pred = data.loc[:, ['DNN_signal', 'DNN_SH', 'DNN_CB']].values
        else:
            pred = data.loc[:, [f'DNN_signal_fold{fold}', f'DNN_SH_fold{fold}', f'DNN_CB_fold{fold}']].values
    
    true = data['classID'].values
    weights = data['weight'].values
    
    if classification == 'binary':
        true = np.where(true == 2, 1, true)
    
    thresholds = [0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95]
    
    if classification == 'binary':
        z = np.zeros_like(thresholds)
        for i, threshold in enumerate(thresholds):
            s = np.sum(weights[true == 0][pred[true == 0] > threshold])
            b = np.sum(weights[true == 1][pred[true == 1] > threshold])
            
            z[i] = Z_score(s, b)
            
        ax.plot(thresholds, z, 'o-')

        if legend is not None:
            fig.legend(legend, fontsize=24)
        
        fig.savefig(name, bbox_inches='tight')
        
    else:
        for i, (label, color) in enumerate(zip(['Signal', 'Single Higgs', 'Continuum background'], ['blue', 'red', 'green'])):
            z = np.zeros_like(thresholds)
            for j, threshold in enumerate(thresholds):
                if i == 0:
                    s = np.sum(weights[true == 0][pred[true == 0][:, i] > threshold])
                    b = np.sum(weights[true != 0][pred[true != 0][:, i] > threshold])
                else:
                    s = np.sum(weights[true == 0][pred[true == 0][:, i] < threshold])
                    b = np.sum(weights[true == i][pred[true == i][:, i] < threshold])
                
                z[j] = Z_score(s, b)
                
            ax.plot(thresholds, z, 'o-', label=label)
            
            if fold is None:
                ax.set_title(f'Signal significance - {channel}', fontsize=32, pad=125)
            else:
                ax.set_title(f'Signal significance fold {fold} - {channel}', fontsize=32, pad=125)
        
        fig.legend(fontsize=24, bbox_to_anchor=(0.67, 0.94), ncol=1)
        
        fig.tight_layout()
        fig.savefig(name)
    
    return fig, ax


def compare_significance(
    data: pd.DataFrame,
    name: str|Path,
    channel: str
):
    
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.grid(which='both')
    ax.set_xlim(0, 1)

    weights = data['weight'].values
    true = data['classID'].values

    thresholds = [0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95]
    for mva, label in zip(['DNN', 'BDTG'], ['NN - BC', 'BDT']):
        pred = data[mva].values
        z = np.zeros_like(thresholds)
        for i, threshold in enumerate(thresholds):
            s = np.sum(weights[true == 0][pred[true == 0] > threshold])
            b = np.sum(weights[true != 0][pred[true != 0] > threshold])
            
            z[i] = Z_score(s, b)
            
        ax.plot(thresholds, z, 'o-', label=label)
    
    pred = data.loc[:, ['DNN_signal', 'DNN_SH', 'DNN_CB']].values
    for i, (label, color) in enumerate(zip(['Signal vs Total bkg', 'Signal vs Single Higgs', 'Signal vs Continuum Background'], ['blue', 'red', 'green'])):
        z = np.zeros_like(thresholds)
        for j, threshold in enumerate(thresholds):
            if i == 0:
                s = np.sum(weights[true == 0][pred[true == 0][:, i] > threshold])
                b = np.sum(weights[true != 0][pred[true != 0][:, i] > threshold])
            else:
                s = np.sum(weights[true == 0][pred[true == 0][:, i] < threshold])
                b = np.sum(weights[true == i][pred[true == i][:, i] < threshold])
            
            z[j] = Z_score(s, b)
            
        # z /= np.max(z)
        
        ax.plot(thresholds, z, 'o-', label=f'{label} - MC')
    
    fig.legend(fontsize=16, bbox_to_anchor=(1.0, 0.93), loc='upper right', ncol=2)
    
    ax.set_xlabel('MVA score', fontsize=26)
    ax.set_ylabel('Significance', fontsize=26)
    ax.tick_params(axis='both', which='major', labelsize=24)
    ax.set_title(f'Signal significance - {channel}', fontsize=32, pad=100)
    
    fig.tight_layout()
    
    fig.savefig(name)
    

def main(
    channel: str = '1l0tau',
    suffix: str = 'pdf',
):
    
    R.gStyle.SetOptStat(0) # Disable the statistics box
    R.TH1.AddDirectory(False) # Disable the ROOT directory
    R.gErrorIgnoreLevel = 3000 # Ignore ROOT prints
    R.gROOT.SetBatch(True)  # Prevent ROOT from showing figures when drawing
    
    print('+' + '-' * 100 + '+')
    print(f'|{"Produce plots for {channel}".format(channel=channel):^100}|')
    print('+' + '-' * 100 + '+')
    
    
    # +--------------------------------------------------------------------------------------------------+
    # |                                 Hyperparameter Optimisation                                      |
    # +--------------------------------------------------------------------------------------------------+
    
    print('+' + '-' * 100 + '+')
    print(f'|{"Hyperparameter Optimisation":^100}|')
    print('+' + '-' * 100 + '+')
    
    with Timer(exit_msg='Finished Hyperparameter Optimisation'):
        input_hp = workspace_path / f'Output/{channel}/NN/Hyperparameter_Optimisation'
        output = input_hp / 'plots'
        
        output.mkdir(parents=True, exist_ok=True)
        
        # Grid search parameters
        hp_opts = {
            'lr': ('eta', 'lambda'),
            'lb': ('eta', 'batch_size'),
            'h': ('num_hidden_neurons', 'num_hidden_layers'),
        }
        
        for file in input_hp.glob(f'*.root'):
            hp = file.stem.split('_')[-1]
            x, y = hp_opts[hp]
            
            with uproot.open(file) as f:
                df = f['Hyperparameter_Optimisation'].arrays(library='pd')
                
            heatmap(
                df, 
                x=x,
                y=y,
                z=f'test_auc_mean',
                name=workspace_path / output / f'{file.stem}.{suffix}',
            )
            
        plt.close('all')
    
    # +--------------------------------------------------------------------------------------------------+
    # |                                          MVA Evaluation                                          |
    # +--------------------------------------------------------------------------------------------------+
    
    print('+' + '-' * 100 + '+')
    print(f'|{"MVA Evaluation":^100}|')
    print('+' + '-' * 100 + '+')
    
    roc_figs = {
        f'fold{fold}': {'fig': None, 'ax': None} for fold in range(1, 5)
    }
    
    # Create the output directory for the combined MVA plots
    combined_output = workspace_path / f'Output/{channel}/plots'
    combined_output.mkdir(parents=True, exist_ok=True)
    
    
    input = workspace_path / f'Output/{channel}/NN'
    output = input / 'plots'
    
    output.mkdir(parents=True, exist_ok=True)
    
    # Evaluate the BDTG training performance
    
    with Timer(exit_msg='Finished BDT evaluation',):
        input = workspace_path / f'Output/{channel}/BDTG'
        output = input / 'plots'
        
        output.mkdir(parents=True, exist_ok=True)
        # Read the data from each fold
        for i, file in enumerate(input.glob('BDTG*fold*.root'), 1):
            with uproot.open(file) as f:
                datasets = {
                    'train': f['TrainTree'].arrays(library='pd'),
                    'test': f['TestTree'].arrays(library='pd'),
                }
                
            datasets['train'][f'BDTG_fold{i}'] = (datasets['train'][f'BDTG_fold{i}'] + 1) / 2
            datasets['test'][f'BDTG_fold{i}'] = (datasets['test'][f'BDTG_fold{i}'] + 1) / 2
            
            fig, ax = plot_roc(
                    datasets['test'],
                    mva='BDTG',
                    channel=channel,
                    name=combined_output / f'{channel}_roc_fold{i}.{suffix}',
                    fold=i,
                    **roc_figs[f'fold{i}'],
                )
                
            roc_figs[f'fold{i}']['fig'] = fig
            roc_figs[f'fold{i}']['ax'] = ax
                
            # Plot the overfitting check for each fold
            overfitting_test(
                datasets,
                name=output / f'BDTG_{channel}_overfitting_check_fold{i}.{suffix}',
                fold=i,
                channel=channel,
                mva='BDTG',
            )
            
            
    # Evaluate the Neural Network training performance
    input = workspace_path / f'Output/{channel}/NN'
    output = input / 'plots'
    
    # Read the data from each fold
    for cl, label in zip(['bc', 'mc'], ['binary', 'multiclass']):
        with Timer(exit_msg=f'Finished NN {label} classification evaluation',):
            data = None
            for i, file in enumerate(input.glob(f'DNN_{cl}*fold*.root'), 1):
                with Timer(exit_msg=f'Finished processing fold {i}',):
                    with uproot.open(file) as f:
                        if data is None:
                            data = f['per_epoch'].arrays(library='pd')
                        else:
                            data += f['per_epoch'].arrays(library='pd')
                            
                        datasets = {
                            'train': f['TrainTree'].arrays(library='pd'),
                            'test': f['TestTree'].arrays(library='pd'),
                        }
                        
                    fig, ax = plot_roc(
                        datasets['test'],
                        mva='DNN',
                        channel=channel,
                        name=combined_output / f'{channel}_roc_fold{i}.{suffix}',
                        fold=i,
                        classification=label,
                        **roc_figs[f'fold{i}'],
                    )
                    
                    roc_figs[f'fold{i}']['fig'] = fig
                    roc_figs[f'fold{i}']['ax'] = ax
                        
                    # Plot the overfitting check for each fold
                    overfitting_test(
                        datasets,
                        name=output / f'DNN_{cl}_{channel}_overfitting_check_fold{i}.{suffix}',
                        fold=i,
                        channel=channel,
                        mva='DNN',
                        classification=label,
                    )
                    
                    if cl == 'mc':
                        plot_ternary(
                            datasets['test'],
                            channel=channel,
                            name=output / f'DNN_{cl}_{channel}_ternary_fold{i}.{suffix}',
                            fold=i,
                        )
                    
            with Timer(exit_msg='Finished processing evolving plots',):
                data /= i
                data['epochs'] = np.arange(1, len(data['train_loss']) + 1)
                    
                # Plot the loss rate
                fig, ax = plot(
                    data,
                    x='epochs',
                    y='train_loss',
                    channel=channel,
                    name=output / f'DNN_{cl}_{channel}_loss.{suffix}',
                )
                plot(
                    data,
                    x='epochs',
                    y='test_loss',
                    channel=channel,
                    name=output / f'DNN_{cl}_{channel}_loss.{suffix}',
                    fig=fig,
                    ax=ax,
                )
                
                # Plot the AUC
                fig, ax = plot(
                    data,
                    x='epochs',
                    y='train_auc',
                    channel=channel,
                    name=output / f'DNN_{cl}_{channel}_auc.{suffix}',
                )
                plot(
                    data,
                    x='epochs',
                    y='test_auc',
                    channel=channel,
                    name=output / f'DNN_{cl}_{channel}_auc.{suffix}',
                    fig=fig,
                    ax=ax,
                )
            
        plt.close('all')
        
    for cl, classification in zip(['bc', 'mc'], ['binary', 'multiclass']):
        with uproot.open(workspace_path / f'Output/{channel}/NN/DNN_{cl}_{channel}.root') as f:
            datasets = {
                'train': f['TrainTree'].arrays(library='pd'),
                'test': f['TestTree'].arrays(library='pd'),
            }
            
            overfitting_test(
                datasets,
                name=workspace_path / f'Output/{channel}/NN/plots/DNN_{cl}_{channel}_overfitting_check.{suffix}',
                fold=None,
                channel=channel,
                mva='DNN',
                classification=classification,
            )
            
            plot_significance(
                datasets['test'],
                channel=channel,
                name=workspace_path / f'Output/{channel}/NN/plots/DNN_{cl}_{channel}_significance.{suffix}',
                fold=None,
                mva='DNN',
                classification=classification,
            )
            
            plot_roc(
                datasets['test'],
                mva='DNN',
                channel=channel,
                name=workspace_path / f'Output/{channel}/NN/plots/DNN_{cl}_{channel}_roc.{suffix}',
                fold=None,
                classification=classification,
            )
            
            if classification == 'multiclass':
                plot_ternary(
                    datasets['test'],
                    channel=channel,
                    name=workspace_path / f'Output/{channel}/NN/plots/DNN_{cl}_{channel}_ternary.{suffix}',
                    fold=None,
                )
        
    # +--------------------------------------------------------------------------------------------------+
    # |                                          Compare MVAs                                            |
    # +--------------------------------------------------------------------------------------------------+
    
    print('+' + '-' * 100 + '+')
    print(f'|{"Compare MVAs":^100}|')
    print('+' + '-' * 100 + '+')
    
    with Timer(exit_msg='Finished comparing MVAs'):
        input_files = workspace_path / 'Output/Files'
        
        dataframes = []
        for file in input_files.glob(f'*{channel}.root'):
            if not 'data' in file.stem:
                with uproot.open(file) as f:
                    df = f['output'].arrays(['BDTG', 'DNN', 'DNN_signal', 'DNN_SH', 'DNN_CB', 'weight'], library='pd')
                    
                    if 'signal' in file.stem:
                        df['classID'] = 0
                    elif 'SH' in file.stem:
                        df['classID'] = 1
                    else:
                        df['classID'] = 2
                        
                    dataframes.append(df)
                    
                    # Plot weights distribution
                    plot_weights(
                        df['weight'].values,
                        name=workspace_path / f'Output/{channel}/plots/{file.stem}_weights.{suffix}',
                        process='yy+jets' if 'Sherpa' in file.stem else file.stem,
                        channel=channel,
                    )
        
        data = pd.concat(dataframes, ignore_index=True)
        
        compare_significance(data, combined_output / f'combined_{channel}_significance.{suffix}', channel)
            
        plot_significance(
            data,
            channel=channel,
            name=workspace_path / f'Output/{channel}/NN/plots/DNN_mc_{channel}_significance.{suffix}',
            mva='DNN',
            classification='multiclass',
        )
            
        fig = None
        ax = None
        for mva, classification in zip(['BDTG', 'DNN', 'DNN'], ['binary', 'binary', 'multiclass']):
            fig, ax = plot_roc(
                data,
                mva=mva,
                channel=channel,
                classification=classification,
                name=combined_output / f'combined_{channel}_roc.{suffix}',
                fig=fig,
                ax=ax,
            )
        
        plot_ternary(
            data,
            channel=channel,
            name=combined_output / f'combined_{channel}_ternary.{suffix}',
            fold=None,
        )
        
        plt.close('all')
    
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Produce plots from the dataset")
    parser.add_argument(
        "--input_path",
        type=str,
        default="../Files/plots/",
        help="Path to the input data file",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="../Files/plots/",
        help="Path to the output data file",
    )
    parser.add_argument(
        '-c', '--channel', type=str, default='1l0tau', help='Name of the channel. Default: 1l0tau'
    )
    parser.add_argument(
        '-s', '--suffix', type=str, default='pdf', help='Suffix of the output files. Default: pdf'
    )
    
    args = parser.parse_args()
    
    workspace_path = Path(__file__).resolve().parents[1]
    
    with Timer(exit_msg=f'Finished producing plots for {args.channel}'):
        main(
            channel=args.channel,
            suffix=args.suffix,
        )