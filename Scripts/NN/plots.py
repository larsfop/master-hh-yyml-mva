import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import uproot
import ROOT as R
import argparse
import sys

sys.path.append(str(Path(__file__).resolve().parents[1]))

from utils.metrics import auc_score, Z_score, roc_auc


def plot(
    x: np.ndarray, 
    *y_arrays: np.ndarray, 
    yscale='linear',
    name: str = None,
    prefix: str = 'pdf',
    path: Path = Path('Output'),
    fig: plt.Figure = None,
    ax: plt.Axes = None,
) -> tuple[plt.Figure, plt.Axes]:
    
    if len(y_arrays) == 2:
        legend = ['Train sample', 'Test sample']
    
    if fig is None or ax is None:
        fig, ax = plt.subplots(figsize=(14, 8))
    
    for y in y_arrays:
        
        ax.plot(x, y)
        
    ax.grid(which='both')
    ax.legend(legend, fontsize=20)
    
    ax.set_yscale(yscale)
    
    ax.tick_params(axis='both', which='major', labelsize=16)
    ax.tick_params(axis='both', which='minor', labelsize=16)
    
    if 'loss' in name:
        ax.set_ylabel('Mean loss rate', fontsize=18)
    elif 'auc' in name:
        ax.set_ylabel('AUC', fontsize=18)
        
    ax.set_xlabel('Epochs', fontsize=18)
    ax.set_title(name.replace('_', ' '), fontsize=24)
    
    fig.savefig(path / f'{name}.{prefix}')
    
    return fig, ax
    
    
def plot_significances(
    pred: np.ndarray,
    true: np.ndarray,
    weights: np.ndarray,
    thresholds: np.ndarray,
    name: str,
    path: Path,
    prefix: str = 'pdf',
    legend: list[str] = None,
    fig: plt.Figure = None,
    ax: plt.Axes = None,
):
    
    s = weights[true == 1]
    b = weights[true == 0]
    
    pred_s = pred[true == 1]
    pred_b = pred[true == 0]
    
    if fig is None or ax is None:
        fig, ax = plt.subplots(figsize=(14, 8))
        
        xmin = -1 if np.min(pred) < 0 else 0
        ax.set_xlim(xmin, 1)
            
        ax.grid(which='both')
        
        ax.set_xlabel('MVA score', fontsize=18)
        ax.set_ylabel('Significance', fontsize=18)
        
        ax.tick_params(axis='both', which='major', labelsize=16)
        ax.tick_params(axis='both', which='minor', labelsize=16)
    
    
    z_points = []
    for threshold in thresholds:
        z = Z_score(
            np.sum(s[pred_s > threshold]),
            np.sum(b[pred_b > threshold])
        )
        z_points.append(z)
        
    ax.plot(thresholds, z_points, 'o-')
    
    if legend is not None:
        fig.legend(legend, fontsize=20)
    
    fig.savefig(path / f'{name}.{prefix}')
    
    return fig, ax


def normalise_histograms(hist1, hist2):
    R.TMVA.TMVAGlob.NormalizeHists(hist1, hist2)
    return hist1, hist2


def overfitting_test(
    train_data: dict[str, np.ndarray],
    test_data: dict[str, np.ndarray],
    channel: str,
    path: Path,
    mva: str = 'DNN',
    prefix: str = 'pdf',
):
    
    bins = 40
    h_train_signal = R.TH1F('train_signal', 'train_signal', bins, 0, 1)
    h_train_background = R.TH1F('train_background', 'train_background', bins, 0, 1)
    h_test_signal = R.TH1F('test_signal', 'test_signal', bins, 0, 1)
    h_test_background = R.TH1F('test_background', 'test_background', bins, 0, 1)
    
    train_pred = train_data['DNN']
    train_target = train_data['target']
    train_weights = train_data['weights']
    
    test_pred = test_data['DNN']
    test_target = test_data['target']
    test_weights = test_data['weights']
    
    h_train_signal.FillN(
        len(train_pred[train_target == 1]),
        np.asarray(train_pred[train_target == 1], dtype=np.float64),
        np.asarray(train_weights[train_target == 1], dtype=np.float64)
    )
    h_train_background.FillN(
        len(train_pred[train_target == 0]),
        np.asarray(train_pred[train_target == 0], dtype=np.float64),
        np.asarray(train_weights[train_target == 0], dtype=np.float64)
    )
    h_test_signal.FillN(
        len(test_pred[test_target == 1]),
        np.asarray(test_pred[test_target == 1], dtype=np.float64),
        np.asarray(test_weights[test_target == 1], dtype=np.float64)
    )
    h_test_background.FillN(
        len(test_pred[test_target == 0]),
        np.asarray(test_pred[test_target == 0], dtype=np.float64),
        np.asarray(test_weights[test_target == 0], dtype=np.float64)
    )
    
    h_train_signal, h_train_background = normalise_histograms(
        h_train_signal, h_train_background
    )
    h_test_signal, h_test_background = normalise_histograms(
        h_test_signal, h_test_background
    )
    
    C = R.TCanvas('c', 'c', 800, 600)
    
    h_test_signal.SetLineColor(R.kBlue)
    h_test_background.SetLineColor(R.kRed)
    
    h_test_signal.SetFillColorAlpha(R.kBlue, 0.2)
    h_test_background.SetFillColorAlpha(R.kRed, 0.2)
    
    h_test_signal.Draw('samehist')
    h_test_background.Draw('samehist')
    
    h_train_signal.SetLineColor(R.kBlue)
    h_train_background.SetLineColor(R.kRed)
    
    h_train_signal.Draw('elsame')
    h_train_background.Draw('elsame')
    
    h_test_signal.SetTitle(f'Overtraining Check - {channel}')
    h_test_signal.SetXTitle(f'{mva} score')
    h_test_signal.SetYTitle('Events')
    
    ymax = np.max(
        [
            h_train_signal.GetMaximum(),
            h_train_background.GetMaximum(),
            h_test_signal.GetMaximum(),
            h_test_background.GetMaximum()
        ]
    ) + 1
    h_test_signal.GetYaxis().SetRangeUser(0, np.floor(ymax))
    
    legend1 = R.TLegend(0.1, 0.8, 0.5, 0.9)
    legend1.AddEntry(h_test_signal, 'Signal (Test sample)', 'f')
    legend1.AddEntry(h_test_background, 'Background (Test sample)', 'f')
    legend1.Draw()
    
    legend2 = R.TLegend(0.5, 0.8, 0.9, 0.9)
    legend2.AddEntry(h_train_signal, 'Signal (Train sample)', 'l')
    legend2.AddEntry(h_train_background, 'Background (Train sample)', 'l')
    legend2.Draw()
    
    ks_signal = h_train_signal.KolmogorovTest(h_test_signal)
    ks_background = h_train_background.KolmogorovTest(h_test_background)
    
    print(f'KS: signal (background) = {ks_signal} ({ks_background})')
    
    tt = R.TText(
        0.12, 0.76, f'Kolmogorov-Smirnov test: signal (background) probability = {ks_signal:.3f} ({ks_background:.3f})'
    )
    tt.SetNDC()
    tt.SetTextSize(0.03)
    tt.Draw()
    
    C.SetGrid()
    C.Draw()
    
    C.Print(str(path) + f'/{channel}_overfitting_check.{prefix}')
    
    
def plot_roc(
    y_pred: np.ndarray,
    y_true: np.ndarray,
    weights: np.ndarray,
    channel: str,
    mva: str,
    path: Path,
    prefix: str = 'pdf',
    fig: plt.Figure = None,
    ax: plt.Axes = None,
) -> tuple[plt.Figure, plt.Axes]:
    
    if fig is None or ax is None:
        fig, ax = plt.subplots(figsize=(12, 12))
    
        ax.grid(which='both')
        ax.set_box_aspect(1)
        
        ax.set_xlabel('Background efficiency', fontsize=18)
        ax.set_ylabel('Signal efficiency', fontsize=18)
        
        ax.set_title(f'ROC curve - {channel}', fontsize=24)
        
        ax.tick_params(axis='both', which='major', labelsize=16)
        ax.tick_params(axis='both', which='minor', labelsize=16)
        
        ax.axline((0, 0), slope=1, color='red', linestyle='--')
    
    roc, auc = roc_auc(
        y_pred,
        y_true,
        weights
    )
    
    ax.plot(roc[0], roc[1], label=f'AUC - {mva} = {auc:.3f}')
    
    ax.legend(fontsize=20, loc='lower right')
    fig.savefig(path / f'{channel}_roc_curve.{prefix}')
    
    return fig, ax
        

def main(
    channel: str,
    classification: str,
    mva: str,
    prefix: str,
):
    
    cl = 'bc' if classification == 'binary' else 'mc'
    path = workspace_path / f'Output/{channel}/NN/plots'
    
    path.mkdir(parents=True, exist_ok=True)
    
    R.gStyle.SetOptStat(0)
    
    
    # MVA evaluation
    
    with uproot.open(workspace_path / f'Output/{channel}/NN/{cl}_{channel}_fold1.root') as f:
        data = f['per_epoch'].arrays(library='np')

        train_loss = data['train_loss']
        test_loss = data['test_loss']
        
        train_auc = data['train_auc']
        test_auc = data['test_auc']
        
        train_data = f['train'].arrays(library='np')
        test_data = f['test'].arrays(library='np')
    
    nepochs = np.arange(1, len(train_loss) + 1)
    
    # evaluate the loss
    
    plot(
        nepochs,
        train_loss,
        test_loss,
        name=f'{cl}_{channel}_loss',
        prefix=prefix,
        yscale='log',
        path=path
    )
    
    # evaluate the AUC
    
    plot(
        nepochs,
        train_auc,
        test_auc,
        name=f'{cl}_{channel}_auc',
        prefix=prefix,
        path=path
    )
    
    
    # plot significances
    
    thresholds = [0.5, 0.6, 0.7, 0.8, 0.9, 0.99]
    fig, ax = plot_significances(
        test_data['DNN'],
        test_data['target'],
        test_data['weights'],
        thresholds,
        name=f'{cl}_{channel}_significances',
        prefix=prefix,
        path=path
    )
    
    plot_significances(
        train_data['DNN'],
        train_data['target'],
        train_data['weights'],
        thresholds,
        legend=['Test sample', 'Train sample'],
        name=f'{cl}_{channel}_significances',
        prefix=prefix,
        path=path,
        fig=fig,
        ax=ax
    )
    
    
    # Overfitting check
    
    overfitting_test(
        train_data,
        test_data,
        channel,
        path,
        prefix=prefix
    )
    
    
    # ROC curve
    
    plot_roc(
        test_data['DNN'],
        test_data['target'],
        test_data['weights'],
        channel,
        mva,
        path,
        prefix=prefix
    )
    
    
if __name__=="__main__":
    parser = argparse.ArgumentParser(
        description='Plotting the results of the neural network'
    )
    
    parser.add_argument(
        '-c', '--channel', type=str, default='1l0tau', help='Name of the channel. Default: 1l0tau'
    )
    parser.add_argument(
        '-m', '--mva', type=str, default='DNN', help='Name of the MVA. Default: DNN'
    )
    parser.add_argument(
        '-cl', '--classification', type=str, default='binary', choices=['binary', 'multiclass'], help='Type of classification. Default: binary'
    )
    parser.add_argument(
        '--prefix', type=str, default='pdf', help='Prefix for the output file. Default: .pdf'
    )
    
    args = parser.parse_args()
    
    workspace_path = Path(__file__).resolve().parents[2]
    
    main(
        args.channel, 
        args.classification,
        args.mva,
        args.prefix,
    )