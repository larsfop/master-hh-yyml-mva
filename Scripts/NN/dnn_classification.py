import torch
import torch.nn as nn
from pathlib import Path
import argparse
import multiprocessing as mp
import sys
import yaml

sys.path.append(str(Path(__file__).resolve().parents[1]))

from neural_network import NeuralNetwork, EarlyStopping
from utils.model_dict import set_model_structure
from utils.hh_dataset import HH_Dataset_hdf5
from utils.timer import Timer
from utils.logger import NNLogger


def main(
    channel: str,
    path: Path|str,
    epochs: int,
    balanced_weights: bool,
    generator_weights: bool,
    pbar: bool,
    early_stopping: bool,
    classification: str = 'binary',
    kfold: int = 0,
    njobs: int = 1,
):
    
    mp.set_start_method('spawn')
    
    path = path if isinstance(path, Path) else Path(path)

    # Use cuda if available, else cpu
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    loss_functions = {
        'binary_crossentropy': nn.BCEWithLogitsLoss,
        'crossentropy': nn.CrossEntropyLoss,
    }
    
    optimizers = {
        'adam': torch.optim.Adam,
        'sgd': torch.optim.SGD,
    }
    
    cl = 'mc' if classification == 'multiclass' else 'bc'
    with open(workspace_path / f'Configs/NN/{cl}_{channel}_config.yaml', 'r') as f:
        config = yaml.safe_load(f)
        
    model = config['model']
    # Setup the model architecture
    model_dict = set_model_structure(
        model['input_shape'], # Input nodes
        model['output_shape'], # Output nodes
        *model['layers'], # Hidden layers
    )
    
    # Load the data for the given features
    training = config['training']
    
    features = training['input_features']
    with HH_Dataset_hdf5(workspace_path / path / f'{channel}.hdf5') as hf:
        X = hf.tensors(features=features, device=device, dtype=torch.float32)
        y = hf.labels(device=device, multiclass=classification == 'multiclass')
        mc_weights = hf.tensors('weight', device=device, dtype=torch.float32)
    
    if generator_weights:
        # Remove events with weights less than 0
        X = X[mc_weights > 0]
        y = y[mc_weights > 0]
        mc_weights = mc_weights[mc_weights > 0]
    
    # Setup early stopping
    if early_stopping:
        print('Using early stopping')
        es_config = config['EarlyStopping']
        early_stopping = EarlyStopping(
            patience=es_config['patience'],
            rise_delta=float(es_config['rise_delta']),
            threshold_delta=float(es_config['threshold_delta'])
        )
    else:
        early_stopping = None
    
    with Timer() as t:
        net = NeuralNetwork(
            model_dict=model_dict,
            optimizer=optimizers[training['optimizer']],
            loss_function=loss_functions[training['loss_function']], 
            logger=NNLogger,
            epochs=epochs,
            batch_size=training['batch_size'],
            lr=float(training['learning_rate']),
            lmbda=float(training['regularization']),
            balanced_weights=balanced_weights,
            generator_weights=generator_weights,
            device=device, 
            rng=125,
            pbar=pbar,
            early_stopping=early_stopping,
            classification=classification,
            cv=kfold>1,
            output_path=workspace_path / f'Output/{channel}/NN',
            channel=channel,
            features=features,
        )

        net.fit(
            X, 
            y,
            mc_weights,
            norm_mode='balanced',
            test_size=0.2,
            nfolds=kfold,
            njobs=njobs,
        )
        
        if kfold < 2:
            net.save_model(workspace_path / f'Output/{channel}/NN/weights')


if __name__=="__main__":
    parser = argparse.ArgumentParser(
        description='Neural network for training on di-Higgs channels.'
    )
    
    parser.add_argument(
        '-c', '--channel', type=str, default='1l0tau', help='Name of the channel you want to train on, default: <1l0tau>'
    )
    parser.add_argument(
        '-p', '--path', type=Path, default=Path('Input_Files'), help='Path to the hdf5 file containing the data for the chosen channel, default: </Data>'
    )
    parser.add_argument(
        '-e', '--epochs', type=int, default=100, help='Number of training epochs'
    )
    parser.add_argument(
        '-bw', '--balanced_weights', action='store_false', help='Use balanced weights for the training, default: True'
    )
    parser.add_argument(
        '-gw', '--generator_weights', action='store_false', help='Use generator weights for the training, default: True'
    )
    parser.add_argument(
        '-pb', '--pbar', action='store_false', help='Show progress bar during training'
    )
    parser.add_argument(
        '-es', '--early_stopping', action='store_false', help='Use early stopping, default: True, Not applicable for k-fold cross-validation'
    )
    parser.add_argument(
        '-k', '--kfold', type=int, default=0, help='Number of folds for k-fold cross-validation, default: 0 (no k-fold cross-validation)'
    )
    parser.add_argument(
        '-cl', '--classification', type=str, default='binary', help='Type of classification, default: <binary>'
    )
    parser.add_argument(
        '-j', '--njobs', type=int, default=1, help='Number of jobs for multiprocessing, default: 1'
    )
    
    args = parser.parse_args()
    
    workspace_path = Path(__file__).resolve().parents[2]
    
    print(f'Processing channel: {args.channel}')
    
    main(
        args.channel,
        args.path,
        args.epochs,
        args.balanced_weights,
        args.generator_weights,
        args.pbar,
        args.early_stopping,
        args.classification,
        args.kfold,
        args.njobs,
    )