
from ray.train._internal.storage import StorageContext
from ray.tune.experiment import Trial
from ray.tune.logger import Logger
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import random_split, TensorDataset, DataLoader, SubsetRandomSampler
from torcheval.metrics import BinaryAccuracy

from sklearn import metrics
from sklearn.utils.class_weight import compute_class_weight
from sklearn.model_selection import train_test_split, StratifiedKFold

from pathlib import Path
import argparse
import uproot
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import time
import sys
import os
import json
import warnings
from typing_extensions import Callable, Dict, Any
import sys

sys.path.append(str(Path(__file__).resolve().parents[1]))


# Set environment variables for Ray Tune
os.environ['TUNE_MAX_PENDING_TRIALS_PG'] = '10'
os.environ['TUNE_DISABLE_AUTO_CALLBACK_LOGGERS'] = '1'
os.environ['TUNE_DISABLE_STRICT_METRIC_CHECKING'] = '1'
os.environ['TUNE_RESULTS_DIR'] = './Hyperparameter_Optimization'

try:
    os.environ['RAY_BACKEND_LOG_LEVEL'] = 'fatal'
except:
    pass

import ray
from ray import train, tune
from ray.train import Checkpoint, get_checkpoint, RunConfig
from ray.tune.schedulers import ASHAScheduler
from ray.tune.search.bayesopt import BayesOptSearch
from ray.tune.logger import LoggerCallback
import ray.cloudpickle as pickle

from utils.hh_dataset import HH_Dataset_hdf5
from utils.model_dict import set_model_structure
from neural_network import NeuralNetwork, Model, EarlyStopping
from utils.metrics import auc_score



class CustomLogger(LoggerCallback):
    def __init__(self, channel: str, classification: str, kfold: int = 0, grid_search: str = ''):
        self.channel = channel
        self.kfold = kfold
        cl = 'bc' if classification == 'binary' else 'mc'
        self.path = workspace_path / Path(f'Output/{channel}/NN/Hyperparameter_Optimisation/{cl}_{channel}_{grid_search}.root')
        
        self.path.parent.mkdir(parents=True, exist_ok=True)

        self.results = {}
        self.config = {}
        
    def on_trial_start(self, trial, **info):
        pass
    
        
    def on_trial_result(self, iteration, trial, result, **info):
        pass
        
        
    def on_trial_complete(self, iteration, trials, trial, **info):
        self.config[trial.trial_id] = trial.config
        
    
    def on_experiment_end(self, trials, **info):
        # with uproot.open(self.path) as oldfile:
        print(
            '------------------------------------------\n'\
            '   Hyperparameter optimization complete\n'\
            f'  Writing results to {self.path.stem}\n'\
            '------------------------------------------'
        )
        
        n = len(trials)
        
        params = {
            'eta': np.zeros(n),
            'lambda': np.zeros(n),
            'batch_size': np.zeros(n),
            'num_hidden_layers': np.zeros(n),
            'num_hidden_neurons': np.zeros(n),
        }
        if not self.kfold:
            params.update({
                'train_loss': np.zeros(n),
                'train_auc': np.zeros(n),
                'test_loss': np.zeros(n),
                'test_auc': np.zeros(n),
            })
        else:
            params.update({
                f'{name}_fold_{i}': np.zeros(n) for i in range(1, self.kfold + 1) for name in ['train_loss', 'train_auc', 'test_loss', 'test_auc']
            })
            params.update({
                'train_loss_mean': np.zeros(n),
                'train_auc_mean': np.zeros(n),
                'test_loss_mean': np.zeros(n),
                'test_auc_mean': np.zeros(n),
            })
            
        for i, trial in enumerate(trials):
            result = trial.last_result
            config = trial.config
            
            params['eta'][i] = config['lr']
            params['lambda'][i] = config['lmbda']
            params['batch_size'][i] = config['batch_size']
            params['num_hidden_layers'][i] = len(config['hidden_neurons'])
            params['num_hidden_neurons'][i] = config['hidden_neurons'][0]
            
            if not self.kfold:
                params['train_loss'][i] = result['train_loss']
                params['train_auc'][i] = result['train_auc']
                params['test_loss'][i] = result['test_loss']
                params['test_auc'][i] = result['test_auc']
                
            else:
                for fold in range(1, self.kfold + 1):
                    params[f'train_loss_fold_{fold}'][i] = result[f'fold_{fold}']['train_loss']
                    params[f'train_auc_fold_{fold}'][i] = result[f'fold_{fold}']['train_auc']
                    params[f'test_loss_fold_{fold}'][i] = result[f'fold_{fold}']['test_loss']
                    params[f'test_auc_fold_{fold}'][i] = result[f'fold_{fold}']['test_auc']
                    
                params['train_loss_mean'][i] = result['train_loss']
                params['train_auc_mean'][i] = result['train_auc']
                params['test_loss_mean'][i] = result['test_loss']
                params['test_auc_mean'][i] = result['test_auc']
                    
        with uproot.recreate(self.path) as file:
            
            file['Hyperparameter_Optimisation'] = params
                
    
class DNN_Tuner(tune.Trainable, NeuralNetwork):
    def setup(
            self, 
            config: dict, 
            channel: str,
            features: list, 
            device: str,
            classification: str
    ):
        
        for key, value in config.items():
            if not isinstance(value, Callable):
                setattr(self, key, value)

        # self.background_weight = 1/self.signal_weight
        
        # change the config to list the neurons for logging
        self.config['num_hidden_layers'] = len(self.hidden_neurons)
        self.config['num_features'] = len(self.features)
        
        features_indices = {
            features[i]: i for i in range(len(features))
        }

        indices = [features_indices[feature] for feature in self.features]
    
        output_neurons = 1 if classification == 'binary' else 3
        md = set_model_structure(
            len(self.features),
            *[{'neurons': neurons, 'activation': 'relu'} for neurons in self.hidden_neurons], output_neurons
        )
        
        self.device = device
        self.classification = classification
        self.activation_output = nn.Sigmoid() if classification == 'binary' else nn.Softmax(dim=1)
        
        with HH_Dataset_hdf5(workspace_path / f'Input_Files/{channel}.hdf5') as hf:
            X = hf.tensors(
                features=self.features,
                dtype=torch.float32,
                device=self.device
            )
            y = hf.labels(
                multiclass=classification == 'multiclass',
                device=self.device
            )
            mc_weights = hf.tensors(
                features='weight',
                dtype=torch.float32,
                device=self.device
            )
        
        # Remove events with negative weights
        mask = mc_weights > 0
        X = X[mask]
        y = y[mask]
        mc_weights = mc_weights[mask]
        
        if not self.kfold:
            self.prepare_NN(X, y, mc_weights, indices, md)
        else:
            self.prepare_NN_cv(X, y, mc_weights, indices, md)

    def step(self):
        if not self.kfold:
            train_loss, train_pred, train_y, train_w = self._train(self.train_dataloader, self.model, self.optimizer, self.loss_function)

            test_loss, test_pred, test_y, test_w = self._test(self.test_dataloader, self.model, self.loss_function)
            
            results = {
                'train_loss': train_loss,
                'train_auc': auc_score(train_pred, train_y, weight=train_w),
                'test_loss': test_loss,
                'test_auc': auc_score(test_pred, test_y, weight=test_w),
            }
            
        else:
            results = {
                'train_loss': 0,
                'train_auc': 0,
                'test_loss': 0, 
                'test_auc': 0
            }
            for fold in range(1, self.kfold + 1):
                train_loss, train_pred, train_y, train_w = self._train(self.cv_dataloaders[fold]['train'], self.cv_models[fold], self.cv_optimisers[fold], self.cv_loss_functions[fold])
                
                test_loss, test_pred, test_y, test_w = self._test(self.cv_dataloaders[fold]['test'], self.cv_models[fold], self.cv_loss_functions[fold])
                
                train_auc = auc_score(train_pred, train_y, weight=train_w)
                test_auc = auc_score(test_pred, test_y, weight=test_w)
                
                results[f'fold_{fold}'] = {
                    'train_loss': train_loss,
                    'train_auc': train_auc,
                    'test_loss': test_loss,
                    'test_auc': test_auc,
                }
            
                results['train_loss'] += train_loss
                results['train_auc'] += train_auc
                results['test_loss'] += test_loss
                results['test_auc'] += test_auc
                
            results['train_loss'] /= self.kfold
            results['train_auc'] /= self.kfold
            results['test_loss'] /= self.kfold
            results['test_auc'] /= self.kfold

        return results
    
    
    def prepare_NN(self, X, y, w, indices, md):

            # Split the data into training and test sets
            X_train, X_test, y_train, y_test, w_train, w_test = train_test_split(
                X, 
                y, 
                w, 
                test_size=0.2, 
            )
            
            # Normalise the sum of weights
            w_norm = w_train[y_train == 0].sum() / w_train[y_train == 1].sum()
            w_train[y_train == 1] *= w_norm
            w_test[y_test == 1] *= w_norm
            
            # Setup the dataloaders
            train_ds = TensorDataset(X_train[:, indices], y_train, w_train)
            test_ds = TensorDataset(X_test[:, indices], y_test, w_test)
            
            self.train_dataloader = DataLoader(train_ds, self.batch_size)
            self.test_dataloader = DataLoader(test_ds, self.batch_size)

            self.model = Model(md)
            self.model.to(self.device)
        
            if self.classification == 'binary':
                if self.class_weights == 'balanced':
                    class_weights = (y[train_ds.indices] == 0).sum() / y[train_ds.indices].sum()
                elif self.class_weights == 'none':
                    class_weights = None
                else:
                    class_weights = torch.tensor(self.class_weights)
            else:
                if self.class_weights == 'balanced':
                    class_weights = compute_class_weight(
                        'balanced',
                        classes=np.unique(y[train_ds.indices].cpu().numpy()),
                        y=y[train_ds.indices].cpu().numpy()
                    )
                    class_weights = torch.tensor(class_weights, dtype=torch.float32, device=device)
                elif self.class_weights == 'none':
                    class_weights = None
                else:
                    class_weights = torch.tensor((self.class_weights, 1, 1), dtype=torch.float32, device=device)

            if self.classification == 'binary':
                self.loss_function = nn.BCEWithLogitsLoss(
                    reduction='none',
                    pos_weight=class_weights
                    )
            else:
                self.loss_function = nn.CrossEntropyLoss(
                    reduction='none',
                    weight=class_weights
                )
            
            self.optimizer = optim.Adam(
                self.model.parameters(),
                lr=self.lr,
                weight_decay=self.lmbda
            )

            self.early_stopping = EarlyStopping(
                self.patience,
                self.delta
            )
            
            
    def prepare_NN_cv(self, X, y, w, indices, md):
        
        self.cv_models = {}
        self.cv_optimisers = {}
        self.cv_loss_functions = {}
        self.cv_dataloaders = {}
        self.cv_indices = {}
        
        skf = StratifiedKFold(n_splits=self.kfold, shuffle=True)
        
        for fold, (train_idx, test_idx) in enumerate(skf.split(X.cpu(), y.cpu()), 1):
            self.cv_indices[fold] = {
                'train': train_idx,
                'test': test_idx
            }
            
            w_new = w.clone()
            
            # if self.norm_mode == 'balanced':
            if self.classification == 'binary':
                w_norm = w[train_idx][y[train_idx] == 0].sum() / w[train_idx][y[train_idx] == 1].sum()
                w_new[y == 1] *= w_norm
                
                print((w_new[y == 1].sum(), w_new[y == 0].sum()))
                
            else:
                w_sum = w[train_idx].sum()
                for i in range(3):
                    w_norm = w_sum / w[train_idx][y[train_idx] == i].sum()
                    w_new[y == i] *= w_norm
                    
                print(w_new[y == 0].sum(), w_new[y == 1].sum(), w_new[y == 2].sum())
            
            ds = TensorDataset(X[:, indices], y, w_new)
            
            self.cv_dataloaders[fold] = {
                'train': DataLoader(ds, self.batch_size, sampler=SubsetRandomSampler(train_idx)),
                'test': DataLoader(ds, self.batch_size, sampler=SubsetRandomSampler(test_idx))
            }
            
            self.cv_models[fold] = Model(md)
            self.cv_models[fold].to(self.device)
            
            if self.classification == 'binary':
                self.cv_loss_functions[fold] = nn.BCEWithLogitsLoss(
                    reduction='none',
                    pos_weight=None
                )
            else:
                self.cv_loss_functions[fold] = nn.CrossEntropyLoss(
                    reduction='none',
                    weight=None
                )
                
            self.cv_optimisers[fold] = optim.Adam(
                self.cv_models[fold].parameters(),
                lr=self.lr,
                weight_decay=self.lmbda
            )
            
        
           
if __name__ == "__main__":
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    parser = argparse.ArgumentParser(
        description='Optimize hyperparameters'
    )
    
    parser.add_argument(
        '-c', '--channel', type=str, default='1l0tau', help='Channel to train on, default: 1l0tau'
    )
    parser.add_argument(
        '-p', '--params', type=Path, default=None, help='Filepath for the hyperparameters to optimize, with None it will use default parameter grid. Default: None'
    )
    parser.add_argument(
        '-e', '--epochs', type=int, default=100, help='Number of epochs for each training trial, default: 100'
    )
    parser.add_argument(
        '-t', '--trials', type=int, default=1, help='Number of trials to use for the optimization, default: 10'
    )
    parser.add_argument(
        '-k', '--kfold', type=int, default=0, help='Number of folds for the k-fold cross-validation, default: 0'
    )
    parser.add_argument(
        '-g', '--grid_search', type=str, default='', help='Grid search for hyperparameters'
    )
    parser.add_argument(
        '-r', '--random_search', type=str, default='', help='Random search for hyperparameters'
    )
    parser.add_argument(
        '-n', '--n_jobs', type=int, default=10, help='Number of processes to run on the gpu in parallell, default: 1'
    )
    parser.add_argument(
        '-cl', '--classification', type=str, default='binary', help='Classification type, default: binary'
    )
    parser.add_argument(
        '-s', '--scheduler', action='store_true', help='Use ASHA scheduler for hyperparameter optimization'
    )
    parser.add_argument(
        '-v', '--verbose', action='store_false', help='Print verbose output'
    )

    args = parser.parse_args()

    if args.verbose:
        ray.init(log_to_driver=False)

    channel = args.channel
    
    workspace_path = Path(__file__).resolve().parents[2]
    
    """if args.feature_search:
        features = [
            # 'm_H',
            # 'E_H',
            # 'phi_H',
            # 'pt_H',
            # 'eta_H',
            # 'Dr_H',
            'lep_E_1',
            # 'lep_E_2',
            'lep_eta_1',
            # 'lep_eta_2',
            'lep_m_1',
            # 'lep_m_2',
            'lep_phi_1',
            # 'lep_phi_2',
            'lep_phi0_1',
            # 'lep_phi0_2',
            'lep_pt_1',
            # 'lep_pt_2',
            # 'lep_q_1',
            # 'lep_q_2',
            'met_phi',
            'met_phi0',
            'met',
            'sumet',
            # 'mu',
            'N_j_central',
            # 'N_j',
            # 'HT',
            'y1_E',
            'y1_eta',
            'y1_phi',
            'y1_phi0',
            'y1_pt',
            'y2_E',
            'y2_eta',
            'y2_phi',
            'y2_phi0',
            'y2_pt',
            'pt_ll',
            'eta_ll',
            'phi_ll',
            'm_ll',
            # 'Dr_ll',
            'Dphi_ll',
            'Dr_yyll',
            'Dphi_yyll',
            'pt_yyll',
            'm_yyll',
            'pt_lv',
            'Dr_lv',
            'pt_llv',
            'pt_W',
            'eta_W',
            'phi_W',
            # 'pt_W2',
            'eta_W2',
            # 'phi_W2',
            'm_W1',
            # 'm_W2',
            'Dr_yyW',
            'MT_W1',
            # 'MT_W2',
            'MT',
            'Dr_yyl1',
            'Dphi_metll',
            'Dphi_metyy',
            'minDphi_metjl',
            # 'minDR_jl1',
            # 'minDR_jl2',
            # 'N_j_removal',
            # 'N_jrec',
            # 'N_Lrec',
            'mbig',
            'ptbig',
            'Dy_bigyy',
            'mbig2',
            'ptbig2',
            'Dy_bigyy2',
            # 'N_Cluster',
            # 'rfr0',
            # 'rfr1',
            # 'msum',
            # 'msum2',
            # 'ptsum',
            # 'ptsum2',
            # 'Jet_pt1',
            # 'Jet_pt2',
            # 'Dr_jj',
            # 'Dr_lj',
            # 'etsum',
            # 'etsumtot',
            # 'etsumleft',
        ]
        
    else:"""
    features = [
        'pt_H',
        'lep_phi0_1',
        'met', 
        'lep_pt_1',
        'N_j_central',
        'Dphi_metyy'
    ]
    
    if channel == '1l0tau':
        features += [
            'y1_phi0',
            'minDphi_metjl',
            'Dr_lv', 
            'Dr_yyW', 
            'eta_W',
        ]
    
    elif channel == '0l1tau':
        features += [
            'y1_phi0',
            'y1_eta',
        ]
    
    else:
        features += [
            'phi_H', 
            'lep_pt_2', 
            'met_phi',
            'minDphi_metjl',
            'Dphi_metll',
            'Dr_lv', 
            'm_ll',
            'Dr_ll',
            'Dphi_ll',
            'Dr_yyll',
            'Jet_pt1',
        ]
    
    # setup parameter grid for optimization
    # Use set values to change base parameters for the network
    rng_uniform = torch.distributions.uniform.Uniform
    
    config = {
        'lr': 1e-5,
        'lmbda': 1e-5,
        'batch_size': 10000,
        'hidden_neurons': [1024, 1024, 1024, 1024],
        'features': features,
        'patience': 10,
        'delta': 1e-5,
        'class_weights': 'none',
    }
    
    config['kfold'] = args.kfold
    
    for var in args.grid_search:
        match var:
            case 'w':
                config['class_weights'] = tune.grid_search(['none', 'balanced', *np.linspace(0.1, 10, 18)])
            case 'l':
                config['lr'] = tune.grid_search(np.logspace(-1, -6, 6))
            case 'r':
                config['lmbda'] = tune.grid_search(np.logspace(-3, -7, 5))
            case 'b':
                config['batch_size'] = tune.grid_search([2**i for i in range(10, 18)])
            case 'h':
                config['hidden_neurons'] = tune.grid_search([
                    [neurons]*i for i in range(1, 6) for neurons in np.arange(128, 2050, 128)
                ])
                
    for var in args.random_search:
        match var:
            case 'w':
                config['class_weights'] = tune.uniform(0.1, 10)
            case 'l':
                config['lr'] = tune.loguniform(1e-3, 1e-6)
            case 'r':
                config['lmbda'] = tune.loguniform(1e-4, 1e-6)
            case 'b':
                config['batch_size'] = tune.choice([2**i for i in range(10, 18)])
            case 'h':
                config['hidden_neurons'] = tune.sample_from(lambda: np.random.randint(128, 208, np.random.randint(1, 6)))
        
    
    #---------------------------------------------------------------------
    #                         NOT IMPLEMENTED
    if args.params is not None:
        path = Path(args.params)
        config = open(file=path)
    #---------------------------------------------------------------------
    
    scheduler = ASHAScheduler(
            max_t=args.epochs,
            grace_period=5,
            reduction_factor=2
        )
    
    logger = CustomLogger(channel, args.classification, args.kfold, args.grid_search)

    tuner = tune.Tuner(
        # trainable class used by framework
        trainable=tune.with_resources(
            trainable=tune.with_parameters(
                DNN_Tuner, 
                channel=channel,
                features=features,
                device=device,
                classification=args.classification
            ),
            resources={'cpu': 1, 'gpu': 1/args.n_jobs}
        ),
        tune_config=tune.TuneConfig(
                metric='test_loss', # Metric to optimize
                mode='min',
                scheduler=scheduler if args.scheduler else None, # Use ASHA scheduler
                # search_alg=BayesOptSearch(random_search_steps=trials),
                num_samples=args.trials, # Number of trials to run
        ),
        run_config=train.RunConfig(
            stop={"training_iteration": args.epochs}, # Number of epochs for each trial
            checkpoint_config=train.CheckpointConfig(checkpoint_at_end=False), # Remove checkpointing
            callbacks=[logger], # Add custom logger
        ),
        param_space=config,
    )

    results = tuner.fit()
    
    print(
        '------------------------------------------\n'\
        '   Hyperparameter optimization complete\n'\
        '------------------------------------------'\
    )
    