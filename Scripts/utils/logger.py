import numpy as np
import torch
from pathlib import Path
import uproot
from ray.tune.logger import LoggerCallback

from .metrics import auc_score

class NNLogger:
    def __init__(
        self, 
        channel: str, 
        features: list[str],
        path: str|Path,
        classification: str = 'binary'
    ):
        
        self.channel = channel
        self.features = features
        self.path = path if isinstance(path, Path) else Path(path)
        self.classification = classification
        
        self._results: dict[str, np.ndarray] = {}
        self._params: dict[str, np.ndarray] = {}
        self.last_results: dict[str, float] = {}
        
        self.path.mkdir(parents=True, exist_ok=True)
        
    
    def on_start(
        self, 
        input_features: dict[str, torch.Tensor],
        targets: dict[str, torch.Tensor],
        weights: dict[str, torch.Tensor],
        fold: int = None
    ):
        # Initialize the logger without cross-validation
        if fold is None:
            self._params = {
                'input_features': input_features,
                'targets': targets,
                'weights': weights,
            }
            
            self._results = {
                'train_loss': np.empty(0),
                'train_auc': np.empty(0),
                'test_loss': np.empty(0),
                'test_auc': np.empty(0),
            }
            
        # Initialize the logger with cross-validation
        else:
            self._params[fold] = {
                'input_features': input_features,
                'targets': targets,
                'weights': weights,
            }
            
            self._results[fold] = {
                'train_loss': np.empty(0),
                'train_auc': np.empty(0),
                'test_loss': np.empty(0),
                'test_auc': np.empty(0),
            }
    
        
    def log(
        self,
        results: dict[str, tuple[float, torch.Tensor, torch.Tensor]],
        fold: int = None
    ):
        train_results = results['train']
        test_results = results['test']
        
        results = self._results if fold is None else self._results[fold]
        params = self._params if fold is None else self._params[fold]
        
        w_train = params['weights']['train']
        w_test = params['weights']['test']
        
        y_train = params['targets']['train']
        y_test = params['targets']['test']
        
        with torch.no_grad():
            results['train_loss'] = np.append(results['train_loss'], train_results[0])
            results['train_auc'] = np.append(results['train_auc'], auc_score(train_results[1], y_train, weight=w_train, pos_label=1))
            results['test_loss'] = np.append(results['test_loss'], test_results[0])
            results['test_auc'] = np.append(results['test_auc'], auc_score(test_results[1], y_test, weight=w_test, pos_label=1))
            
            results['train_pred'] = train_results[1]
            results['test_pred'] = test_results[1]
            
            if fold is None:
                self.last_results = {
                    'train_loss': results['train_loss'][-1].item(),
                    'train_auc': results['train_auc'][-1].item(),
                    'test_loss': results['test_loss'][-1].item(),
                    'test_auc': results['test_auc'][-1].item(),
                }
            else:
                self.last_results[fold] = {
                    'train_loss': results['train_loss'][-1].item(),
                    'train_auc': results['train_auc'][-1].item(),
                    'test_loss': results['test_loss'][-1].item(),
                    'test_auc': results['test_auc'][-1].item(),
                }
                
            
    def on_complete(self, cv: bool = False):
        cl = 'bc' if self.classification == 'binary' else 'mc'
        
        if not cv:
            self.write_to_root(f'DNN_{cl}_{self.channel}.root')
        else:
            for fold in range(len(self._results)):
                self.write_to_root(f'DNN_{cl}_{self.channel}_fold{fold+1}.root', fold)
                
        
    def write_to_root(self, file_name: str, fold: int = None):
        results = self._results if fold is None else self._results[fold]
        params = self._params if fold is None else self._params[fold]
        
        fold_n = '' if fold is None else f'_fold{fold+1}'
        
        # format results into dictionary   
        results_fmt = {
            'train': {
                'loss': results['train_loss'][-1],
                'auc': results['train_auc'][-1],
                'pred': results['train_pred'],
                'target': params['targets']['train'],
                'weights': params['weights']['train'],
                'features': params['input_features']['train'],
            },
            
            'test': {
                'loss': results['test_loss'][-1],
                'auc': results['test_auc'][-1],
                'pred': results['test_pred'],
                'target': params['targets']['test'],
                'weights': params['weights']['test'],
                'features': params['input_features']['test'],
            }
        }
        
        if self.classification == 'binary':
            results_fmt['train']['pred'] = results['train_pred']
            results_fmt['test']['pred'] = results['test_pred']
        else:
            results_fmt['train'].update({
                'signal': results['train_pred'][:, 0],
                'SH': results['train_pred'][:, 1],
                'CB': results['train_pred'][:, 2],
            })
            results_fmt['test'].update({
                'signal': results['test_pred'][:, 0],
                'SH': results['test_pred'][:, 1],
                'CB': results['test_pred'][:, 2],
            })
        
        # Save results to file
        with uproot.recreate(self.path / file_name) as f:
            f['per_epoch'] = {
                'train_loss': results['train_loss'],
                'train_auc': results['train_auc'],
                'test_loss': results['test_loss'],
                'test_auc': results['test_auc'],
            }
            
            for name, value in zip(['TrainTree', 'TestTree'], results_fmt.values()):
                if self.classification == 'binary':
                    output = {
                        f'DNN{fold_n}': value['pred'].numpy(),
                        # Swap class labels for more consisentsy between the methods
                        'classID': np.where(value['target'] == 0, 1, 0),
                        'weight': value['weights'].numpy(),
                    }
                else:
                    output = {
                        f'DNN_signal{fold_n}': value['signal'].numpy(),
                        f'DNN_SH{fold_n}': value['SH'].numpy(),
                        f'DNN_CB{fold_n}': value['CB'].numpy(),
                        'classID': value['target'].numpy(),
                        'weight': value['weights'].numpy(),
                    }
                
                for i, feature in enumerate(self.features):
                    output[feature] = value['features'][:, i]
                    
                f[name] = output
                
    def get_last_results(self, fold: int = None):
        last_results = self.last_results if fold is None else self.last_results[fold]
        
        return last_results
    
class CustomLogger(LoggerCallback):
    def __init__(self, kfold: int = 0, path: str|Path = 'Output'):
        self.kfold = kfold
        self.path = self.path if isinstance(self.path, Path) else Path(self.path)
        
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