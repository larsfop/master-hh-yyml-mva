import numpy as np
import torch
from pathlib import Path
import h5py


class HH_Dataset_hdf5:
    def __init__(self, file: Path|str) -> None:
        self.file = file if isinstance(file, Path) else Path(file)
        self.channel = self.file.stem

            
    def __enter__(self):
        self.open(self.file)
        
        return self
    
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
        
        
    def tensors(
        self,
        features: list[str]|str|None = None,
        processes: list[str]|None = ['signal_ggF', 'signal_VBF', 'SH', 'Sherpa', 'Vyy'],
        dtype = torch.float64,
        device = 'cpu',
    ):
        
        X = []
        if isinstance(features, str):
            for process in processes:
                process = process + f'_{self.channel}'
                
                X.append(torch.tensor(self.hfile[process][features], dtype=dtype, device=device))
            
        else:
            for process in processes:
                process = process + f'_{self.channel}'
                
                X_new = []
                for feature in features:
                    X_new.append(torch.tensor(self.hfile[process][feature], dtype=dtype, device=device).reshape(-1, 1))
                    
                X.append(torch.cat(X_new, dim=1))
                
        return torch.cat(X, dim=0)
        
        
    def labels(
        self,
        multiclass: bool = False,
        onehot: bool = False,
        processes: list[str] = ['signal_ggF', 'signal_VBF', 'SH', 'Sherpa', 'Vyy'],
        dtype: torch.dtype = torch.int64, 
        device: str = 'cpu',
    ) -> torch.Tensor:
        
        y = torch.empty(0, dtype=torch.int64, device=device)
        
        for process in processes:
            process = process + f'_{self.channel}'
            
            if multiclass:
                if 'signal' in process:
                    y = torch.concatenate((y, torch.zeros(self.grp_sizes[process], dtype=dtype, device=device)))
                elif 'SH' in process:
                    y = torch.concatenate((y, torch.ones(self.grp_sizes[process], dtype=dtype, device=device)))
                else:
                    y = torch.concatenate((y, torch.full((self.grp_sizes[process],), 2 , dtype=dtype, device=device)))
            else:
                if 'signal' in process:
                    y = torch.concatenate((y, torch.ones(self.grp_sizes[process], dtype=dtype, device=device)))
                else:
                    y = torch.concatenate((y, torch.zeros(self.grp_sizes[process], dtype=dtype, device=device)))
                
        if onehot:
            y = torch.nn.functional.one_hot(y, num_classes=3)
            
        return y
            
        
    def open(self, file):
        self.hfile = h5py.File(file, 'r')
        
        self.grp_sizes = {}
        for key in self.hfile.keys():
            self.grp_sizes[key] = self.hfile[key].attrs['size']
            
    
    def close(self):
        self.hfile.close()    