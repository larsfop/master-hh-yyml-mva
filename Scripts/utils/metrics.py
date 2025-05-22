
import numpy as np
import torch
import sklearn.metrics as metrics
from sklearn.preprocessing import OneHotEncoder
import scipy.integrate as integrate


def auc_score(y_pred: torch.Tensor, y_true: torch.Tensor, weight: torch.Tensor|None = None, pos_label: int = 0) -> float:
    _, auc = roc_auc(
        y_pred, 
        y_true, 
        weight,
        pos_label=pos_label
    )
            
    return auc


def roc_auc(y_pred: np.ndarray, y_true: np.ndarray, weight:np.ndarray, pos_label: int = 0) -> tuple[tuple[np.ndarray, np.ndarray, np.ndarray], float]:
    auc = 0.0    
    if len(y_pred.shape) == 1:
        roc = metrics.roc_curve(y_true, y_pred, sample_weight=weight, pos_label=pos_label)
        auc += integrate.trapezoid(x=roc[0], y=roc[1])
    else:
        if y_pred.shape != y_true.shape:
            oh_encoder = OneHotEncoder()
            
            y_true = oh_encoder.fit_transform(y_true.reshape(-1, 1)).toarray()
        
        for i in range(y_pred.shape[1]):
            roc = metrics.roc_curve(y_true[:, i], y_pred[:, i], sample_weight=weight)
            auc += integrate.trapezoid(x=roc[0], y=roc[1])
            
        auc /= y_pred.shape[1]
        
    return roc, auc


def Z_score(s, b):
    z = np.sqrt(2 * ((s + b) * np.log(1 + s/(b + 1e-6)) - s))
    
    z = np.nan_to_num(z, nan=0.0)
    z = 0 if z < 0 else z
    
    return z