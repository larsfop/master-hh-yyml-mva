import torch.multiprocessing as mp
mp.set_sharing_strategy('file_system')
import torch.nn as nn
import torch
from torch.utils.data import TensorDataset, DataLoader
from numpy import ndarray
import numpy as np
from tqdm import tqdm
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.utils.class_weight import compute_class_weight
import sys
import time
from typing import Callable, OrderedDict
from copy import deepcopy
# import multiprocessing as mp
from multiprocessing.managers import BaseManager
import enlighten
from pathlib import Path
import sys
import traceback

sys.path.append(str(Path(__file__).resolve().parents[1]))


from utils.model_dict import set_model_structure

class EarlyStopping:
    def __init__(
            self,
            patience: int = 1,
            rise_delta: float = 0,
            threshold_delta: float = 0,
    ) -> None:
        
        self.patience = patience
        self.rise_delta = rise_delta
        self.threshold_delta = threshold_delta
        self.prev_score = float('inf')
        self.counter = 0
        self.counted = False

    def __call__(self, train_score, test_score: float) -> bool:
        if test_score < self.prev_score:
            self.counter = 0
            self.prev_score = test_score

        if self.check_rise(test_score) or self.check_threshold(train_score, test_score):
            return True
        
        self.counted = False
        
        return False
    
    
    def check_rise(self, test_score: float) -> bool:
        """
        Check if the test score has risen above the previous score by a certain delta.

        Args:
            test_score (float): input test score

        Returns:
            bool: True if the test score has risen above the previous score by the delta, False otherwise.
        """
        if test_score > (self.prev_score + self.rise_delta):
            self.counter += 1
            self.counted = True
            if self.counter >= self.patience:
                return True

        return False
    
        
    def check_threshold(self, train_score: float, test_score: float) -> bool:
        """
        Check if the test score is above the train score by a certain delta.

        Args:
            train_score (float): input train score
            test_score (float): input test score

        Returns:
            bool: True if the test score is above the train score by the delta, False otherwise.
        """
        if test_score > (train_score + self.threshold_delta):
            self.counter += 1
            self.counted = True
            if self.counter >= self.patience:
                return True
        
        return False


class Model(nn.Module):
    def __init__(self, model_dict: OrderedDict):
        super().__init__()
        self.linear = nn.Sequential(
            model_dict
        )

    def forward(self, x):
        # x = x.flatten(start_dim=1)
        logits = self.linear(x)
        return logits
    

class NeuralNetwork:
    def __init__(
        self,
        model_dict: OrderedDict = None,
        optimizer = torch.optim.Adam,
        loss_function = nn.BCELoss,
        logger: Callable|None = None,
        epochs: int = 5,
        batch_size: int = 2048,
        lr: float = 1e-3,
        lmbda: float = 0,
        balanced_weights: bool = False,
        generator_weights: bool = False,
        nfolds: int = 4,
        njobs: int = 1,
        device: str = "cpu",
        classification: str = 'binary',
        cv : bool = False,
        output_path: str|Path = 'Output',
        rng: int = None,
        pbar: bool = True,
        early_stopping: Callable|None = None,
        channel: str = '',
        features: list = None,
    ) -> None:
        """
        Initialize the NeuralNetwork class.

        Parameters:
        model (Model): The neural network model to be trained.
        optimizer (torch.optim.Optimizer): The optimizer to use for training.
        loss_function (Callable): The loss function to use for training.
        lr (float): Learning rate for the optimizer.
        lmbda (float): Weight decay (L2 regularization) for the optimizer.
        class_weights (torch.Tensor|None): Class weights for the loss function.
        device (str): Device to run the model on ("cpu" or "cuda").
        rng (int|None): Random seed for reproducibility.
        """
        
        # Set random seed for reproducibility if provided
        self.seed = rng
        if rng is not None:
            torch.manual_seed(rng)
            np.random.seed(rng)
        
        self.classification = classification
        # Initialize model, loss function, and optimizer
        self.model_dict = model_dict
        self.optimizer = optimizer
        self.loss_function = loss_function
        self.logger = logger

        self.epochs = epochs
        self.batch_size = batch_size
            
        self.lr = lr
        self.lmbda = lmbda
        self.balanced_weights = balanced_weights
        self.generator_weights = generator_weights

        self.nfolds = nfolds
        self.njobs = njobs
        
        self.device = device
        self.pbar = pbar
        self.cv = cv
        
        self.activation_output = nn.Sigmoid() if classification == 'binary' else nn.Softmax(dim=1)
        
        if early_stopping is None:
            self.early_stopping = lambda x, y: False
        else:
            self.early_stopping = early_stopping
        
        self.norm_mode = None
        self.test_size = 0.2
        
        self.channel = channel
        self.features = features
        self.output_path = output_path

        self._results = {}
        self.cv_models = {}
        self.cv_indices = {}
        
    def _train(
            self, 
            dataloader: DataLoader,
            model: Model,
            optimizer: torch.optim.Optimizer,
            loss_function: Callable,
        ) -> tuple[float, torch.Tensor, torch.Tensor, torch.Tensor]:

        model.train()
        num_batches = len(dataloader)
        
        loss_train = 0
        pred_array = torch.empty(0, device=self.device)
        y_array = torch.empty(0, device=self.device)
        w_array = torch.empty(0, device=self.device)
        for X, y, w in dataloader:
            pred = model(X)
            
            # Check for binary classification and if True, create 1d tensors for the predictions and set true values as float
            if self.classification == 'binary':
                pred = pred[:,0]
                y = y.float()
                
            # Compute the loss
            loss = loss_function(pred, y)
                        
            # Add the event-by-event weights
            loss *= w
            
            loss = loss.mean()
                        
            # Do the backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # Save the results
            loss_train += loss.item()
            pred_array = torch.cat((pred_array, pred))
            y_array = torch.cat((y_array, y))
            w_array = torch.cat((w_array, w))

        with torch.no_grad():
            return loss_train/num_batches, self.activation_output(pred_array.cpu()), y_array.cpu(), w_array.cpu()

    def _test(
            self, 
            dataloader: DataLoader,
            model: Model,
            loss_function: Callable,
        ) -> tuple[float, torch.Tensor, torch.Tensor, torch.Tensor]:

        model.eval()
        num_batches = len(dataloader)
        
        loss_test = 0
        pred_array = torch.empty(0, device=self.device)
        y_array = torch.empty(0, device=self.device)
        w_array = torch.empty(0, device=self.device)
        with torch.no_grad():
            for X, y, w in dataloader:
                pred = model(X)

                if self.classification == 'binary':
                    pred = pred[:,0]
                    y = y.float()

                loss = loss_function(pred, y)

                loss *= w
                
                loss_test += torch.mean(loss).item()
                pred_array = torch.cat((pred_array, pred))
                y_array = torch.cat((y_array, y))
                w_array = torch.cat((w_array, w))

            return loss_test/num_batches, self.activation_output(pred_array).cpu(), y_array.cpu(), w_array.cpu()

    def fit(
            self,
            X: torch.Tensor,
            y: torch.Tensor,
            w: torch.Tensor|None = None,
            **kwargs
    ) -> None:
        """_summary_

        Args:
            X (torch.Tensor): Input features, should have shape (num_events, num_features).
            y (torch.Tensor): Target labels, for multiclass should be indexed i=0,1,2,...,C, where C is the number of classes. Must have shape (num_events).
            w (torch.Tensor | None, optional): Event weight. Defaults to None. Must have shape (num_events).
            **kwargs: Additional keyword arguments for the model.
        """
        
        for key, value in kwargs.items():
            try:
                setattr(self, key, value)
            except AttributeError:
                raise AttributeError(f'Attribute {key} does not exist in the NeuralNetwork class.')


        if self.cv:
            self.cv_models = {}
            self.cv_indices = {}
            
            skf = StratifiedKFold(self.nfolds, shuffle=True)

            pool = mp.Pool(self.njobs)
            
            def error_handler(e):
                traceback.print_exception(type(e), e, e.__traceback__)
            
            
            # Prepare the cross-validation processes
            with mp.Manager() as manager, enlighten.get_manager() as pbar_manager:
                BaseManager.register('Logger', self.logger)
                logger = BaseManager()
                logger.start()
                logger = logger.Logger(
                    channel=self.channel,
                    features=self.features,
                    path=self.output_path,
                    classification=self.classification,
                )
                
                # Prepare the progress bar, works with multiprocessing
                active = {}
                bar_format = u'{desc}{desc_pad}{percentage:3.0f}%|{bar}| ' + \
                        u'S:' + pbar_manager.term.yellow2(u'{count_0:{len_total}d}') + u' ' + \
                        u'F:' + pbar_manager.term.green3(u'{count_1:{len_total}d}') + u' ' + \
                        u'E:' + pbar_manager.term.red2(u'{count_2:{len_total}d}')

                pb_started = pbar_manager.counter(
                    all_fields=False, total=self.nfolds, desc='Folds:', color='yellow2', bar_format=bar_format, enabled=self.pbar
                )
                pb_finished = pb_started.add_subcounter('green3', all_fields=True)
                pb_error = pb_started.add_subcounter('red2', all_fields=True)

                # Setup each fold for training and start them
                for fold, (train_idx, test_idx) in enumerate(skf.split(X.cpu(), y.cpu())):
                    queue = manager.Queue() # queue for accessing results
                    
                    # Start the cross-validation process
                    result = pool.apply_async(
                        self.cross_validation, 
                        args=(X, y, w, queue, logger, fold, train_idx, test_idx),
                        error_callback=error_handler,
                    )
                    
                    # Start the progress bar for the fold
                    counter = pbar_manager.counter(total=self.epochs, desc=f'   Fold {fold+1}', unit='epochs', leave=True, enabled=self.pbar)

                    pb_started.update()
                    active[fold] = (result, queue, counter)
                    self.cv_indices[fold] = (train_idx, test_idx)
                        
                # Update the progress bars
                while active:
                    for fold in tuple(active.keys()):
                        result, queue, counter = active[fold]
                        alive = result.ready()
                        
                        try:
                            count = queue.get_nowait()
                            # last_results = logger.get_last_results(fold)
                        except:
                            count = 0
                            
                        if count:
                            counter.update(count - counter.count)

                        # Check if the process is finished
                        if alive:
                            counter.close()
                            del active[fold]
                            
                            try:
                                result.successful()
                                pb_finished.update_from(pb_started)
                            except:
                                pb_error.update_from(pb_started)


                    time.sleep(0.1) # Sleep to avoid busy waiting

                pool.close()
                pool.join()
                
                # Create a new copy of the logger on the main process to avoid ROOT problems
                logger = deepcopy(logger)
                
                # Write the results to file
                logger.on_complete(cv=True)

        else:
            logger = self.logger(
                self.channel,
                features=self.features,
                path=self.output_path,
                classification=self.classification,
            )

            X_train, X_test, y_train, y_test, w_train, w_test = train_test_split(
                X, y, w, test_size=self.test_size
            )            

            with torch.no_grad():
                # setup logger
                logger.on_start(
                    {'train': X_train.cpu(), 'test': X_test.cpu()},
                    {'train': y_train.cpu(), 'test': y_test.cpu()},
                    {'train': w_train.cpu(), 'test': w_test.cpu()},
                )
                
            # Prepare training weights
            class_weights = None
            if self.balanced_weights: 
                if self.generator_weights:
                    # Normalize the generator weights
                    if self.classification == 'binary':
                        w_norm = w_train[y_train == 0].sum() / w_train[y_train == 1].sum()
                        w_train[y_train == 1] *= w_norm
                        w_test[y_test == 1] *= w_norm
                        
                        print((w_train[y_train == 1].sum(), w_train[y_train == 0].sum()))
                        
                    else:
                        w_sum = w_train.sum()
                        for i in range(3):
                            w_train[y_train == i] *= w_sum / w_train[y_train == i].sum()
                            w_test[y_test == i] *= w_sum / w_train[y_train == i].sum()
                            
                        print(w_train[y_train == 0].sum(), w_train[y_train == 1].sum(), w_train[y_train == 2].sum())
                else:    
                    # Compute class weights
                    w_train = torch.ones_like(y_train, device=self.device)
                    w_test = torch.ones_like(y_test, device=self.device)
                    
                    if self.classification == 'binary':
                        class_weights = (y_train == 0).sum() / y_train.sum()
                    
                    else:
                        class_weights = torch.tensor(
                            compute_class_weight('balanced', classes=np.unique(y_train.cpu().numpy()), y=y_train.cpu().numpy()), 
                            device=self.device,
                            dtype=torch.float
                        )
                
            elif not self.generator_weights:
                # No weighting
                w_train = torch.ones_like(y_train, device=self.device)
                w_test = torch.ones_like(y_test, device=self.device)
                
            # Create dataloaders
            train_dataset = TensorDataset(X_train, y_train, w_train)
            test_dataset = TensorDataset(X_test, y_test, w_test)
            
            train_dataloader = DataLoader(train_dataset, batch_size=self.batch_size)
            test_dataloader = DataLoader(test_dataset, batch_size=self.batch_size)
            
            # Initialize model, loss function, and optimizer
            model = Model(self.model_dict)
            if self.classification == 'binary':
                loss_function = self.loss_function(
                    reduction='none',
                    pos_weight=class_weights
                )
            else:
                loss_function = self.loss_function(
                    reduction='none',
                    weight=class_weights
                )
            optimizer = self.optimizer(
                model.parameters(),
                lr=self.lr,
                weight_decay=self.lmbda
            )
            model.to(self.device)
            
            # Start training and testing the model
            # for t in range(self.epochs):
            pbar = tqdm(range(self.epochs), desc='Epochs', disable=not self.pbar, unit='epochs')
            for t in pbar:
                train_results = self._train(train_dataloader, model, optimizer, loss_function)

                test_results = self._test(test_dataloader, model, loss_function)
                
                logger.log(
                    {'train': train_results, 'test': test_results},
                )
        
                pbar.set_postfix(logger.get_last_results())
                
                if self.early_stopping(train_results[0], test_results[0]):
                    print(f"Early stopping at epoch {t+1}")
                    break
                
            self.model = model
            self.loss_function = loss_function
            
            logger.on_complete()
            
            
    def cross_validation(
            self, 
            X: torch.Tensor,
            y: torch.Tensor,
            w: torch.Tensor|None,
            queue: mp.Queue,
            logger,
            fold: int, 
            train_idx: ndarray, 
            test_idx: ndarray
        ) -> None:
        
        np.random.shuffle(train_idx)
        np.random.shuffle(test_idx)
        
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        w_train, w_test = w[train_idx], w[test_idx]
        
        logger.on_start(
            {'train': X_train.cpu(), 'test': X_test.cpu()},
            {'train': y_train.cpu(), 'test': y_test.cpu()},
            {'train': w_train.cpu(), 'test': w_test.cpu()},
            fold=fold
        )
        
        if self.norm_mode == 'balanced':
            # w_norm = w_train[y_train == 0].sum() / w_train[y_train == 1].sum()
            # w_train[y_train == 1] *= w_norm
            # w_test[y_test == 1] *= w_norm
            if self.classification == 'binary':
                w_norm = w_train[y_train == 0].sum() / w_train[y_train == 1].sum()
                w_train[y_train == 1] *= w_norm
                w_test[y_test == 1] *= w_norm
                
                print((w_test[y_test == 1].sum(), w_test[y_test == 0].sum()))
                
            else:
                w_sum = w_train.sum()
                for i in range(3):
                    w_norm = w_sum / w_train[y_train== i].sum()
                    w_train[y_train == i] *= w_norm
                    w_test[y_test == i] *= w_norm
                    
                print(w_test[y_test == 0].sum(), w_test[y_test == 1].sum(), w_test[y_test == 2].sum())

        train_ds = TensorDataset(X_train, y_train, w_train)
        test_ds = TensorDataset(X_test, y_test, w_test)
        
        # train_subsampler = SubsetRandomSampler(train_idx)
        # test_subsampler = SubsetRandomSampler(test_idx)

        train_dataloader = DataLoader(train_ds, batch_size=self.batch_size)#, sampler=train_subsampler)
        test_dataloader = DataLoader(test_ds, batch_size=self.batch_size)#, sampler=test_subsampler)

        model = Model(self.model_dict)
        model.to(self.device)
        
        if self.classification == 'binary':
            loss_function = self.loss_function(
                reduction='none',
                pos_weight=self.class_weights
            )
        else:
            loss_function = self.loss_function(
                reduction='none', 
                weight=self.class_weights
            )
        optimizer = self.optimizer(
            model.parameters(),
            lr=self.lr,
            weight_decay=self.lmbda
        )

        for t in range(self.epochs):
            train_results = self._train(train_dataloader, model, optimizer, loss_function)

            test_results = self._test(test_dataloader, model, loss_function)

            queue.put(t+1)
            
            logger.log(
                {'train': train_results, 'test': test_results},
                fold=fold
            )

            # if self.early_stop:
            #     break
            

    def predict(self, X: torch.Tensor) -> torch.Tensor:
        """
        Predict the output of the model for a given input tensor.
        This function uses the trained model to make predictions on the input data, the model can be loaded using the <load_model> function.
        The input tensor should be of shape (num_samples, num_features).

        Args:
            X (torch.Tensor): Input tensor of shape (num_samples, num_features).

        Returns:
            torch.Tensor: Predicted output tensor of shape (num_samples, num_classes) for multiclass classification or (num_samples) for binary classification.
        """
        with torch.no_grad():
            self.model.eval()
            
            pred = self.model(X)
            
            if self.classification == 'binary':
                pred = self.activation_output(pred[:,0])
            else:
                pred = self.activation_output(pred)
                
            return pred.cpu()


    def save_model(self, path: str|Path) -> None:
        # Create the directory if it doesn't exist
        path.mkdir(parents=True, exist_ok=True)
        
        path = str(path)
        
        cl = 'bc' if self.classification == 'binary' else 'mc'
        name = f'{path}/{cl}_{self.channel}_model.pt'
        
        # Save the model
        torch.save(self.model.state_dict(), name)
        
        
    def load_model(self, path: str|Path) -> None:
        path = str(path)
        
        cl = 'bc' if self.classification == 'binary' else 'mc'
        name = f'{path}/{cl}_{self.channel}_model.pt'
        
        state_dict = torch.load(name, map_location=self.device, weights_only=True)
        
        keys = list(state_dict.keys())
        
        layers = []
        for key, value in state_dict.items():
            if 'weight' in key and not 'batch' in key:
                n = value.shape[0]
                
                if 'output' in key:
                    output_shape = n
                else:
                    layers.append({'neurons': n, 'activation': 'relu'})
                
        input_shape = state_dict[keys[0]].shape[1]
        md = set_model_structure(input_shape, output_shape, *layers)
        
        self.model = Model(md)
        self.optimizer = self.optimizer(
            self.model.parameters(),
            lr=self.lr,
            weight_decay=self.lmbda
        )
        
        self.model.to(self.device)
        
        self.model.load_state_dict(state_dict)
        