import torch
import numpy as np
from pathlib import Path
import uproot
import argparse
import os
import sys
import yaml

sys.path.append(str(Path(__file__).resolve().parents[1]))

from neural_network import NeuralNetwork
from utils.timer import Timer

def main(
    channel: str = '1l0tau',
    input_path: Path = Path('Data'),
    output_path = Path('../Files'),
    classification: str = 'binary'
) -> None:
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    cl = 'bc' if classification == 'binary' else 'mc'
    with open(workspace_path / f'Configs/NN/{cl}_{channel}_config.yaml', 'r') as f:
        config = yaml.safe_load(f)

    features = config['training']['input_features']

    net = NeuralNetwork(
        classification=classification,
        channel=channel,
        device=device,
    )
    
    net.load_model(workspace_path / f'Output/{channel}/NN/weights')
    
    processes = ['SH', 'Sherpa', 'Vyy', 'signal_ggF', 'signal_VBF', 'data']
    with torch.no_grad():
        for process in processes:
            with Timer() as t:
                try:
                    with uproot.open(workspace_path / output_path / f'{process}_{channel}.root') as oldfile:
                        output = oldfile['output']
                        old = uproot.concatenate(output)
                            
                        data = output.arrays(features, library='np')
                    
                except:
                    with uproot.open(workspace_path / input_path / f'{process}_{channel}.root') as oldfile:
                        output = oldfile['output']
                        old = uproot.concatenate(output)
                            
                        data = output.arrays(features, library='np')
                
                X = torch.tensor(np.array([arr for arr in data.values()]).T, dtype=torch.float32, device=device)
                
                t.set_msg(f'Finished processing {process}_{channel}.root with {X.size(0)} events')
                
                pred = net.predict(X)
                
                
                if classification == 'multiclass':
                    old['DNN_signal'] = pred[:,0]
                    old['DNN_SH'] = pred[:,1]
                    old['DNN_CB'] = pred[:,2]                
                else:
                    old['DNN'] = pred
                
                with uproot.recreate(workspace_path / output_path / f'{process}_{channel}.root') as newfile:
                    newfile['output'] = old
                

if __name__=="__main__":
    parser = argparse.ArgumentParser(
        description='Running the trained neural network and writing the predictions to root files'
    )
    
    parser.add_argument(
        '-c', '--channel', type=str, default='1l0tau', help='Name of the channel. Default: 1l0tau'
    )
    parser.add_argument(
        '-i', '--input', type=Path, default=Path('Input_Files'), help='Input path for the hdf5 file. Default: /Input_Files'
    )
    parser.add_argument(
        '-o', '--output', type=Path, default=Path('Output/Files'), help='Output path for the root files, these files must already exist. Default: Output/Files'
    )
    parser.add_argument(
        '-cl', '--classification', type=str, default='binary', choices=['binary', 'multiclass'], help='Type of classification. Default: binary'
    )
    
    args = parser.parse_args()
    
    workspace_path = Path(__file__).resolve().parents[2]
    
    # Check if output path exists, if not create them
    try:
        os.makedirs(workspace_path / args.output)
    except FileExistsError:
        pass
    
    main(args.channel, args.input, args.output, args.classification)