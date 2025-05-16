from pathlib import Path
import awkward as ak
import uproot
import h5py
import argparse

def procces_root_files(file: Path) -> ak.Array:
    print(f"Processing file: {file.name}")
    with uproot.open(str(file)) as f:
        output = f["output"]
        data = output.arrays()
        
    return data


def save_hdf5(hf: h5py.File, output: ak.Array, files: Path):
    i = 0
    for f in files:
        print(f'Processing group: {f.stem}')
        grp = hf.create_group(f.stem)
        grp.attrs['size'] = len(output[i].eventnumber)
        for key in output[i].fields:
            grp[key] = output[i][key]
            
        i += 1

        
def read_data(channel: str, output_path: Path, input_files: list[Path]|None, path: Path = None) -> None:
    print(f'-----------------------------------------\nStart processing channel: {channel}\n-----------------------------------------')
    
    files = []
    output_path /= f"{channel}.hdf5"
    
    if len(input_files) == 0:
        files = list(path.glob(f"*{channel}*.root"))
    else:        
        for file in input_files:
            pf = path / file
            if pf.is_dir():
                files.append(*pf.iterdir())
            elif pf.exists():
                files.append(pf)
            else:
                for new_file in path.glob(file.name):
                    files.append(new_file)
                    
    if len(files) == 0:
        raise FileNotFoundError(f"No ROOT files found in path: {path} for channel: {channel}")
                    
    data: list[ak.Array] = [procces_root_files(file) for file in files]
    
    print(f'-----------------------------------------\nStart write to file {output_path.name}\n-----------------------------------------')
    
    with h5py.File(str(output_path), "w") as hf:
        save_hdf5(hf, data, files)
            
    
    
if __name__=="__main__":
    parser = argparse.ArgumentParser(
        description="Processes root files into hdf5 files"
    )
    
    parser.add_argument(
        "-o", "--output", type=Path, default=Path('Input_Files'), help="Set the output file, default: /Data"
    )
    parser.add_argument(
        "-p", "--path", type=Path, default=Path('Input_Files'), help="Path where the root files are located, default: Input_Files"
    )
    parser.add_argument(
        '-c', '--channel', type=str, default=None, help="Channel to process, default: None, all channels will be processed"
    )
    parser.add_argument(
        "files", type=Path, nargs="*", default=None, help="root files you want to process, default uses path as directory"
    )
    
    args = parser.parse_args()
    
    if args.channel is None and len(args.files) == 0:
        print('--------------------------------\nNo channel specified, all channels will be processed\n-----------------------------------------')
        channels = ['2l0tau', '1l0tau', '1l1tau', '0l1tau', '0l2tau']
        
        for channel in channels:
            read_data(channel, args.output, args.files, args.path)
    else:
        read_data(args.channel, args.output, args.files, args.path,)
        
    print('-----------------------------------------\nAll files processed\n-----------------------------------------')
    