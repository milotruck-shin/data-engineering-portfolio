from pathlib import Path
import argparse
import pandas as pd
import numpy as np

def parse_args():
    parser = argparse.ArgumentParser(
        description="Alarm correlation script"
    )
    parser.add_argument(
        "input_path",
        type=Path,
        help="Path to input CSV file or folder"
    )
    return parser.parse_args()

def load_csv_folder(folder : Path, pattern: str="*.csv",**read_csv_kwargs) -> pd.DataFrame:
    csv_files = sorted(folder.glob("*.csv"))
    if not csv_files:
        raise FileNotFoundError (f"No CSV files matching '{pattern} in {folder}") #raise ExceptionType("message")
    dfs=[]

    for f in csv_files:
        data_frame = pd.read_csv(f,**read_csv_kwargs)
        dfs.append(data_frame)
    return (pd.concat(dfs, ignore_index=True))

if __name__=="__main__":

    args = parse_args()
    input_path = args.input_path
    if input_path.is_dir():

        out_path = input_path.joinpath("MERGED.csv")
        if out_path.exists():
            out_path.unlink()   # delete old result.csv
        df = load_csv_folder(input_path)

    elif input_path.is_file():
        df = pd.read_csv(input_path)
        out_path = input_path.with_name("MERGED.csv")

    else:
        raise ValueError("Input path must be a file or directory")
    
    df.to_csv(out_path,index=False)
