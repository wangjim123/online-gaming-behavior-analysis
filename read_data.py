import pandas as pd

def main(file): #位址需手動更改
    file_path = f"Data/{file}"
    df = pd.read_csv(file_path)
    return df