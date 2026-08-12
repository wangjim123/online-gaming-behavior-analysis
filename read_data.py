import pandas as pd

def main():
    file_path = "C:/Users/USER/Desktop/PythonProject/Data/"
    df = pd.read_csv(file_path + 'online_gaming_behavior_dataset.csv')
    return df