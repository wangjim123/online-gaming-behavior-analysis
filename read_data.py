import pandas as pd

def main(file): #位址需手動更改
    file_path = "C:/Users/user/PycharmProjects/online-gaming-behavior-analysis/Data/"
    df = pd.read_csv(file_path + file)
    return df