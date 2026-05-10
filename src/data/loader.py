import pandas as pd


class DataLoader:
    def __init__(self, filepath):
        self.filepath = filepath

    def load_data(self):
        df = pd.read_csv(self.filepath)

        return df