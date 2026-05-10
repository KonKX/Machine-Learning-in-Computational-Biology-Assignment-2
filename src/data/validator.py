class DatasetValidator:
    def __init__(self, df):
        self.df = df

    def check_shape(self):
        return self.df.shape

    def check_missing_values(self):
        missing_counts = self.df.isna().sum()
        return missing_counts[missing_counts > 0]

    def check_duplicates(self):
        return self.df.duplicated().sum()

    def check_dtypes(self):
        return self.df.dtypes

    def class_distribution(self, target_col):
        return self.df[target_col].value_counts(normalize=True)