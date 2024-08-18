import pandas as pd

def clean_data(input_path, output_path):
    data = pd.read_csv(input_path, na_values=' ?', skipinitialspace=True)
    data.to_csv(output_path, index=False)

if __name__ == "__main__":
    clean_data('data/census.csv', 'data/clean_census.csv')

