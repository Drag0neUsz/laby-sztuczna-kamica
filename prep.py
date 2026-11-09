import pandas as pd
from pathlib import Path

split_pos=60

def prep_data(): 
    file_train = Path('tr_set.csv')
    file_test = Path('test.csv')

    if file_train.exists() and file_test.exists():
        return

    data = pd.read_csv('data.csv', sep=',')
    data=data.sample(frac=1)

    tr_set = data.iloc[:split_pos]
    train_min = tr_set.min()
    train_max = tr_set.max()
    tr_set = (tr_set - train_min) / (train_max - train_min)

    test_set = data.iloc[split_pos:]
    test_set = (test_set - train_min) / (train_max - train_min)
    tr_set.to_csv('tr_set.csv', index=False)
    test_set.to_csv('test.csv', index=False)

if __name__ == '__main__':
    prep_data()