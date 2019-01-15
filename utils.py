import numpy as np
import pandas as pd


def error(truth, estimation):
    return np.linalg.norm(truth-estimation) / np.linalg.norm(truth)


def accuracy(truth, estimation):
    return (truth == estimation).mean()
    

def split_dataframe(frac, df):
    _, cols = df.shape
    indices = np.arange(cols)
    np.random.shuffle(indices)
    split_i = int(frac * cols)

    df_train = df.iloc[:,indices[:split_i]]
    df_test = df.iloc[:,indices[split_i:]]
    
    return df_train, df_test
    
def split_dataframe_into_chunks(number, df):
    _, cols = df.shape
    indices = np.arange(cols)
    np.random.shuffle(indices)
    size = (int)(np.ceil(cols/number))
    res = []
    lower_i = 0

    for higher_i in range(size,cols,size):
        res.append(df.iloc[:,indices[lower_i:higher_i]])
        lower_i += size

    if higher_i != cols:
        res.append(df.iloc[:,indices[higher_i:cols]])
    
    return res
	
def append_new_column(old_adjacency, newcol):

    n = len(old_adjacency)
    new_adjacency = np.zeros((n+1, n+1))
    new_adjacency[:n,:n] = old_adjacency
    new_adjacency[n,:] = newcol
    new_adjacency[:,n] = newcol
    
    return new_adjacency
