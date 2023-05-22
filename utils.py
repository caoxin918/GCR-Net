import numpy as np

def randomShuffle(x_data,y,num_point):
    seq = np.random.choice(num_point, num_point, replace=False)
    x_data_shuffle = x_data[:,seq,:]
    y_shuffle = y[:,seq]
    return x_data_shuffle, y_shuffle, seq
