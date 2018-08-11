from sklearn.neighbors import NearestNeighbors
import h5py
import numpy as np
import pandas as pd
import os.path
import datetime

with h5py.File('descriptor_model_4.hdf5', 'r') as f:
    dsc = f['descriptor'][...]

print(dsc.shape)

def readfile(path):
    df = pd.read_table(path, header=None)
    return df, len(df)

train, train_len = readfile('train.txt')
val, val_len = readfile('validation.txt')
test, test_len = readfile('test.txt')

full = [train, val, test]
full_data = pd.concat(full)
print(full_data.shape)

from sklearn import preprocessing
features = dsc.reshape(dsc.shape[0], dsc.shape[1] * dsc.shape[2])
# print(features[0])
min_max_scaler = preprocessing.MinMaxScaler()
normalize_features = min_max_scaler.fit_transform(features)
#  print(normalize_features[0])
start = datetime.datetime.now()
nbrs = NearestNeighbors(n_neighbors=10, algorithm='ball_tree').fit(features)
end = datetime.datetime.now()
time = end - start
print(int(time.total_seconds() * 1000)) # milliseconds

distances, indices = nbrs.kneighbors(features)
# print(indices[:5])
# print(distances[:5])

# print(indices)

import h5py
import matplotlib.pyplot as plt

with h5py.File('descriptor_model_3.hdf5', 'r') as f:
    descriptor = f['descriptor'][...]

def show_neighbors(idx, distances, indices, full_data):
    neighbors = indices[idx]
    dists = distances[idx]
    print(dists)
    result = []
    path = './descriptor'
    for i, neighbor in enumerate(neighbors):
        file_name = full_data.iloc[neighbor, 0]
        mrc = file_name[26:30]
        result.append(mrc)
        plt.gray()
        plt.imshow(descriptor[neighbor])
        file = os.path.join(path, str(mrc) + '.png')
        plt.savefig(file)
    print(result)

for i in range(50):
    print("No %d retrieval:" %(i + 1))
    show_neighbors(i, distances, indices, full_data)



