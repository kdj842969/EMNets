import gzip
from pathlib import Path
import os.path

input_folder = '/mnt/sdc/ftp.rcsb.org/pub/pdb/data/structures/divided/pdb'
output_folder = '/mnt/sdc/yjj801/output/high_res'
load_file_path = './load_file.txt'
apix = 1.0
box_size = 64
resolutions = [3, 4, 5]

with open(load_file_path) as file:
    for line in file:
        if line.startswith('#'):
            continue
        fname = line.rstrip('\n')
        source_zip_file = os.path.join(input_folder, line[1:3], 'pdb' + fname + '.ent.gz')
        if os.path.isfile(source_zip_file):
            for res in resolutions:
                destination_file = os.path.join(output_folder, str(res) + 'A', fname + '.mrc')
                if Path(destination_file).exists():
                    print(destination_file + ' already exists. Skipping its generation...')
                    continue
                unzipped_file = os.path.join(output_folder, 'pdb', fname + '.pdb')
                if not Path(unzipped_file).exists():
                    with gzip.open(source_zip_file, 'rb') as f_in:
                        contents = f_in.read()
                        f_in.close()
                    open(unzipped_file, 'wb').write(contents)
                os.system('/home/user/EMAN2/bin/e2pdb2mrc.py {} {} --box {} --apix {} --res {} --center'.format(unzipped_file, destination_file, box_size, apix, res))
        else:
            print(source_zip_file + ' not exists. Skipping its generation...')
            continue
import glob
from random import shuffle
import numpy as np

combined =[]
for res in resolutions:
    folder = os.path.join(output_folder,str(res)+'A')
    temp_addrs = glob.glob(folder+'/*.mrc')
    temp_labels = [res for addr in temp_addrs]
    temp_combined = list(zip(temp_addrs,temp_labels))
    combined += temp_combined

shuffle(combined)
mrcfiles, resolutions = zip(*combined)

from sklearn.model_selection import train_test_split
mrcfile_train, mrcfile_test, resolut_train, resolut_test = train_test_split(mrcfiles,resolutions,test_size=0.4)
mrcfile_test, mrcfile_validation, resolut_test, resolut_validation = train_test_split(mrcfile_test, resolut_test, test_size=0.5)

train_file = open('train.txt', 'w')
for item in mrcfile_train:
    train_file.write("%s\n" % item)

test_file = open('test.txt', 'w')
for item in mrcfile_test:
    test_file.write("%s\n" % item)

validation_file = open('validation.txt', 'w')
for item in mrcfile_validation:
    validation_file.write("%s\n" % item)

#normalization function
from copy import deepcopy
import mrcfile as mrc

def normalize(mrcobject, mode='linear'):
    print("Normalizing")
    if mode not in ['linear', 'sigmoid']:
        raise ValueError("Mode '{0}' is not valid/supported".format(mode))
    normalized_data = deepcopy(mrcobject.data)
    if('linear' in mode):
        for index,value in np.ndenumerate(normalized_data):
            if (value <= 0.0):
                normalized_data[index] = 0.0
            else:
                normalized_data[index] = normalized_data[index] / mrcobject.header['dmax']
    if ('sigmoid' in mode):
        # sigmoid mapping between -1 and +1
        print('sigmoid')
        width = mrcobject.header['dmax'] - mrcobject.header['dmin']
        for index, value in np.ndenumerate(normalized_data):
            normalized_data[index] = 2 / (1 + math.exp((-normalized_data[index]) / width)) - 1
    return normalized_data

import h5py

train_shape = (len(mrcfile_train), box_size, box_size, box_size)
val_shape = (len(mrcfile_validation), box_size, box_size, box_size)
test_shape = (len(mrcfile_test), box_size, box_size, box_size)

hdf5_file = h5py.File("proteins.hdf5","w")
hdf5_file.create_dataset("train_mrc", train_shape, np.float32)
hdf5_file.create_dataset("val_mrc", val_shape, np.float32)
hdf5_file.create_dataset("test_mrc", test_shape, np.float32)

hdf5_file.create_dataset("train_resolution", (len(resolut_train), 1), np.float32)
hdf5_file.create_dataset("val_resolution",   (len(resolut_validation), 1), np.float32)
hdf5_file.create_dataset("test_resolution",  (len(resolut_test), 1), np.float32)

print(resolut_train)
print(resolut_test)
print(resolut_validation)

for i in range(len(mrcfile_train)):
    print("opening " + mrcfile_train[i])
    print(resolut_train[i])
    current_mrc = mrc.open(mrcfile_train[i], mode='r')
    normalized_data = normalize(current_mrc)
    hdf5_file["train_mrc"][i, ...] = normalized_data[None]
    hdf5_file["train_resolution"][i] = resolut_train[i]

for j in range(len(mrcfile_test)):
    print("opening " + mrcfile_test[j])
    print(resolut_test[j])
    current_mrc = mrc.open(mrcfile_test[j], mode='r')
    normalized_data = normalize(current_mrc)
    hdf5_file["test_mrc"][j, ...] = normalized_data[None]
    hdf5_file["test_resolution"][j] = resolut_test[j]

for k in range(len(mrcfile_validation)):
    print("opening " + mrcfile_validation[k])
    print(resolut_validation[k])
    current_mrc = mrc.open(mrcfile_validation[k], mode='r')
    normalized_data = normalize(current_mrc)
    hdf5_file["val_mrc"][k, ...] = normalized_data[None]
    hdf5_file["val_resolution"][k] = resolut_validation[k]

hdf5_file.close()
print('finish')
