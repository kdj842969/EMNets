import h5py
import numpy as np
from keras.models import Model
from keras.models import load_model

# load mrc dataset
with h5py.File('proteins.hdf5', 'r') as f:
    train_data = f['train_mrc'][...]
    val_data = f['val_mrc'][...]
    test_data = f['test_mrc'][...]

print(train_data.shape)
print(val_data.shape)
print(test_data.shape)

train_num = train_data.shape[0]
val_num = val_data.shape[0]
test_num = test_data.shape[0]
box_size = train_data.shape[1]

box_size = 64
train_data = train_data.reshape([-1, box_size, box_size, box_size, 1])
val_data = val_data.reshape([-1, box_size, box_size, box_size, 1])
test_data = test_data.reshape([-1, box_size, box_size, box_size, 1])

print(train_data.shape)
print(val_data.shape)
print(test_data.shape)

descriptor_len = 16

# load previously trained model
autoencoder = load_model('autoencoder_model_4.h5')

# Get encoder layer from trained model
encoder = Model(inputs=autoencoder.input, outputs=autoencoder.get_layer('encoder').output)

print("shape of learned_codes is:")

# compute descriptor for training set
learned_codes = encoder.predict(train_data)
print(learned_codes.shape)
learned_codes = learned_codes.reshape(learned_codes.shape[0], learned_codes.shape[1]*learned_codes.shape[2], learned_codes.shape[3]*learned_codes.shape[4])

# compute descriptor for validation set
val_codes = encoder.predict(val_data)
val_codes = val_codes.reshape(val_codes.shape[0], val_codes.shape[1]*val_codes.shape[2], val_codes.shape[3]*val_codes.shape[4])

# compute descriptor for testing set
test_codes = encoder.predict(test_data)
test_codes = test_codes.reshape(test_codes.shape[0], test_codes.shape[1]*test_codes.shape[2], test_codes.shape[3]*test_codes.shape[4])

import h5py
n_descriptor = len(train_data) + len(val_data) + len(test_data)
descriptor_shape = (n_descriptor, descriptor_len, descriptor_len)

hdf5_file = h5py.File("descriptor_model_4.hdf5", "w")
hdf5_file.create_dataset("descriptor", descriptor_shape, np.float32)

for i in range(len(learned_codes)):
    if i % 100 == 0:
        print('Training descriptor writing has finished: %d/%d' % (i, len(learned_codes)))
    hdf5_file["descriptor"][i, ...] = learned_codes[i]

for j in range(len(val_codes)):
    if j % 100 == 0:
        print('Validation descriptor writing has finished: %d/%d' % (j, len(val_codes)))
    hdf5_file["descriptor"][len(learned_codes) + j, ...] = val_codes[j]

for k in range(len(test_codes)):
    if j % 100 == 0:
        print('Validation descriptor writing has finished: %d/%d' % (k, len(test_codes)))
    hdf5_file["descriptor"][len(learned_codes) + len(val_codes) + k, ...] = val_codes[k]

hdf5_file.close()
print('Finish descriptor writing...')


