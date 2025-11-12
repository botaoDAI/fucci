
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 28 12:22:35 2024

@author: Dai_botao
"""

import pandas as pd
import h5py
import matplotlib.pyplot as plt

# Your hdf5 file path
filename = './output_file_D1_01_0.0375.hdf5'




# Open HDF5 file using pandas
store = pd.HDFStore(filename, 'r')

# Open HDF5 file using h5py
f = h5py.File(filename, 'r')

# View file attributes
print("File attributes:")
for key in f.attrs.keys():
    print(f"{key}: {f.attrs[key]}")
# Get specific attribute
if 'channel' in f.attrs:
    channel = f.attrs['channel']
    print("Channel:", channel)
    
# View contents of each group
print("Contents of the HDF5 file:")
for key in f.keys():
    print(f"{key}:")
#    for subkey in f[key].keys():
#        print(f"  - {subkey}")

# View background image
if 'ImageB3-7-C2/background' in f:
    background = f['ImageB3-7-C2/background/image'][:]
    plt.imshow(background, cmap='gray')
    plt.title('Background Image')
    plt.show()
else:
    print("Background data not found.")
'''
f['background']['im'][:]  # Read data as an array
plt.show()
'''

# Close the file
f.close()

# Assuming you want to view the data of the first frame, replace 'frame0' with the appropriate key
num_frame = 'Imageframe/frame0'
p = pd.read_hdf(store, key=num_frame)
print("DataFrame contents:")
print(p)

store.close()
