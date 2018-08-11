import mrcfile
import os.path
import numpy as np

input_path = '/Users/jingjingy/Python/Research/result/high_res/model_4'

def readmrc(input):
    mrc = mrcfile.open(input, mode='r')
    arr = mrc.data[:]
    return arr

def Hausdorff_dist(vol_a,vol_b):
    dist_lst = []
    for idx in range(len(vol_a)):
        dist_min = 1000.0
        for idx2 in range(len(vol_b)):
            dist= np.linalg.norm(vol_a[idx]-vol_b[idx2])
            if dist_min > dist:
                dist_min = dist
        dist_lst.append(dist_min)
    return np.max(dist_lst)

for i in range(20):
    ori_path = os.path.join(input_path, 'ori', 'ori' + str(i) + '.mrc')
    recon_path = os.path.join(input_path, 'recon', 'recon' + str(i) + '.mrc')
    ori = readmrc(ori_path)
    recon = readmrc(recon_path)
    dist = Hausdorff_dist(ori, recon)
    print(dist)
