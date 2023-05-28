# -*- coding: utf-8 -*-
"""
% The demo is about the Restored Pseudo Heat Flux for detection of small, heterogeneous object insertion defects in CFRP
% If you use this code please cite this paper
% H. Wang et al. "A Physical Information Constrained Decomposition Method for Thermography: Pseudo Restored Heat Flux based on Ensemble Bayesian Variance Tensor Fraction," in IEEE Transactions on Industrial Informatics
% Report bugs to Hongjin Wang
% Email address: hongjinwang2017@outlook.com
% Hongjin Wang -at- http://eeit.hnu.edu.cn/info/1506/4922.htm
% Checked 28/5/2023
"""

import scipy.io as sio
import numpy as np
import h5py
import matplotlib.pyplot as plt
import RPHF_v0 as phf
import cv2
from PIL import Image

import numpy as np
import platform

import time

print('SYSTEM:', platform.system())

# input mat file
matfn = 's_60.mat'

# the width, height and length of the mat file
wp = 320 * 2
hp = 240 * 2
frM = 3470
f = h5py.File(matfn, 'r')
data = f.get('T_smth')
data = np.array(data).astype('single')
rs = 5

# create AOI
data = np.array(data[:, :, rs:])
plt.imshow(data[1, :, :])

# use algorithm of RPHF to deal with the data
T1 = time.perf_counter()
RPHF = phf.RPHF_v0(data, 9)
T2 = time.perf_counter()
print('program run time:%s ms' % ((T2 - T1) * 1000))
# program run time:0.27023641716203606 ms

# plot the time response of RPHF at the center point of AOI
plt.figure()
plt.plot(RPHF[:, 153, 108])
plt.show()

# show the RPHF at frame 440
frmNN = 440
RPHFc = RPHF[frmNN, ::]
nRPHF = np.zeros(RPHF[frmNN, ::].shape, np.single)
amin = np.min(RPHFc)
amx = np.max(RPHFc)
nRPHF = (RPHFc - amin) / (amx - amin)
img = Image.fromarray((nRPHF * 255).astype(np.uint8))
cv2_img = np.asarray(img)
clahe = cv2.createCLAHE(clipLimit=5, tileGridSize=(8, 8))
cl1 = clahe.apply(cv2_img)
plt.imshow(cl1)
plt.show()
