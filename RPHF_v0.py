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

import numpy as np
import numpy.matlib

import cv2


def RPHF_v0(datan, a2=-1, tscal=-1, sscal=-1, ):
    # initialize variables

    if (tscal < 0):
        tscal = 1

    if (sscal < 0):
        sscal = 1

    if (a2 < 0):
        print('a2 is a require input argument')
        return

    # preprocessing
    tt = len(datan)
    T_bas = datan[0, ::]
    datan = datan - T_bas
    data = datan[np.arange(0, tt, tscal), :-1:sscal, :-1:sscal]
    frM, n, m = data.shape
    vm = int(2 ** np.ceil(np.log2(n)))
    um = int(2 ** np.ceil(np.log2(m)))

    # for long recode where the temperature almost drop to 0,
    # force the padding length to be 1/10 of the original length
    # this assumption may only suitable to the demo data since
    # the temperature in the demo data has almost drop to the temperature 
    # of the environment in the end
    frMN = 500  # int(2**np.ceil(np.log2(frM)))
    PM = int(np.floor((um - m) / 2))
    PN = int(np.floor((vm - n) / 2))

    # padding
    T_altP = np.lib.pad(data, ((10, frMN - 10), (PN, vm - n - PN), (PM, um - m - PM)), 'constant', constant_values=0)

    # sscal=1
    um_e = int(um / sscal)
    vm_e = int(vm / sscal)

    t_m = int(frMN + frM / tscal)  # 2 ** Tn
    gs = 3

    print('Now is Computing ...')
    k = 0
    for k in np.arange(t_m):
        T_altP[k, ::] = cv2.GaussianBlur(T_altP[k, ::], (gs, gs), 0)
    print('Calculation of the Inversion in Progress ...')

    FFT3_altP = np.fft.fftshift((np.fft.fftn(np.fft.fftshift(T_altP))).astype('csingle'))

    # fix the cycle to [0,2pi)
    u = np.fft.fftshift(np.fft.fftfreq(um_e))
    v = np.fft.fftshift(np.fft.fftfreq(vm_e))
    w = np.fft.fftfreq(t_m)
    w = np.fft.fftshift(w)

    # RPHF convolution processing
    muv = (np.tile(u.astype('single') ** 2, vm_e).reshape((vm_e, um_e)) + np.repeat(v.astype('single') ** 2,
                                                                                    um_e).reshape((vm_e, um_e)))

    muvw2 = np.sqrt(
        (np.repeat(w.astype('single'), um_e * vm_e) / a2 * 1j + np.tile(muv.reshape(-1), t_m)).reshape(t_m, vm_e,
                                                                                                       um_e) * 2 * np.pi)  # padarray((np.sqrt(np.matlib.repmat(muv,1,1,t_m) + (reshape(np.ones((vm_e * um_e,1)) * 1j * w / (a2),um_e,vm_e,t_m)))),np.array([0,0,0]),'both')

    MUV = np.fft.ifftshift(np.fft.ifftn(np.fft.ifftshift(np.multiply(FFT3_altP,
                                                                     muvw2))).real)  # fftshift(ifft(ifft(ifft(ifftshift(np.multiply(reshape(FFT3_altP,um_e,vm_e,t_m),muvw2)),[],1,'nonsymmetric'),[],2,'nonsymmetric'),[],3,'nonsymmetric'))

    # result of RPHF
    MUV_e = MUV[1 + 5:-2 - frMN, PN + 7:-PN - 8, PM + 7:-PM - 8] + MUV[2 + 5:-1 - frMN, PN + 7:-PN - 8, PM + 7:-PM - 8]
    print('Finished!')

    return MUV_e
