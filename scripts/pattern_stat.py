from __future__ import print_function, division

import os
import sys
sys.path.append(os.path.dirname(sys.path[0]))
from itertools import tee

import numpy as np
from matplotlib import pyplot as plt

from cryoio import mrc
import geometry
from notimplemented import correlation

import pyximport; pyximport.install(
    setup_args={"include_dirs": np.get_include()}, reload_support=True)
import sincint



def analyse_mrcs_stack(img_path, mass):

    data = mrc.readMRC(img_path)
    imgstack = np.transpose(data, axes=(2, 0, 1))

    num_images, N, _ = imgstack.shape
    pixel_size = 3 * 6

    mean_stat = []
    ratio_stat = []


    # for freq_start, freq_end in zip(np.arange(0.01, 0.05, 0.005), np.arange(0.015, 0.055, 0.005)):
    for freq_start, freq_end in zip(np.arange(0.01, 0.050, 0.002), np.arange(0.012, 0.052, 0.002)):

        beamstop_rad = freq_start * pixel_size
        rad = freq_end * pixel_size

        # print("start rad: {}, end rad: {}.".format(beamstop_rad, rad))

        FtoT = sincint.genfulltotrunc(N, rad, beamstop_rad)
        TtoF = sincint.gentrunctofull(N, rad, beamstop_rad)

        mean_array = np.zeros(num_images)
        ratio_array = np.zeros(num_images)
        for i, img in enumerate(imgstack):
            trunc_img = FtoT.dot(img.flatten())
            # trunc_img = correlation.calc_angular_correlation(trunc_img, N, rad, beamstop_rad, pixel_size, clip=True)
            # trunc_img = correlation.get_corr_img(img, rad, beamstop_rad).reshape(-1, 360)
            trunc_img = correlation.get_corr_img(img, 1, beamstop_rad).reshape(-1, 360)
            # plt.figure()
            # plt.imshow(TtoF.dot(trunc_img).reshape(N, N))
            plt.imshow(trunc_img)
            # plt.xticks_lab([0, ''])
            plt.yticks([])
            plt.ylabel('radius')
            plt.xlabel('$\Phi$')
            plt.axis('auto')
            plt.colorbar()
            # plt.figure()
            # plt.hist(trunc_img)
            plt.show()

            num_trunc = trunc_img.shape[0]
            # print(num_trunc)
            mean_array[i] = trunc_img.mean()
            greater_than_1 = (trunc_img > 1).sum()
            ratio_array[i] = greater_than_1 / num_trunc

        mean_stat.append(mean_array.mean())
        ratio_stat.append(ratio_array.mean())
    
    freqs = np.arange(0.01, 0.052, 0.002)

    fig, ax0 = plt.subplots(figsize=(9.6, 4.8))
    ax0.plot(freqs[:-1], mean_stat, 'bo-')
    print(mean_stat)
    ax0.set_xlabel('Frequency Domain')
    ax0.set_ylabel('Average Intensity', color='b')
    # ax0.set_ylim([0, 60])
    ax0.tick_params('y', colors='b')

    ax1 = ax0.twinx()
    ax1.plot(freqs[:-1], ratio_stat, 'ro-')
    ax1.set_ylabel('Ratio of Intensity greater than 1', color='r')
    ax1.set_ylim([0, 1.1])
    ax1.tick_params('y', colors='r')


    ax0.set_title('EMD 6044 totalmass {:,} oversampling 6'.format(int(mass * 1e5)))
    fig.tight_layout()
    plt.savefig('old_ac_stat_%s' % str(mass).zfill(2), dpi=300, bbox_inches='tight')
    # plt.show()

    return mean_stat, ratio_stat



if __name__ == '__main__':

    # img_path = sys.argv[1]

    # img_path = 'data/EMD6044_xfel_5000_totalmass_2000000_oversampling_6/imgdata.mrc'
    # mass = 20
    # analyse_mrcs_stack(img_path, mass)

    img_path = 'data/EMD6044_xfel_5000_totalmass_1000000_oversampling_6/imgdata.mrc'
    mass = 10
    analyse_mrcs_stack(img_path, mass)

    img_path = 'data/EMD6044_xfel_5000_totalmass_0500000_oversampling_6/imgdata.mrc'
    mass = 5
    analyse_mrcs_stack(img_path, mass)

    img_path = 'data/EMD6044_xfel_5000_totalmass_1500000_oversampling_6/imgdata.mrc'
    mass = 15
    analyse_mrcs_stack(img_path, mass)