from __future__ import print_function, division

import os, sys, time, gc
sys.path.append(os.path.dirname(sys.path[0]))

try:
    import cPickle as pickle  # python 2
except ImportError:
    import pickle  # python 3

import numpy as np
from matplotlib import pyplot as plt

import geometry
import density
from cryoio import mrc
from cryoio.imagestack import MRCImageStack, FourierStack
from cryoio.ctfstack import CTFStack
from cryoio.dataset import CryoDataset
import cryoops
import cryoem
from quadrature import SK97Quadrature

from test.dataset_test import SimpleDataset
from test.likelihood_test import SimpleKernel

import pyximport; pyximport.install(setup_args={"include_dirs": np.get_include()}, reload_support=True)
import sincint


def plot_figures():
    oversampling_factor = 6
    psize = 3 * oversampling_factor
    freq = 0.05
    rad = 0.5 * 2.0 * psize
    beamstop_freq = 0.005
    beamstop_rad = beamstop_freq * 2.0 * psize


    # M = mrc.readMRC('particle/EMD-6044-cropped.mrc')
    # M[M < 0] = 0
    # M_totalmass = 2000000
    # if M_totalmass is not None:
    #     M *= M_totalmass / M.sum()
    # fM = density.real_to_fspace_with_oversampling(M, oversampling_factor=oversampling_factor)
    # fM = fM.real ** 2 + fM.imag ** 2
    # mrc.writeMRC("particle/EMD-6044-cropped_fM_totalmass_%d_oversampling_%d.mrc" % (M_totalmass, oversampling_factor), fM, psz=psize)
    # exit()
    
    fM = mrc.readMRC('particle/EMD-6044-cropped_fM_totalmass_2000000_oversampling_6.mrc')

    N = fM.shape[0]
    TtoF = sincint.gentrunctofull(N=N, rad=rad, beamstop_rad=beamstop_rad)

    theta = np.arange(0, 2*np.pi, 2*np.pi/60)
    degree_R, resolution_R = SK97Quadrature.compute_degree(N, 0.3, 1.0)
    dirs, weights = SK97Quadrature.get_quad_points(degree_R, None)
    Rs = np.vstack([geometry.rotmat3D_dir(vec)[:, 0:2].reshape((1, 3, 2)) for vec in dirs])

    N_R = dirs.shape[0]
    N_I = theta.shape[0]
    N_T = TtoF.shape[1]

    # generate slicing operators
    dir_slice_interp = {'projdirs': dirs, 'N': N, 'kern': 'lanczos', 'kernsize': 6, 
                        'projdirtype': 'dirs', 'onlyRs': True, 'rad': rad, 'sym': None}  # 'zeropad': 0, 'dopremult': True
    R_slice_interp = {'projdirs': Rs, 'N': N, 'kern': 'lanczos', 'kernsize': 6,
                    'projdirtype': 'rots', 'onlyRs': True, 'rad': rad, 'sym': None}  # 'zeropad': 0, 'dopremult': True
    inplane_interp = {'thetas': theta, 'N': N, 'kern': 'lanczos', 'kernsize': 6,
                    'onlyRs': True, 'rad': rad}  # 'zeropad': 1, 'dopremult': True
    inplane_interp['N_src'] = N  # zp_N

    slice_ops = cryoops.compute_projection_matrix(**dir_slice_interp)
    inplane_ops = cryoops.compute_inplanerot_matrix(**inplane_interp)

    # generate slices and inplane-rotated slices
    slices_sampled = cryoem.getslices_interp(fM, slice_ops, rad, beamstop_rad=beamstop_rad).reshape((N_R, N_T))
    curr_img = TtoF.dot(slices_sampled[0]).reshape(N, N)
    rotd_sampled = cryoem.getslices_interp(curr_img, inplane_ops, rad, beamstop_rad=beamstop_rad).reshape((N_I, N_T))

    ## plot figures
    fig, axes = plt.subplots(3, 4, figsize=(9.6, 7.2))
    for i, ax in enumerate(axes.flatten()):
        img = TtoF.dot(slices_sampled[i]).reshape(N, N)
        ax.imshow(img, origin='lower')
    fig.suptitle('slices_sampled')

    fig, axes = plt.subplots(3, 4, figsize=(9.6, 7.2))
    for i, ax in enumerate(axes.flatten()):
        img = TtoF.dot(slices_sampled[i]).reshape(N, N)
        ax.imshow(img, origin='lower')
    fig.suptitle('rotd_sampled')
    plt.show()


##
def sparse(mode, totalmass, use_ac, freq_start, freq_end):
    # 1000000
    dataset_dir = 'data/EMD6044_xfel_5000_totalmass_{}_oversampling_6'.format(str(totalmass).zfill(7))
    data_params = {
        'dataset_name': "1AON",
        'inpath': os.path.join(dataset_dir, 'imgdata.mrc'),
        'gtpath': os.path.join(dataset_dir, 'ctf_gt.par'),
        'ctfpath': os.path.join(dataset_dir, 'defocus.txt'),
        'microscope_params': {'akv': 200, 'wgh': 0.07, 'cs': 2.0},
        'resolution': 3.0 * 6,
        'pixel_size': 3.0 * 6,
        # 'sigma': 'noise_std',
        # 'sigma_out': 'data_std',
        # 'minisize': 150,
        
        'num_images': 200,
        # 'euler_angles': None,
    }

    # euler_angles = []
    # with open(data_params['gtpath']) as par:
    #     par.readline()
    #     # 'C                 PHI      THETA        PSI        SHX        SHY       FILM        DF1        DF2     ANGAST'
    #     while True:
    #         try:
    #             line = par.readline().split()
    #             euler_angles.append([float(line[1]), float(line[2]), float(line[3])])
    #         except Exception:
    #             break
    # data_params['euler_angles'] = np.deg2rad(np.asarray(euler_angles))[0:data_params['num_images']]

    # from scripts.analyze_sparse import get_quadrature
    # quad_domain_R, _ = get_quadrature(N=124, const_rad=0.8)
    # euler_angles = geometry.genEA(quad_domain_R.dirs)
    # rand_idxs = np.arange(euler_angles.shape[0])
    # np.random.shuffle(rand_idxs)
    # data_params['euler_angles'] = euler_angles[rand_idxs]

    refined_model = mrc.readMRC('particle/EMD-6044-cropped_fM_totalmass_{}_oversampling_6.mrc'.format(str(totalmass).zfill(7)))

    cryodata = SimpleDataset(None, data_params, None)

    if use_ac == 1:
        use_angular_correlation = True
    elif use_ac == 0:
        use_angular_correlation = False
    else:
        raise NotImplementedError

    cparams = {'max_frequency': None, 'beamstop_freq': 0.01, 'learn_like_envelope_bfactor': 500}
    # print("Using angular correlation patterns:", use_angular_correlation)
    sk = SimpleKernel(cryodata, use_angular_correlation=use_angular_correlation)

    # if mode == 1:
        # for freq in np.arange(0.015, 0.055, 0.005):
        # for freq in np.arange(0.025, 0.055, 0.005):
        # for freq in np.arange(0.035, 0.055, 0.005):
    #         cparams['max_frequency'] = freq
    #         sk.set_data(refined_model, cparams)

    #         # for i in range(cryodata.num_images):
    #         for i in range(500):
    #             tic = time.time()
    #             workspace = sk.worker(i)
    #             print("idx: {}, time per worker: {}".format(i, time.time()-tic))
    #             # sk.plot_distribution(workspace, sk.quad_domain_R, sk.quad_domain_I,
    #             #                     correct_ea=cryodata.euler_angles[i], lognorm=False)
        
    #         with open('sparse/mode-{}-totalmass-{}-use_ac-{}-freq-{}-workspace.pkl'.format(mode, totalmass, use_angular_correlation, freq), 'wb') as pkl:
    #             pickle.dump(sk.cached_workspace, pkl)

    # elif mode == 2:
        # for freq_start, freq_end in zip(np.arange(0.010, 0.05, 0.005), np.arange(0.015, 0.055, 0.005)):
        # for freq_start, freq_end in zip(np.arange(0.020, 0.05, 0.005), np.arange(0.025, 0.055, 0.005)):
        # for freq_start, freq_end in zip(np.arange(0.030, 0.05, 0.005), np.arange(0.035, 0.055, 0.005)):
    cparams['max_frequency'] = freq_end
    cparams['beamstop_freq'] = freq_start
    sk.set_data(refined_model, cparams)

    # sk.concurrent_worker(range(200))
    # sk.plot_distribution(sk.cached_cphi[1], sk.quad_domain_R, sk.quad_domain_I,
    #                      correct_ea=cryodata.euler_angles[1], lognorm=False)

    timer = list()

    for i in range(4):
        tic = time.time()
        _ = sk.worker(i)
        toc = time.time() - tic
        # print("idx: {}, time per worker: {}".format(i, toc))
        timer.append(toc)
        # workspace = sk.worker(i)
        # sk.plot_distribution(workspace, sk.quad_domain_R, sk.quad_domain_I,
        #                     correct_ea=cryodata.euler_angles[i], lognorm=False)
    
    print( "freq_start-{:.3f}-freq_end-{:.3f}-time-{}".format(freq_start, freq_end, sum(timer[2:]) / len(timer[2:])))

    # dir_name = 'sparse-mode1-const_rad-0.6-non-entropy-new_ACI'
    # if not os.path.exists(dir_name):
    #     os.makedirs(dir_name)

    # with open(dir_name+'/mode-{}-totalmass-{}-use_ac-{}-freq_start-{:.3f}-freq_end-{:.3f}-workspace.pkl'.format(mode, str(totalmass).zfill(7), use_angular_correlation, freq_start, freq_end), 'wb') as pkl:
    #     pickle.dump(sk.cached_cphi, pkl)

    # else:
    #     raise NotImplementedError

if __name__ == '__main__':
    # plot_figures()

    # dataset_loading_test()

    # mode = int(sys.argv[1])
    # totalmass = sys.argv[2]
    # use_ac = int(sys.argv[3])

    mode = 1
    # totalmass_list = [500000, 1000000, 1500000]
    totalmass_list = [1000000]
    use_ac = 1


    print('mode-{}-totalmass-{}-use_ac-{}.'.format(mode, totalmass_list, use_ac))

    for totalmass in totalmass_list:
        if mode == 2:
            # for freq_start, freq_end in zip(np.arange(0.030, 0.05, 0.005), np.arange(0.035, 0.055, 0.005)):
            for freq_start, freq_end in zip(np.arange(0.010, 0.50, 0.005), np.arange(0.015, 0.055, 0.005)):
                sparse(mode, totalmass, use_ac, freq_start, freq_end)
                gc.collect()
        elif mode == 1:
            for freq in np.arange(0.015, 0.055, 0.005):
                sparse(mode, totalmass, use_ac, 0.010, freq)
                gc.collect()


    mode = 1
    # totalmass_list = [500000, 1000000, 1500000]
    totalmass_list = [1000000]
    use_ac = 0

    print('mode-{}-totalmass-{}-use_ac-{}.'.format(mode, totalmass_list, use_ac))

    for totalmass in totalmass_list:
        if mode == 2:
            # for freq_start, freq_end in zip(np.arange(0.030, 0.05, 0.005), np.arange(0.035, 0.055, 0.005)):
            for freq_start, freq_end in zip(np.arange(0.010, 0.50, 0.005), np.arange(0.015, 0.055, 0.005)):
                sparse(mode, totalmass, use_ac, freq_start, freq_end)
                gc.collect()
        elif mode == 1:
            for freq in np.arange(0.015, 0.055, 0.005):
                sparse(mode, totalmass, use_ac, 0.010, freq)
                gc.collect()
