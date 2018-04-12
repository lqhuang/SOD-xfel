from __future__ import print_function, division

import os, sys, glob, time
sys.path.append(os.path.dirname(sys.path[0]))


from scripts.pattern_stat import analyse_mrcs_stack 

try:
    import cPickle as pickle  # python 2
except ImportError:
    import pickle  # python 3

import numpy as np
from matplotlib import pyplot as plt

import geometry
import quadrature

PKL_DIR = 'sparse-mode1-const_rad-0.6-non-entropy'
CONST_RAD = 0.6

def get_euler_angles(dataset_dir):
    gtpath =  os.path.join(dataset_dir, 'ctf_gt.par')
    euler_angles = []
    with open(gtpath) as par:
        par.readline()
        # 'C                 PHI      THETA        PSI        SHX        SHY       FILM        DF1        DF2     ANGAST'
        while True:
            try:
                line = par.readline().split()
                euler_angles.append([float(line[1]), float(line[2]), float(line[3])])
            except Exception:
                break
    eas = np.deg2rad(np.asarray(euler_angles))
    return eas


def get_quadrature(N=124, const_rad=0.8):
    
    is_sym = None
    slice_params = {'quad_type': 'sk97'}

    tic = time.time()
    # set slicing quadrature
    usFactor_R = 1.0
    quad_R = quadrature.quad_schemes[('dir', slice_params['quad_type'])]
    degree_R, resolution_R = quad_R.compute_degree(N, const_rad, usFactor_R)
    # print(np.rad2deg(resolution_R))
    slice_quad = {}
    slice_quad['resolution'] = max(0.5*quadrature.compute_max_angle(N, const_rad), resolution_R)
    # slice_quad['resolution'] = np.rad2deg(resolution_R)
    slice_quad['dir'], slice_quad['W'] = quad_R.get_quad_points(degree_R, is_sym)
    slice_quad = slice_quad
    quad_domain_R = quadrature.FixedSphereDomain(slice_quad['dir'], slice_quad['resolution'], sym=is_sym)
    N_R = len(quad_domain_R)
    print("  Slice Ops: %d, resolution: %.2f degree, generated in: %.4f seconds" \
                % (N_R, np.rad2deg(quad_domain_R.resolution), time.time()-tic))

    tic = time.time()
    # set inplane quadrature
    usFactor_I = 1.0
    maxAngle = quadrature.compute_max_angle(N, const_rad, usFactor_I)
    degree_I = np.uint32(np.ceil(2.0 * np.pi / maxAngle))
    resolution_I = max(0.5*quadrature.compute_max_angle(N, const_rad), 2.0*np.pi / degree_I)
    inplane_quad = {}
    inplane_quad['resolution'] = resolution_I
    inplane_quad['thetas'] = np.linspace(0, 2.0*np.pi, degree_I, endpoint=False)
    # inplane_quad['thetas'] += inplane_quad['thetas'][1]/2.0
    inplane_quad['W'] = np.require((2.0*np.pi/float(degree_I))*np.ones((degree_I,)), dtype=np.float32)
    inplane_quad = inplane_quad
    quad_domain_I = quadrature.FixedCircleDomain(inplane_quad['thetas'],
                                                    inplane_quad['resolution'])
    # generate inplane operators
    N_I = len(quad_domain_I)
    print("  Inplane Ops: %d, resolution: %.2f degree, generated in: %.4f seconds." \
        % (N_I, np.rad2deg(quad_domain_I.resolution), time.time()-tic))

    return quad_domain_R, quad_domain_I


def plot_rmsd(cached_cphi, euler_angles, quad_domain_R, ac=0):
    processed_idxs = cached_cphi.keys()
    quad_domain_R = quad_domain_R

    num_idxs = len(processed_idxs)
    top1_rmsd = np.zeros(num_idxs)
    topN_rmsd = np.zeros(num_idxs)
    topN_C_rmsd = np.zeros(num_idxs)
    cutoff_R = 5
    topN_weight = np.exp(-np.arange(cutoff_R) / 2)
    topN_weight /= topN_weight.sum()

    for i, idx in enumerate(processed_idxs):
        ea = euler_angles[idx][0:2]
        tiled_ea = np.tile(ea, (cutoff_R, 1))
        cphi_R = cached_cphi[idx]['cphi_R']
        sorted_indices_R = (-cphi_R).argsort()
        potential_R = quad_domain_R.dirs[sorted_indices_R[0:cutoff_R]]
        eas_of_dirs = geometry.genEA(potential_R)[:, 0:2]
        if ac == 0:
            top1_rmsd[i] = np.sqrt(((ea[0:2] - eas_of_dirs[0])**2).mean())
            topN_rmsd[i] = (np.sqrt(((tiled_ea - eas_of_dirs) ** 2).mean(axis=1)) * topN_weight).sum()
            topN_C_rmsd[i] = ( np.sqrt(((tiled_ea - eas_of_dirs) ** 2).mean(axis=1)) ).min()
        else:
            top1_dir = eas_of_dirs[0]
            top1_dir_chiral = np.pi + np.asarray([+1.0, -1.0]) * eas_of_dirs[0]  # 手性对称位置的坐标
            top1_dir_chiral[0] -= (top1_dir_chiral[0] > 2*np.pi) * 2*np.pi  # 超过 2pi 的地方减去 2pi
            
            first = np.sqrt(( (ea[0:2] - top1_dir)**2 ).mean())
            first_chiral = np.sqrt( ( (ea[0:2] - top1_dir_chiral )**2 ).mean() )
            top1_rmsd[i] = np.min( [first, first_chiral] )

            topN_rmsd[i] = (np.sqrt(((tiled_ea - eas_of_dirs) ** 2).mean(axis=1)) * topN_weight).sum()

            eas_of_dirs_chiral = ( np.pi + np.tile([+1.0, -1.0], (cutoff_R, 1)) * eas_of_dirs )
            eas_of_dirs_chiral[:, 0] -= (eas_of_dirs_chiral[:, 0] > 2*np.pi) * 2*np.pi

            topN_C = np.sqrt( ( (tiled_ea - eas_of_dirs) ** 2 ).mean(axis=1) )
            topN_C_chiral = np.sqrt( ( (tiled_ea - eas_of_dirs_chiral) ** 2 ).mean(axis=1) )

            topN_C_rmsd[i] = ( np.minimum(topN_C, topN_C_chiral) ).min()

    print("Slicing quadrature scheme, resolution {}, num of points {}".format(
        quad_domain_R.resolution, len(quad_domain_R.dirs)))
    print("Top 1 RMSD:", top1_rmsd.mean())
    print("Top {} RMSD: {}".format(cutoff_R, topN_rmsd.mean()))
    print("Top {}:RMSD {}".format(cutoff_R, topN_C_rmsd.mean()))

    # plt.hist(np.rad2deg(topN_C_rmsd))
    # plt.show()

    return top1_rmsd.mean(), topN_rmsd.mean(), topN_C_rmsd.mean()

def plot_figures():

    quad_domain_R, quad_domain_I = get_quadrature(const_rad=CONST_RAD)

    pkl_list_05_0 = glob.glob(os.path.join(PKL_DIR,
        'mode-?-totalmass-0500000-use_ac-False-freq_start-*-freq_end-*-workspace.pkl'))
    pkl_list_10_0 = glob.glob(os.path.join(PKL_DIR,
        'mode-?-totalmass-1000000-use_ac-False-freq_start-*-freq_end-*-workspace.pkl'))
    pkl_list_15_0 = glob.glob(os.path.join(PKL_DIR,
        'mode-?-totalmass-1500000-use_ac-False-freq_start-*-freq_end-*-workspace.pkl'))
    pkl_list_20_0 = glob.glob(os.path.join(PKL_DIR,
        'mode-?-totalmass-2000000-use_ac-False-freq_start-*-freq_end-*-workspace.pkl'))
    pkl_list_05_1 = glob.glob(os.path.join(PKL_DIR,
        'mode-?-totalmass-0500000-use_ac-True-freq_start-*-freq_end-*-workspace.pkl'))
    pkl_list_10_1 = glob.glob(os.path.join(PKL_DIR,
        'mode-?-totalmass-1000000-use_ac-True-freq_start-*-freq_end-*-workspace.pkl'))
    pkl_list_15_1 = glob.glob(os.path.join(PKL_DIR,
        'mode-?-totalmass-1500000-use_ac-True-freq_start-*-freq_end-*-workspace.pkl'))
    pkl_list_20_1 = glob.glob(os.path.join(PKL_DIR,
        'mode-?-totalmass-2000000-use_ac-True-freq_start-*-freq_end-*-workspace.pkl'))

    data_dir_05 = 'data/EMD6044_xfel_5000_totalmass_0500000_oversampling_6'
    data_dir_10 = 'data/EMD6044_xfel_5000_totalmass_1000000_oversampling_6'
    data_dir_15 = 'data/EMD6044_xfel_5000_totalmass_1500000_oversampling_6'
    data_dir_20 = 'data/EMD6044_xfel_5000_totalmass_2000000_oversampling_6'

    euler_angles_05 = get_euler_angles(data_dir_05)
    euler_angles_10 = get_euler_angles(data_dir_10)
    euler_angles_15 = get_euler_angles(data_dir_15)
    euler_angles_20 = get_euler_angles(data_dir_20)

    top1_rmsd_05_0, _, topN_C_rmsd_05_0 = plot_error(pkl_list_05_0, euler_angles_05, quad_domain_R,  5, 0)
    top1_rmsd_05_1, _, topN_C_rmsd_05_1 = plot_error(pkl_list_05_1, euler_angles_05, quad_domain_R,  5, 1)

    top1_rmsd_10_0, _, topN_C_rmsd_10_0 = plot_error(pkl_list_10_0, euler_angles_10, quad_domain_R, 10, 0)
    top1_rmsd_10_1, _, topN_C_rmsd_10_1 = plot_error(pkl_list_10_1, euler_angles_10, quad_domain_R, 10, 1)

    top1_rmsd_15_0, _, topN_C_rmsd_15_0 = plot_error(pkl_list_15_0, euler_angles_15, quad_domain_R, 15, 0)
    top1_rmsd_15_1, _, topN_C_rmsd_15_1 = plot_error(pkl_list_15_1, euler_angles_15, quad_domain_R, 15, 1)


    # freqs = np.arange(0.01, 0.055, 0.005)
    freqs = np.arange(0.015, 0.060, 0.005)

    fig, ax0 = plt.subplots(figsize=(9.6, 4.8))
    ax0.plot(freqs[:-1], np.rad2deg(topN_C_rmsd_05_1), '-^', label='Totalmass 500,000')
    ax0.plot(freqs[:-1], np.rad2deg(topN_C_rmsd_10_1), '-^', label='Totalmass 1,000,000')
    ax0.plot(freqs[:-1], np.rad2deg(topN_C_rmsd_15_1), '-^', label='Totalmass 1,500,000')
    ax0.legend(loc=2,frameon=False)
    ax0.set_ylim([0, 12])
    # ax0.set_ylim([0, 70])
    ax0.set_title('Angular Correlation Orientation RMSD for the closest one of Top 5 determination')
    ax0.set_xlabel('Frequency Domain (1/angstrom)')
    ax0.set_ylabel('Root Mean Square Deviation (Degree)')
    fig.tight_layout()
    plt.savefig('ori_topN_C_error_compare_ac', dpi=300, bbox_inches='tight')

    fig, ax0 = plt.subplots(figsize=(9.6, 4.8))
    ax0.plot(freqs[:-1], np.rad2deg(topN_C_rmsd_05_0), '-o', label='Totalmass 500,000')
    ax0.plot(freqs[:-1], np.rad2deg(topN_C_rmsd_10_0), '-o', label='Totalmass 1,000,000')
    ax0.plot(freqs[:-1], np.rad2deg(topN_C_rmsd_15_0), '-o', label='Totalmass 1,500,000')
    ax0.legend(loc=2,frameon=False)
    ax0.set_ylim([0, 12])
    # ax0.set_ylim([0, 70])
    ax0.set_title('Orientation RMSD for the closest one of Top 5 determination')
    ax0.set_xlabel('Frequency Domain (1/angstrom)')
    ax0.set_ylabel('Root Mean Square Deviation (Degree)')
    fig.tight_layout()
    plt.savefig('ori_topN_C_error_compare', dpi=300, bbox_inches='tight')

    fig, ax0 = plt.subplots(figsize=(9.6, 4.8))
    ax0.plot(freqs[:-1], np.rad2deg(topN_C_rmsd_05_0), '-o', label='RMSD for the closest one of Top 5 - Without Angular Correlation')
    ax0.plot(freqs[:-1], np.rad2deg(topN_C_rmsd_05_1), '-^r', label='RMSD for the closest one of Top 5 - With Angular Correlation')
    ax0.legend(loc=2,frameon=False)
    ax0.set_ylim([0, 12])
    # ax0.set_ylim([0, 70])
    ax0.set_title('Orientation Error (EMD6044 totalmass {:,} oversampling 6)'.format(int(5 * 1e5)))
    ax0.set_xlabel('Frequency Domain (1/angstrom)')
    ax0.set_ylabel('Root Mean Square Deviation (Degree)')

    fig.tight_layout()
    plt.savefig('ori_error_compare_%s' % ('05'), dpi=300, bbox_inches='tight')
    
    fig, ax0 = plt.subplots(figsize=(9.6, 4.8))
    ax0.plot(freqs[:-1], np.rad2deg(topN_C_rmsd_10_0), '-o', label='RMSD for the closest one of Top 5 - Without Angular Correlation')
    ax0.plot(freqs[:-1], np.rad2deg(topN_C_rmsd_10_1), '-^r', label='RMSD for the closest one of Top 5 - With Angular Correlation')
    ax0.legend(loc=2,frameon=False)
    ax0.set_ylim([0, 12])
    # ax0.set_ylim([0, 70])
    ax0.set_title('Orientation Error (EMD6044 totalmass {:,} oversampling 6)'.format(int(10 * 1e5)))
    ax0.set_xlabel('Frequency Domain (1/angstrom)')
    ax0.set_ylabel('Root Mean Square Deviation (Degree)')

    fig.tight_layout()
    plt.savefig('ori_error_compare_%d' % (10,), dpi=300, bbox_inches='tight')

    fig, ax0 = plt.subplots(figsize=(9.6, 4.8))
    ax0.plot(freqs[:-1], np.rad2deg(topN_C_rmsd_15_0), '-o', label='RMSD for the closest one of Top 5 - Without Angular Correlation')
    ax0.plot(freqs[:-1], np.rad2deg(topN_C_rmsd_15_1), '-^r', label='RMSD for the closest one of Top 5 - With Angular Correlation')
    ax0.legend(loc=2,frameon=False)
    ax0.set_ylim([0, 12])
    # ax0.set_ylim([0, 70])
    ax0.set_title('Orientation Error (EMD6044 totalmass {:,} oversampling 6)'.format(int(15 * 1e5)))
    ax0.set_xlabel('Frequency Domain (1/angstrom)')
    ax0.set_ylabel('Root Mean Square Deviation (Degree)')

    # ax1 = ax0.twinx()
    # ax1.plot(freqs[:-1], ratio_15, 'ro-')
    # ax1.set_ylabel('Ratio of Intensity greater than 1', color='r')
    # ax1.set_ylim([0, 1.1])
    # ax1.tick_params('y', colors='r')

    fig.tight_layout()
    plt.savefig('ori_error_compare_%d' % (15,), dpi=300, bbox_inches='tight')

    # plt.show()


def plot_error(pkl_list, euler_angles, quad_domain_R, mass, ac):
    top1_rmsd = np.zeros_like(pkl_list, dtype=float)
    topN_rmsd = np.zeros_like(pkl_list, dtype=float)
    topN_C_rmsd = np.zeros_like(pkl_list, dtype=float)

    for i, pkl_file in enumerate(pkl_list):
        with open(pkl_file, 'rb') as pkl:
            cached_cphi = pickle.load(pkl)
        
        top1_rmsd[i], topN_rmsd[i], topN_C_rmsd[i] = plot_rmsd(cached_cphi, euler_angles, quad_domain_R, ac)

    # freqs = np.arange(0.01, 0.055, 0.005)
    freqs = np.arange(0.015, 0.060, 0.005)

    if ac == 0:
        pat = '-o'
    else:
        pat = '-^'

    fig, ax0 = plt.subplots(figsize=(9.6, 4.8))
    ax0.plot(freqs[:-1], np.rad2deg(top1_rmsd), pat, label='Top 1 RMSD')
    ax0.plot(freqs[:-1], np.rad2deg(topN_rmsd), pat, label='Top 5 average RMSD')
    ax0.plot(freqs[:-1], np.rad2deg(topN_C_rmsd), pat, label='RMSD for the closest one of Top 5')
    ax0.legend(frameon=False)
    ax0.set_title('Orientation Error (EMD6044 totalmass %d00000 oversampling 6)' % mass)
    ax0.set_xlabel('Frequency Domain (1/angstrom)')
    ax0.set_ylabel('Root Mean Square Deviation (Degree)')
    fig.tight_layout()
    plt.savefig('ori_error_%s_%d' % (str(mass).zfill(2), int(ac)), dpi=300, bbox_inches='tight')
    # plt.show()

    return top1_rmsd, topN_rmsd, topN_C_rmsd


if __name__ == "__main__":
    plot_figures()