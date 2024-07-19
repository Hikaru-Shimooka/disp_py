#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt
from scipy.special import wofz
from multiprocessing import Pool
from tqdm import tqdm
import os

import input_ia as input


def plasma_dispersion_func(zeta_j):
    """
    Calculate plasma dispersion function
    """
    zeta = 1j * np.sqrt(np.pi) * wofz(zeta_j)
    return zeta

def plasma_dispersion_func_deriv(zeta_j):
    """
    Calculate derivative of plasma dispersion function

    input : complex
    """
    zeta = plasma_dispersion_func(zeta_j)
    return -2 * (1 + zeta_j*zeta)

def plasma_dispersion_func_deriv2(zeta_j):
    """
    Calculate 2nd derivative of plasma dispersion function

    input : complex
    """
    zeta = plasma_dispersion_func(zeta_j)
    return -2 * (zeta - 2*zeta_j - 2*(zeta_j**2)*zeta)


def process_cal_disp_s1(params):

    """
    Dispersion Relation Calculation for ion acoustic wave

    --------------------------------------------------
    Parameters

    # Particles
    vj : Thermal speed of the j-th component
    kj: Debye wavenumber of the j-th component


    # Calculation Area
    k_min : Minimum wavenumber
    k_max : Maximum wavenumber
    k_num : Total number of wavenumbers

    wr_min : Minimum real part of omega
    wr_max : Maximum real part of omega
    wr_num : Total number of real omega values

    wi_min : Minimum imaginary part of omega
    wi_max : Maximum imaginary part of omega
    wi_num : Total number of imaginary omega values

    eps : Threshold value below which a value is treated as zero



    # Return
    ans = [k, wr, wi]

    --------------------------------------------------
    """

    k, ve, vi, ke, ki,\
        wr_min, wr_max, wr_num, wi_min, wi_max, wi_num, eps = params

    ans = []

    wr = np.linspace(wr_min, wr_max, wr_num)
    wi = np.linspace(wi_min, wi_max, wi_num)
    wr, wi = np.meshgrid(wr, wi)
    z_e = (wr + 1j*wi)/(np.sqrt(2)*k*ve)
    z_i = (wr + 1j*wi)/(np.sqrt(2)*k*vi)
    zeta_e = plasma_dispersion_func_deriv(z_e)
    zeta_i = plasma_dispersion_func_deriv(z_i)
    zeta_e[np.isnan(zeta_e)] = 0
    zeta_e[np.isinf(zeta_e)] = 0
    zeta_e = np.where(np.abs(zeta_e) >= 1e3, 0, zeta_e)
    zeta_i[np.isnan(zeta_i)] = 0
    zeta_i[np.isinf(zeta_i)] = 0
    zeta_i = np.where(np.abs(zeta_i) >= 1e3, 0, zeta_i)
    # disp = 1 - ((ke**2)/(2*(k**2)))*zeta_e - ((ki**2)/(2*(k**2)))*zeta_i
    disp = 2*(k**2) - (ke**2)*zeta_e - (ki**2)*zeta_i
    dr = disp.real
    di = disp.imag

    dr_idx = []
    for li in range(dr.shape[0]):
        for c in range(dr.shape[1]-1):
            if dr[li, c]*dr[li, c+1] < 0:
                dr_idx.append([li, c])
                dr_idx.append([li, c+1])

    for c in range(dr.shape[1]):
        for li in range(dr.shape[0]-1):
            if dr[li, c]*dr[li+1, c] < 0:
                dr_idx.append([li, c])
                dr_idx.append([li+1, c])

    di_idx = []
    for li in range(di.shape[0]):
        for c in range(di.shape[1]-1):
            if di[li, c]*di[li, c+1] < 0:
                di_idx.append([li, c])
                di_idx.append([li, c+1])

    for c in range(di.shape[1]):
        for li in range(di.shape[0]-1):
            if di[li, c]*di[li+1, c] < 0:
                di_idx.append([li, c])
                di_idx.append([li+1, c])

    ans_tmp = []

    for ele in dr_idx:
        #print(ele)
        if ele in di_idx:
            line = ele[0]
            col = ele[1]
            if (abs(dr[line, col])<eps and abs(di[line, col])<eps):
                ans_tmp.append([wr[line, col], wi[line, col]])

    # Newton method
    if len(ans_tmp) != 0:
        for w in ans_tmp:
            # print(zeta_j)
            omega = w[0]+w[1]*1j
            for it2 in range(int(1e6)):
                # print('it2', it2)
                z_e = omega/(np.sqrt(2)*k*ve)
                z_i = omega/(np.sqrt(2)*k*vi)
                zeta_e = plasma_dispersion_func_deriv(z_e)
                zeta_i = plasma_dispersion_func_deriv(z_i)
                zeta_e2 = plasma_dispersion_func_deriv2(z_e)
                zeta_i2 = plasma_dispersion_func_deriv2(z_i)
                disp = 2*(k**2) - (ke**2)*zeta_e - (ki**2)*zeta_i
                
                dr = disp.real
                di = disp.imag
                
                if (abs(dr)<1e-5 and abs(di)<1e-5):
                    ans.append([k, omega.real, omega.imag])
                    break
                # print('it2', it2, dr, di)
                disp_w = -((ke**2)/(2*(k**2)))*(1/(np.sqrt(2)*k*ve))*zeta_e2 \
                         -((ki**2)/(2*(k**2)))*(1/(np.sqrt(2)*k*vi))*zeta_i2
                omega = omega - disp/disp_w

    ans = np.array(ans)
    return ans


def cal_disp_s1_p(
                  ve, vi,
                  ke, ki,
                  k_min, k_max, k_num,
                  wr_min, wr_max, wr_num, wi_min,
                  wi_max, wi_num,
                  eps
                  ):
    """
    Calculate dispersion relation each k
    """
    print('Start calculating ...')
    k_list = np.linspace(k_min, k_max, k_num)

    params_list = \
        [
         (k,
          ve, vi, ke, ki,
          wr_min, wr_max, wr_num,
          wi_min, wi_max, wi_num,
          eps)
         for k in k_list[1:]
        ]

    with Pool() as pool:
        results = list(
                       tqdm(pool.imap(process_cal_disp_s1, params_list),
                            total=len(params_list)))

    ans = [item for sublist in results for item in sublist]
    ans = np.array(ans)
    print('Finish')
    return ans


if __name__ == '__main__':
    ve = input.ve
    vi = input.vi
    ke = input.ke
    ki = input.ki
    k_min = input.k_min
    k_max = input.k_max
    k_num = input.k_num
    wr_min = input.wr_min
    wr_max = input.wr_max
    wr_num = input.wr_num
    wi_min = input.wi_min
    wi_max = input.wi_max
    wi_num = input.wi_num
    Te = input.Te
    Ti = input.Ti
    mi = input.mi
    eps = input.eps

    wpi = input.wpi

    ans = cal_disp_s1_p(
                        ve, vi,
                        ke, ki,
                        k_min, k_max, k_num,
                        wr_min, wr_max, wr_num,
                        wi_min, wi_max, wi_num,
                        eps
                        )

    os.makedirs('../data_newton', exist_ok=True)
    os.makedirs('../fig_newton', exist_ok=True)

    np.savetxt('../data_newton/ia_te_eq_{}ti_kwrwi.txt'.format(int(Te/Ti)), ans)

    wr_1 = []
    wi_1 = []
    wr_2 = []
    wi_2 = []

    k_list = np.linspace(k_min, k_max, k_num)
    cs = np.sqrt((Te+3*Ti)/mi)

    plt.rcParams["font.size"] = 14

    # zfunc = np.loadtxt('../fig2.2/aaa.dat')

    fig = plt.figure(figsize=(6, 10), dpi=300)
    gs = fig.add_gridspec(2, 1, height_ratios=[2, 1], hspace=0)

    ax1 = fig.add_subplot(gs[0, 0])
    ax1.scatter(ans[:, 0]/ki, ans[:, 1]/wpi,
                c='g', marker='o', s=5, label='Numerical')
    ax1.plot(k_list[1:]/ki, cs*k_list[1:], c='orange', label='Sound speed')
    # ax1.plot(zfunc[:, 0], zfunc[:, 1], c='c', label='zfunc.f90')
    ax1.set_ylabel(r'$\frac{\omega_r}{\omega_i}$',
                   labelpad=10, rotation=0, va='center', fontsize=16)
    ax1.set_ylim(0, 2)
    ax1.set_xlim(0, 1)
    ax1.tick_params(axis='x', which='both',
                    bottom=True, top=False, labelbottom=False, direction='in')
    ax1.legend()

    ax2 = fig.add_subplot(gs[1, 0], sharex=ax1)
    ax2.scatter(ans[:, 0]/ki, ans[:, 2]/wpi,
                c='g', marker='o', s=5, label='Numerical')
    # ax2.scatter(zfunc[:, 0], zfunc[:, 2], c='c', s=5, label='zfunc.f90')
    ax2.set_xlabel(r'$\frac{k}{k_i}$', fontsize=16)
    ax2.set_ylabel(r'$\frac{\gamma}{\omega_i}$',
                   labelpad=10, rotation=0, va='center', fontsize=16)
    ax2.set_ylim(-1, 0)
    ax2.set_xlim(0, 1)
    ax2.set_yticks([-1.0, -0.8, -0.6, -0.4, -0.2])
    ax2.set_yticklabels([-1.0, -0.8, -0.6, -0.4, -0.2])
    ax2.tick_params(bottom=True, top=True, labelbottom=True, direction='in')
    ax2.legend()

    plt.savefig('../fig_newton/ia_disp_te_eq_{}ti.png'.format(int(Te/Ti)),
                bbox_inches="tight")
    plt.close()
