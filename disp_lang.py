#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt
from scipy.special import wofz
from multiprocessing import Pool
from tqdm import tqdm
import os

import input_lang as input


def plasma_dispersion_func_deriv(zeta_j):
    """
    Calculate derivative of plasma dispersion function

    input : complex
    """
    zeta = 1j * np.sqrt(np.pi) * wofz(zeta_j)
    return -2 * (1 + zeta_j*zeta)


def process_cal_disp_s1(params):

    """
    Dispersion Relation Calculation for One Species

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

    k, vj, kj, wr_min, wr_max, wr_num, wi_min, wi_max, wi_num, eps = params

    ans = []

    wr = np.linspace(wr_min, wr_max, wr_num)
    wi = np.linspace(wi_min, wi_max, wi_num)
    wr, wi = np.meshgrid(wr, wi)
    z_j = (wr + 1j*wi)/(np.sqrt(2)*k*vj)
    zeta_d = plasma_dispersion_func_deriv(z_j)
    zeta_d[np.isnan(zeta_d)] = 0
    zeta_d[np.isinf(zeta_d)] = 0
    zeta_d = np.where(np.abs(zeta_d) >= 1e4, 0, zeta_d)
    disp = 1 - ((kj**2)/(2*(k**2)))*zeta_d
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

    for ele in dr_idx:
        if ele in di_idx:
            line = ele[0]
            col = ele[1]
            if (abs(dr[line, col]) < eps and abs(di[line, col]) < eps):
                ans.append([k, wr[line, col], wi[line, col]])

    ans = np.array(ans)
    return ans


def cal_disp_s1_p(
                  vj, kj,
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
         (k, vj, kj, wr_min, wr_max, wr_num, wi_min, wi_max, wi_num, eps)
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
    vj = input.vj
    kj = input.kj
    k_min = input.k_min
    k_max = input.k_max
    k_num = input.k_num
    wr_min = input.wr_min
    wr_max = input.wr_max
    wr_num = input.wr_num
    wi_min = input.wi_min
    wi_max = input.wi_max
    wi_num = input.wi_num
    eps = input.eps

    wpe = input.wpe
    ve = vj

    ans = cal_disp_s1_p(
                        vj, kj,
                        k_min, k_max, k_num,
                        wr_min, wr_max, wr_num,
                        wi_min, wi_max, wi_num,
                        eps
                        )

    os.makedirs('data', exist_ok=True)
    os.makedirs('fig', exist_ok=True)

    np.savetxt('data/lang_kwrwi.txt', ans)

    wr_1 = []
    wi_1 = []
    wr_2 = []
    wi_2 = []

    k_list = np.linspace(k_min, k_max, k_num)

    for k in k_list[1:]:
        t1 = 1
        t2 = 0
        t3 = -(wpe**2)-3*(k**2)*(ve**2)

        ana_ans = np.roots([t1, t2, t3])
        wr_1.append(ana_ans[0].real)
        wi_1.append(ana_ans[0].imag)
        wr_2.append(ana_ans[1].real)
        wi_2.append(ana_ans[1].imag)

    wr_2 = np.array(wr_2)

    plt.rcParams["font.size"] = 14

    # zfunc = np.loadtxt('../fig2.2/aaa.dat')

    fig = plt.figure(figsize=(7, 9), dpi=300)
    gs = fig.add_gridspec(2, 1, height_ratios=[1, 1], hspace=0)

    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(ans[:, 0]/kj, ans[:, 1]/wpe,
             c='g', marker='o', markersize=2, label='Numerical')
    ax1.plot(k_list[1:]/kj, wr_2/wpe, c='orange', label='Theoretical')
    # ax1.plot(zfunc[:, 0], zfunc[:, 1], c='c', label='zfunc.f90')
    ax1.set_ylabel(r'$\frac{\omega_r}{\omega_e}$',
                   labelpad=10, rotation=0, va='center', fontsize=16)
    ax1.set_ylim(0, 2.1)
    ax1.set_xlim(0, 1)
    ax1.tick_params(axis='x', which='both',
                    bottom=True, top=False, labelbottom=False, direction='in')
    ax1.legend()

    ax2 = fig.add_subplot(gs[1, 0], sharex=ax1)
    ax2.scatter(ans[:, 0]/kj, ans[:, 2]/wpe,
                c='g', marker='o', s=5, label='Numerical')
    # ax2.scatter(zfunc[:, 0], zfunc[:, 2], c='c', s=5, label='zfunc.f90')
    ax2.set_xlabel(r'$\frac{k}{k_e}$', fontsize=16)
    ax2.set_ylabel(r'$\frac{\gamma}{\omega_e}$',
                   labelpad=10, rotation=0, va='center', fontsize=16)
    ax2.set_ylim(-1, 0)
    ax2.set_xlim(0, 1)
    ax2.set_yticks([-1.0, -0.8, -0.6, -0.4, -0.2])
    ax2.set_yticklabels([-1.0, -0.8, -0.6, -0.4, -0.2])
    ax2.tick_params(bottom=True, top=True, labelbottom=True, direction='in')
    ax2.legend()

    plt.savefig('fig/langmuir_disp2.png', bbox_inches="tight")
    plt.close()
