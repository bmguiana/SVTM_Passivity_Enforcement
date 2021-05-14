# -*- coding: utf-8 -*-
"""
Created on Mon Apr 12 17:24:05 2021

@author: guia3994
"""

import numpy as np
import numpy.linalg as la
import matplotlib.pyplot as plt

def get_single_Y(gsy_A, gsy_B, gsy_C, gsy_D, gsy_w):
    gsy_s = 1j*gsy_w
    inner = la.inv(gsy_s * np.eye(gsy_A.shape[0]) - gsy_A)
    gsy_Y = gsy_C.dot(inner).dot(gsy_B) + gsy_D
    gsy_G = la.eigvals(gsy_Y).real
    return gsy_Y, gsy_G


def scrub(noisy_A, tol=10):
    lim = 10**-tol
    norm = noisy_A/noisy_A.real
    clean_A = noisy_A.copy()
    for n in np.arange(noisy_A.shape[0]):
        if np.abs(norm[n].imag) < lim:
            clean_A[n] = noisy_A[n].real
    return clean_A


def get_violation_list(viol_efs, viol_A, viol_B, viol_C, viol_D, viol_f):
    num_efs = viol_efs.shape[0]
    viol_list = np.array(())                   # Initialize list of violation eigenfrequencies

    for n in np.arange(num_efs):
        if viol_efs[n].real > 0 and viol_efs[n].imag == 0 and viol_efs[n] < np.max(2*np.pi*viol_f):
            viol_list = np.append(viol_list, viol_efs[n])

    num_viols = viol_list.shape[0]             # Number of crossover frequencies
    viol_list = np.sort(viol_list)             # Sort eigenfrequency list ascending
    viol_type = np.zeros(num_viols)            # Initialize list of eigenfrequency slope signs

    reg_chk = np.append(0, viol_list)
    reg_chk = np.append(reg_chk, np.max(2*np.pi*viol_f))
    mid_type = np.zeros(len(reg_chk)-1)
    for n in range(len(reg_chk)-1):
        mid = (reg_chk[n] + reg_chk[n+1])/2
        Ymid, Gmid = get_single_Y(viol_A, viol_B, viol_C, viol_D, mid)
        if any(Gmid < 0):
            mid_type[n] = 1
        elif all(Gmid > 0):
            mid_type[n] = -1

    viol_type = mid_type[:-1]
    num_viols = len(viol_type)
    # print('X-over freq. (rad/s): ', viol_list.real)
    # print('X-over slope: ', viol_type)

    # for n in np.arange(num_viols):
    #     loYs, loGs = get_single_Y(viol_A, viol_B, viol_C, viol_D, viol_list[n]-1)
    #     hiYs, hiGs = get_single_Y(viol_A, viol_B, viol_C, viol_D, viol_list[n]+1)
    #     if (loGs<0).any() and (hiGs>0).any():
    #         viol_type[n] = 1
    #     if (loGs>0).any() and (hiGs<0).any():
    #         viol_type[n] = -1
    return viol_list, viol_type, num_viols


def update_residue(ur_ef_type, ur_ef_list, ur_eigenfreqs, ur_w, ur_P, ur_QT, ur_A, ur_B, ur_Di):
    ur_Nc = ur_B.shape[1]
    ur_N = ur_B.shape[0]//ur_B.shape[1]

    if ur_ef_type[ur_w] == 1 and not ur_w:
        dwj = ur_ef_list[ur_w]
    elif ur_ef_type[ur_w] == 1 and ur_ef_type[ur_w-1] == -1:
        avg = ur_ef_list[ur_w]/2 + ur_ef_list[ur_w-1]/2
        dwj = ur_ef_list[ur_w] - avg
    elif ur_ef_type[ur_w] == -1:
        avg = ur_ef_list[ur_w]/2 + ur_ef_list[ur_w+1]/2
        dwj = ur_ef_list[ur_w+1] - avg
    elif ur_ef_type[ur_w] == 0:
        return 0

    for n in np.arange(ur_eigenfreqs.shape[0]):
        if ur_ef_list[ur_w] == ur_eigenfreqs[n]:
            ind = n
            break

    dwj *= -ur_ef_type[ur_w]
    dlj = 2*ur_ef_list[ur_w]*dwj + dwj**2
    pj = ur_P[:, ind].reshape(-1, 1)
    qjT = ur_QT[ind, :].reshape(1, -1)
    dS = dlj * (pj.dot(qjT))
    dCp = la.lstsq(ur_A.dot(ur_B).dot(ur_Di), dS, rcond=None)[0]
    dCs = np.zeros_like(dCp)
    for i in range(ur_Nc):                             # Enforce residue matrix delta symmetry
        for j in range(ur_Nc):
            for p in range(ur_N):
                avg = (dCp[i, j*ur_N+p] + dCp[j, i*ur_N+p])/2
                dCs[i, j*ur_N+p] = avg
                dCs[j, i*ur_N+p] = avg

    dSp = ur_A.dot(ur_B).dot(ur_Di).dot(dCs)
    epsilon = dlj/(qjT.dot(dSp).dot(pj))
    dC = epsilon[0, 0] * dCs
    return dC


def enforce_passivity(ep_A, ep_B, ep_C, ep_D, ep_f):
    passive_flag = False
    ep_Nc = ep_B.shape[1]
    ep_N = ep_B.shape[0]//ep_Nc

    ep_Di = la.inv(ep_D)                                 # Matrix inverse of D
    S = ep_A.dot(ep_B.dot(ep_Di).dot(ep_C) - ep_A)                # Singular value test matrix (SVTM)

    eigenvalues, eigenvectors = la.eig(S)          # Get eigendecomposition of SVTM (S @ V = /\ @ V)
    evs_clean = scrub(eigenvalues)                 # Remove noisy imaginary values from SVTM eigenvalues
    P = eigenvectors.copy()                        # Assign SVTM left eigenvector matrix (P = V)
    QT = la.inv(eigenvectors)                      # Assign SVTM right eigenvector matrix (Q.T = V**-1)

    eigenfreqs = np.sqrt(evs_clean)                # Get corresponding eigenfrequency values

                                                    # ef_list: only crossover frequencies
                                                    # ef_type: positive/negative 1 depending on slope
                                                    # viols: number of crossover frequencies
    ef_list, ef_type, viols = get_violation_list(eigenfreqs, ep_A, ep_B, ep_C, ep_D, ep_f)
    if type(ef_list) != np.ndarray or len(ef_list) == 0:
        # print('Passivity Enforced!')
        passive_flag = True
        return ep_C, passive_flag
    ep_Chat = ep_C.copy()
    for w in np.arange(viols):
        delta_C = update_residue(ef_type, ef_list, eigenfreqs, w, P, QT, ep_A, ep_B, ep_Di)
        ep_Chat += delta_C
    return ep_Chat, passive_flag

def passive_driver(SER, s, max_iter=1):
    Nc = SER['B'].shape[1]
    SERhat = SER.copy()
    f = (s/2j/np.pi).real
    SERhat['C'], pf = enforce_passivity(SER['A'], SER['B'], SER['C'], SER['D'], f)

    if max_iter > 1:
        if not pf:
            for it in range(max_iter):
                SERhat['C'], pf = enforce_passivity(SER['A'], SER['B'], SERhat['C'], SER['D'], f)
                if pf:
                    break
            print('Passivity enforced with: ', it+1, ' iteration(s)')

    Yhat = np.zeros( (Nc, Nc, len(f)), dtype=complex )
    Ghat = np.zeros( (Nc, len(f)))
    Y = np.zeros( (Nc, Nc, len(f)), dtype=complex )
    G = np.zeros( (Nc, len(f)))

    for n in range(len(f)):
        Yhat[:, :, n], Ghat[:, n] = get_single_Y(SER['A'], SER['B'], SERhat['C'], SER['D'], 2*np.pi*f[n])
        Y[:, :, n], G[:, n] = get_single_Y(SER['A'], SER['B'], SER['C'], SER['D'], 2*np.pi*f[n])

    # fig, ax = plt.subplots()
    # ax.plot(f, G[0, :], label='P1 eig', c='r', ls='-')
    # ax.plot(f, G[1, :], label='P2 eig', c='r', ls='--')
    # ax.plot(f, G[3, :], label='P4 eig', c='r', ls='-.')
    # ax.plot(f, Ghat[0, :], label='P1 eig', c='b', ls='-')
    # ax.plot(f, Ghat[1, :], label='P2 eig', c='b', ls='--')
    # ax.plot(f, Ghat[3, :], label='P4 eig', c='b', ls='-.')

    return SERhat, Yhat

if __name__ == '__main__':
    pass
    A = np.load('SERA.npy')                        # Load A matrix test file (poles)
    B = np.load('SERB.npy')                        # Load B matrix test file
    C = np.load('SERC.npy')                        # Load C matrix test file (residues)
    D = np.load('SERD.npy')                        # Load D matrix test file (constant)
    # f = np.linspace(1e1, 1e5, num=1001)            # Test FDNE frequency range (Hz)
    f = np.load('SERf.npy')

    Nc = B.shape[1]
    # SERhat = SER.copy()
    # f = (s/2j/np.pi).real
    # Chat, pf = enforce_passivity(A, B, C, D, f)


    passive_flag = False
    # ep_Nc = ep_B.shape[1]
    N = B.shape[0]//Nc

    Di = la.inv(D)                                 # Matrix inverse of D
    S = A.dot(B.dot(Di).dot(C) - A)                # Singular value test matrix (SVTM)

    eigenvalues, eigenvectors = la.eig(S)          # Get eigendecomposition of SVTM (S @ V = /\ @ V)
    evs_clean = scrub(eigenvalues)                 # Remove noisy imaginary values from SVTM eigenvalues
    P = eigenvectors.copy()                        # Assign SVTM left eigenvector matrix (P = V)
    QT = la.inv(eigenvectors)                      # Assign SVTM right eigenvector matrix (Q.T = V**-1)

    eigenfreqs = np.sqrt(evs_clean)                # Get corresponding eigenfrequency values

                                                    # ef_list: only crossover frequencies
                                                    # ef_type: positive/negative 1 depending on slope
                                                    # viols: number of crossover frequencies
    # ef_list, ef_type, viols = get_violation_list(eigenfreqs, A, B, C, D, f)
    # def get_violation_list(viol_efs, viol_A, viol_B, viol_C, viol_D, viol_f):

    num_efs = eigenfreqs.shape[0]
    viol_list = np.array(())                   # Initialize list of violation eigenfrequencies

    for n in np.arange(num_efs):
        if eigenfreqs[n].real > 0 and eigenfreqs[n].imag == 0 and eigenfreqs[n] < np.max(2*np.pi*f):
            viol_list = np.append(viol_list, eigenfreqs[n])

    # num_viols = viol_list.shape[0]             # Number of crossover frequencies
    viol_list = np.sort(viol_list)             # Sort eigenfrequency list ascending
    # viol_type = np.zeros(num_viols)            # Initialize list of eigenfrequency slope signs

    reg_chk = np.append(0, viol_list)
    reg_chk = np.append(reg_chk, np.max(2*np.pi*f))
    mid_type = np.zeros(len(reg_chk)-1)
    for n in range(len(reg_chk)-1):
        mid = (reg_chk[n] + reg_chk[n+1])/2
        Ymid, Gmid = get_single_Y(A, B, C, D, mid)
        if any(Gmid < 0):
            mid_type[n] = 1
        elif all(Gmid > 0):
            mid_type[n] = -1

    viol_type = mid_type[:-1]
    num_viols = len(viol_type)
    # print(viol_list, viol_type, num_viols)

    # for n in np.arange(num_viols):
    #     loYs, loGs = get_single_Y(viol_A, viol_B, viol_C, viol_D, viol_list[n]-1)
    #     hiYs, hiGs = get_single_Y(viol_A, viol_B, viol_C, viol_D, viol_list[n]+1)
    #     if (loGs<0).any() and (hiGs>0).any():
    #         viol_type[n] = 1
    #     if (loGs>0).any() and (hiGs<0).any():
    #         viol_type[n] = -1
    # return viol_list, viol_type, num_viols




    # if np.min(ef_list) > np.max(2*np.pi*ep_f) or type(ef_list) != np.ndarray or len(ef_list) == 0:
    #     print('Passivity Enforced!')
    #     passive_flag = True
    #     return ep_C, passive_flag
    # ep_Chat = ep_C.copy()
    # for w in np.arange(viols):
    #     delta_C = update_residue(ef_type, ef_list, eigenfreqs, w, P, QT, ep_A, ep_B, ep_Di)
    #     ep_Chat += delta_C
    # return ep_Chat, passive_flag


# =============================================================================
# =============================================================================
# =============================================================================
# # #
# =============================================================================
# =============================================================================
# =============================================================================

    # Chat, pass_flag = enforce_passivity(A, B, C, D, f)
    # Chat, pass_flag = enforce_passivity(A, B, Chat, D, f)

    # Y = np.zeros( (Nc, Nc, len(f)), dtype=complex )
    # G = np.zeros( (Nc, len(f)))
    # Yhat = np.zeros_like(Y)
    # Ghat = np.zeros_like(G)
    # for n in range(len(f)):
    #     Y[:, :, n], G[:, n] = get_single_Y(A, B, C, D, 2*np.pi*f[n])
    #     Yhat[:, :, n], Ghat[:, n] = get_single_Y(A, B, Chat, D, 2*np.pi*f[n])

    # fig, ax = plt.subplots()
    # ax.plot(f, G[0, :], label='P1 eig', c='r', ls='-')
    # ax.plot(f, G[1, :], label='P2 eig', c='r', ls='--')
    # ax.plot(f, G[2, :], label='P3 eig', c='r', ls='-.')
    # ax.plot(f, G[3, :], label='P4 eig', c='r', ls=':')
    # ax.plot(f, Ghat[0, :], label='P1 eig', c='b', ls='-')
    # ax.plot(f, Ghat[1, :], label='P2 eig', c='b', ls='--')
    # ax.plot(f, Ghat[2, :], label='P3 eig', c='b', ls='-.')
    # ax.plot(f, Ghat[3, :], label='P4 eig', c='b', ls=':')

    # # ax.set_xlim(())
    # # ax.set_ylim((-3e-5, 4e-5))
    # # ax.set_xlim((0, 100000))
    # # ax.set_ylim((-2e-3, 2e-3))
    # ax.legend()
    # ax.grid(True)
