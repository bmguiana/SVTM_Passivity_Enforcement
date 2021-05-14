# -*- coding: utf-8 -*-
"""
Created on Tue May 11 17:23:45 2021

@author: guia3994
"""

import numpy as np
import warnings
warnings.filterwarnings(action="error", category=np.ComplexWarning)

from VFdriver import VFdriver
# from RPdriver import RPdriver
from plots import plot_figure_11
from create_netlist import create_netlist_file
from passivity_driver import passive_driver
import skrf as rf


big_y_net = rf.Network('input_4_port_ads.s4p')
big_y_npy = big_y_net.y
s = big_y_net.f*2j*np.pi
sh = big_y_npy.shape
bigY = np.zeros([sh[-1], sh[-2], len(s)], dtype=np.complex128)
for n in range(len(s)):
    bigY[:, :, n] = big_y_npy[n, :, :]

# Pole-Residue Fitting
vf_driver = VFdriver(N=50,
                      poletype='linlogcmplx',
                      weightparam='common_sqrt',
                      Niter1=7,
                      Niter2=4,
                      asymp='D',
                      logx=False,
                      plot=False
                      )
poles=None
SER, rmserr, bigYfit = vf_driver.vfdriver(bigY, s, poles)

# Passivity Enforcement
SER, bigYfit_passive = passive_driver(SER, s, max_iter=10)

plot_figure_11(s, bigY, bigYfit_passive, SER)

poles = SER['poles']
residues = SER['C']
Ns = poles.shape[0]
Nc = int(residues.shape[1] / Ns)
poles = poles.reshape((1, -1))
residues = residues.reshape((Nc ** 2, Ns))
create_netlist_file(poles, residues)
