import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import pprint as pp
from findiff import FinDiff

import sys
sys.path.append('../src')

import sindy_utils
import sindy_utils
from sindy_data import AutoSINDyDataset
from dynamics import Linear1D, Linear2D, Quadratic1D
from dynamics import Lorenz
from dynamics import SimplePendulum, DampedPendulum, DampedOsc
from dynamics import DynamicalSystem

from sindy import SINDy, SINDy_LSQ
from autosindy import AutoSINDy
from sindy_algorithms import STLSQ
from sindy_libs import PolynomialLibrary, TrigLibrary, SindyLib

'''
By convention we denote intrinsic coordinates by Z and hidden coordinate T(Z) = X


What are the goals?

First: For the relevant functions we need d/dt(Z) ~= f(Z) with very close precision

where d/dt is numerically computed

This ensures the quality of numical integration methods and further that d/dt(X) ~= DT f(Z)
in other words numerically computed Z actually has derivatives corresponding to f

At length we could break this into two requirements

1) d/dt(Z) ~= f(Z)
2) d/dt(X) ~= DT f(Z)
'''

def test1():
    print('Testing d/dt(Z) = f(Z)')
    print('Linear 2D expanding system f(X) = Ax')
    A = np.array([[1,-3],
                  [2,.5]])
    
    mx = 0
    num = None
    mn = 500
    num_mn = None
    losses = []
    for steps in range(8600,8601):
        linear1 = Linear2D(A)
        linear2 = Linear2D(A)
        #linear1.show()

        ics = []
        for i in np.linspace(-4,4,3):
            for j in np.linspace(-5,6,3):
                ics.append((i,j))

        linear1.add_trajectories(
            initial_conditions=[(-4,3)],
            t=(0,3,steps)
        )

        linear2.add_trajectories(
            initial_conditions=[(-4,3)],
            t=(0,3,steps),
            numerical_ddt=True
        )
    
        Z, f_Z  = linear1.data()
        _, dZdt = linear2.data()

        div_diff = np.abs((dZdt - f_Z) / f_Z)
        #print('abs divided difference')
        #print(div_diff)
        mse = nn.MSELoss()(dZdt, f_Z)
        #print('mean square error loss')
        #print(mse.item())
        #print('max abs div diff for any value')
        curr = torch.argmax(div_diff[:,1]).item()
        print(torch.max(div_diff[:,1]))
        losses.append(curr)
        continue
        if curr >= mx:
            mx = curr
            num = steps
        if curr <= mn:
            mn = curr
            num_mn = steps
    print(mx, num)
    print(mn, num_mn)
    print(losses[0])
    print(Z[6713])
    print(Z[6714])
    print(Z[6715])
    print(f_Z[6714])
    print(dZdt[6714])






if __name__ == '__main__':
    A = np.array([[1,-3],
                  [2,.5]])
    linear1 = Linear2D(A)
    linear2 = Linear2D(A)
    #linear1.show()

    ics = []
    for i in np.linspace(-4,4,3):
        for j in np.linspace(-5,6,3):
            ics.append((i,j))

    steps=100000
    tf = 2
    linear1.add_trajectories(
        initial_conditions=[(-4,3)],
        t=(0,tf,steps)
    )

    linear2.add_trajectories(
        initial_conditions=[(-4,3)],
        t=(0,tf,steps),
        numerical_ddt=True
    )
 
    Z, f_Z  = linear1.data()
    _, dZdt = linear2.data()

    dt = 3 / 1000
    d_dt = FinDiff(0,dt, acc=6)

    Z_dot = torch.Tensor(d_dt(np.array(Z)))

    new = Z_dot / f_Z
    old = dZdt / f_Z
    old_diff = np.abs((dZdt - f_Z) / f_Z).max()
    new_diff = np.abs((Z_dot - f_Z) / f_Z).max()
    new = nn.MSELoss()(Z_dot, f_Z)
    old = nn.MSELoss()(dZdt, f_Z)
    print('old')
    print(old_diff)
    print(old)
    print('new')
    print(new_diff)
    print(new)
    

    









