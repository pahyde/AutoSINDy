import numpy as np
import matplotlib.pyplot as plt
import torch
from scipy.integrate import solve_ivp
from scipy import signal



class AutoSINDyDataset:
    def __init__(self, system, transform, use_jacobian=False):
        self.intrinsic = system.data()
        self.hidden    = self.intrinsic.transform(transform, use_jacobian)
        self.Z         = self.intrinsic.X
        self.Z_dot     = self.intrinsic.X_dot
        self.X         = self.hidden.X
        self.X_dot     = self.hidden.X_dot
        self.equations = system.equations

    def plotZ(self):
        _,m = self.intrinsic.trajectories[0].x.shape
        plt.figure(figsize=(8,6), dpi=100)
        if m == 3:
            ax = plt.axes(projection='3d')
        for z, z_dot, _ in self.intrinsic.trajectories:
            if m == 1:
                plt.plot(z)
            elif m == 2:
                plt.plot(z[:,0], z[:,1], linestyle='None', marker='.', markersize=1)
            elif m == 3:
                ax.plot3D(z[:,0], z[:,1], z[:,2], linestyle='None', marker='.', markersize=1);

    def plotX(self):
        _,m = self.hidden.trajectories[0].x.shape
        plt.figure(figsize=(8,6), dpi=100)
        if m == 3:
            ax = plt.axes(projection='3d')
        for x, x_dot, _ in self.hidden.trajectories:
            if m == 1:
                plt.plot(x)
            elif m == 2:
                plt.plot(x[:,0], x[:,1], linestyle='None', marker='.', markersize=1)
            elif m == 3:
                ax.plot3D(x[:,0], x[:,1], x[:,2], linestyle='None', marker='.', markersize=1);

    def __getitem__(self, index):
        return (self.X[index], self.X_dot[index])

    def __len__(self):
        return len(self.X)




def generate_system_from_dynamics(f, init_conditions, t0, tf, steps, equations=None):
    t = np.linspace(t0, tf, steps)
    system = solve_ivp(f, (t0, tf), init_conditions, dense_output=True)
    X = np.stack(system.sol(t), axis=-1)
    X_dot = np.apply_along_axis(lambda row: f(t=None, X=row), 1, X)
    return DynamicalSystem(
        torch.Tensor(X), 
        torch.Tensor(X_dot), 
        torch.Tensor(t),
        equations,
        init_conditions
    )

def generate_system_from_data(X, dt=0.01):
    n,m = X.shape
    X_dot_0   = (X[1] - X[0]) / dt
    X_dot_mid = (X[1:-1] - X[0:-2]) / (2*dt)
    X_dot_f   = (X[-1] - X[-2]) / dt
    X_dot = np.vstack((X_dot_0, X_dot_mid, X_dot_f))
    t = np.linspace(0, n*dt - dt, n)
    return DynamicalSystem(
        torch.Tensor(X),
        torch.Tensor(X_dot),
        torch.Tensor(t)
    )
'''
class DynamicalSystem:
    def __init__(self, X, X_dot, t, equations=None, init_conditions=None):
        self.X = X
        self.X_dot = X_dot
        self.t = t
        self.eq = equations
        self.ics = init_conditions

    def time_series(self):
        return self.X, self.X_dot, self.t

    def show(self):
        print(f'\ninitial conditions: {self.ics}\n')
        if self.eq is None:
            print('Nothing equation to show. Must initialize object with equations argument')
            return
        print('equations:')
        print(self.eq)
        print()

'''

if __name__ == "__main__":
    from dynamics import Exp2D, Lorenz
    
    exp2d = Exp2D()
    lorenz = Lorenz()

    system = generate_system_from_dynamics(
        lorenz.f,
        init_conditions = (0, 1, 1.05),
        t0=0,
        tf=10,
        steps=10000,
        equations=lorenz.equations
    )

    system.show()
    #col1 = signal.convolve(X_dot[:,0], signal.windows.hann(50), mode="same")
    #col2 = signal.convolve(X_dot[:,1], signal.windows.hann(50), mode="same")
    #filtered = np.stack((col1,col2), axis=1)
