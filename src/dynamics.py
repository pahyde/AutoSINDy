import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.autograd.functional import jacobian
from scipy.integrate import solve_ivp


def stack(*args):
    return '\n'.join([*args])

class DynamicalSystem:
    def __init__(self, f, equations=None):
        self.f = f
        self.equations = equations
        self.init_data()

    def init_data(self):
        self.trajectories = []
        self.samples = []

    def add_trajectory(self, initial_conditions, t):
        t0, tf, steps = t
        time = np.linspace(t0, tf, steps)
        system = solve_ivp(
            self.f, 
            (t0, tf), 
            initial_conditions, 
            dense_output=True
        )
        X = np.stack(system.sol(time), axis=-1)
        self.trajectories.append(X)

    def add_trajectories(self, initial_conditions, t):
        for ics in initial_conditions:
            self.add_trajectory(ics, t)

    def add_random_sample(self):
        pass

    def snapshots(self):
        snapshots = self.trajectories + self.samples
        X = np.vstack(snapshots)
        X_dot = np.apply_along_axis(lambda row: self.f(None, row), 1, X)
        return DynamicalSystemData(X, X_dot)

    def plot(self):
        if len(self.trajectories) == 0:
            print('Nothing to plot. Add a trajectory')
            return
        _, m = self.trajectories[0].shape
        plt.figure(figsize=(8,6), dpi=100)
        if m == 3:
            ax = plt.axes(projection='3d')
        for traj in self.trajectories:
            if m == 1 or m > 3:
                plt.plot(traj);
            elif m == 2:
                plt.plot(traj[:,0], traj[:,1]);
            elif m == 3:
                ax.plot3D(traj[:,0], traj[:,1], traj[:,2]);


    def show(self):
        print(f'num trajectories: {len(self.trajectories)}')
        print(f'num samples:      {len(self.samples)}')
        print()
        if self.equations is None:
            print('Nothing equation to show. Must initialize object with equations argument')
            return
        print('equations:')
        print(self.equations)
        print()


class DynamicalSystemData:
    def __init__(self, X, X_dot):
        self.X = X
        self.X_dot = X_dot

    def transform(self, T):
        Z = np.apply_along_axis(T, 1, self.X)
        Z_dot = np.zeros(Z.shape)
        for i in range(Z_dot.shape[0]):
            Z_dot[i] = self.__translate_derivative(
                self.X[i], 
                self.X_dot[i],
                T
            )
        return DynamicalSystemData(Z, Z_dot)

    def __translate_derivative(self, x, x_dot, T):
        x = torch.Tensor(x)
        x_dot = torch.Tensor(x_dot)
        z_dot = jacobian(T,x) @ x_dot
        return z_dot.numpy()

    def __iter__(self):
        yield from [self.X, self.X_dot]


class Exp1D(DynamicalSystem):
    def __init__(self, k=1):
        self.k = k
        self.equations = f'(x)\' = {k}x'
        self.init_data()

    def f(self, t, x):
        return self.k * x


class Exp2D(DynamicalSystem):
    def __init__(self):
        x1 = '(x)\' = 2 * y'
        x2 = '(y)\' = -x'
        self.equations = stack(x1,x2)
        self.init_data()


    def f(self, t, X):
        x,y = X
        return np.array([
            2 * y,
            -x
        ])


class Lorenz(DynamicalSystem):
    def __init__(self, sigma=10, beta=2.667, rho=28):
        self.sigma = sigma
        self.beta = beta
        self.rho = rho
        x1 = f'(x)\' = {self.sigma}(y - x)'
        x2 = f'(y)\' = x({self.rho} - z) - y'
        x3 = f'(z)\' = xy - {self.beta}z'
        self.equations = stack(x1,x2,x3)
        self.init_data()

    def f(self, t, X):
        x,y,z = X
        return np.array([
            self.sigma * (y - x),
            x * (self.rho - z) - y,
            x * y - self.beta*z
        ]) 



class SimplePendulum(DynamicalSystem):
    def __init__(self, g=9.8, l=2):
        self.g = g
        self.l = l
        x1 = f'(θ)\'     = θ_dot'
        x2 = f'(θ_dot)\' = -{self.g}/{self.l} * sin(θ)'
        self.equations = stack(x1,x2)
        self.init_data()

    def f(self, t, X):
        theta, theta_dot = X
        g = self.g
        l = self.l
        mu = self.mu
        return np.array([
            theta_dot,
            -g / l * np.sin(theta)# - mu * theta_dot
        ])


class DampedPendulum(DynamicalSystem):
    def __init__(self, g=9.8, l=2, mu=0.5):
        self.g = g
        self.l = l
        self.mu = mu
        x1 = f'(θ)\'     = θ_dot'
        x2 = f'(θ_dot)\' = -{self.g}/{self.l} * sin(θ) - {self.mu} * θ_dot'
        self.equations = stack(x1,x2)
        self.init_data()

    def f(self, t, X):
        theta, theta_dot = X
        g = self.g
        l = self.l
        mu = self.mu
        return np.array([
            theta_dot,
            -g / l * np.sin(theta)# - mu * theta_dot
        ])




def simple_harmonic_osc_dynamics(t, X, k=0.2):
    x, x_dot = X
    return np.array([
        x_dot,
        -k * x
    ])

if __name__ == "__main__":
    import sindy_data

    exp = sindy_data.generate_system_from_dynamics(
        exp_2d_dynamics,
        init_conditions=(1,1),
        t0=0,
        tf=4,
        steps=10000
    )

    exp.show()
