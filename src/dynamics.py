import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.autograd.functional import jacobian
from scipy.integrate import solve_ivp

import sindy_utils

# trajectories
def stack(*args):
    return '\n'.join([*args])

class Trajectory:
    def __init__(self, x, x_dot, time):
        self.x = x
        self.x_dot = x_dot
        self.time = time

    def __iter__(self):
        yield from [
            self.x, 
            self.x_dot,
            self.time
        ]

class VectorField:
    def __init__(self, x, x_dot):
        self.x = x
        self.x_dot = x_dot

    def __iter__(self):
        yield from [self.x, self.x_dot]


class DynamicalSystem:
    def __init__(self, f, equations=None, order=1):
        self.f = f
        self.equations = equations
        self.init_data()

    def init_data(self):
        self.trajectories = []
        self.vector_field = None
        self.samples = []

    def add_trajectory(self, initial_conditions, t, method='RK45', numerical_ddt=False):
        t0, tf, steps = t
        time = np.linspace(t0, tf, steps)
        system = solve_ivp(
            self.f, 
            (t0, tf), 
            initial_conditions, 
            dense_output=True,
            method=method
        )
        X = np.stack(system.sol(time), axis=-1)
        self.trajectories.append(Trajectory(
            X,
            self.__dxdt(X,time) if numerical_ddt else self.__derivatives(X),
            time
        ))

    def add_trajectories(self, initial_conditions, t, method='RK45', numerical_ddt=False):
        for ics in initial_conditions:
            self.add_trajectory(ics, t, method, numerical_ddt)

    def add_vector_field(self, density=20):
        X = np.zeros((density*density,2))
        X_dot = np.zeros((density*density,2))
        x0 = min(traj.x[:,0].min() for traj in self.trajectories)
        xf = max(traj.x[:,0].max() for traj in self.trajectories)
        y0 = min(traj.x[:,1].min() for traj in self.trajectories)
        yf = max(traj.x[:,1].max() for traj in self.trajectories)
        for i,y in enumerate(np.linspace(y0,yf,density)):
            for j,x in enumerate(np.linspace(x0,xf,density)):
                loc = np.array([x,y])
                direction = self.f(None,loc)
                X[i*density + j] = loc
                X_dot[i*density + j] = direction
        self.vector_field = VectorField(X, X_dot)


    def add_random_sample(self):
        pass

    def data(self):
        return DynamicalSystemData(
            self.trajectories, 
            self.vector_field,
            self.equations
        )

    def __derivatives(self, X):
        f_x = lambda x: self.f(None, x)
        return np.apply_along_axis(f_x, 1, X)

    def __dxdt(self, X, t):
        n,m = X.shape
        return np.gradient(X,t,axis=0)

    def plot(
        self, 
        figsize=(8,8),
        dpi=100,
        scatter=False,
        marker='.',
        markersize=2,
        time_domain=False,
        vector_field=True
    ):
        if self.vector_field is None:
            vector_field = False

        _, m = self.trajectories[0].x.shape
        args = (figsize, dpi, scatter, marker, markersize)
        if time_domain or m == 1:
            return sindy_utils.plot_time(
                *args,
                dim=m,
                trajectories=self.trajectories
            )
        else:
            print(self.vector_field)
            return sindy_utils.plot_phase(
                *args,
                dim=m,
                trajectories=self.trajectories,
                vector_field=self.vector_field if vector_field else None
            )

    def plot1(self):
        if len(self.trajectories) == 0:
            print('Nothing to plot. Add a trajectory')
            return
        _, m = self.trajectories[0].x.shape
        plt.figure(figsize=(8,6), dpi=100)
        if m == 3:
            ax = plt.axes(projection='3d')
        for traj in self.trajectories:
            snapshots = traj.x
            if m == 1 or m > 3:
                plt.plot(snapshots);
            elif m == 2:
                plt.plot(snapshots[:,0], snapshots[:,1]);
            elif m == 3:
                ax.plot3D(snapshots[:,0], snapshots[:,1], snapshots[:,2]);


    def show(self):
        print(f'num trajectories: {len(self.trajectories)}')
        print(f'num samples:      {len(self.samples)}')
        print()
        if self.equations is None:
            print('No equation to show. Initialize object with equations argument')
            return
        print('equations:')
        print(self.equations)
        print()


class DynamicalSystemData:
    def __init__(self, trajectories, vector_field, equations=None):
        self.trajectories = trajectories
        self.vector_field = vector_field
        self.equations = equations
        self.X = torch.Tensor(
            np.vstack([traj.x for traj in trajectories])
        )
        self.X_dot = torch.Tensor(
            np.vstack([traj.x_dot for traj in trajectories])
        )

    def transform(self, T, use_jacobian=False):
        transformed = []
        for x, x_dot, time in self.trajectories:
            z = self.__transform(x,T)
            z_dot = self.__f_Z(x,x_dot,T) if use_jacobian else self.__dzdt(z,time)
            transformed.append(Trajectory(z, z_dot, time))
        return DynamicalSystemData(transformed)

    def __f_Z(self, X, X_dot, T):
        DZ = np.apply_along_axis(
            lambda x: jacobian(T,torch.Tensor(x)).numpy(),
            1, X
        )
        Z_dot = self.__mmult_components(DZ, X_dot)
        return Z_dot

    def __dzdt(self, X, time):
        n,m = X.shape
        return np.gradient(X,time,axis=0)

    def __transform(self, X, T):
        return np.apply_along_axis(T, 1, X)

    def __mmult_components(self, A, x):
        _,N,M = A.shape
        n,m   = x.shape
        return (A @ x.reshape(n,m,1)).reshape(n,N)

    def plot(
        self, 
        figsize=(8,8),
        dpi=100,
        scatter=False,
        marker='.',
        markersize=2,
        time_domain=False,
        vector_field=True
    ):
        if self.vector_field is None:
            vector_field = False

        _, m = self.trajectories[0].x.shape
        args = (figsize, dpi, scatter, marker, markersize)
        if time_domain or m == 1:
            return sindy_utils.plot_time(
                *args,
                dim=m,
                trajectories=self.trajectories
            )
        else:
            print(self.vector_field)
            return sindy_utils.plot_phase(
                *args,
                dim=m,
                trajectories=self.trajectories,
                vector_field=self.vector_field if vector_field else None
            )

        

    
    def __iter__(self):
        yield from [self.X, self.X_dot]

    #def transform(self, T):
    #    Z = np.apply_along_axis(T, 1, self.X)
    #    Z_dot = np.zeros(Z.shape)
    #    for i in range(Z_dot.shape[0]):
    #        Z_dot[i] = self.__translate_derivative(
    #            self.X[i], 
    #            self.X_dot[i],
    #            T
    #        )
    #    return DynamicalSystemData(Z, Z_dot)

    #def __translate_derivative(self, x, x_dot, T):
    #    x = torch.Tensor(x)
    #    x_dot = torch.Tensor(x_dot)
    #    z_dot = jacobian(T,x) @ x_dot
    #    return z_dot.numpy()



class Linear1D(DynamicalSystem):
    def __init__(self, a=1):
        self.a = a
        self.equations = f"""
        (x)' = {a}x
        """ 
        self.init_data()

    def f(self, t, x):
        return self.a * x


class Affine1D(DynamicalSystem):
    def __init__(self, a=1, b=1):
        self.a = a
        self.b = b
        self.equations = f"""
        (x)' = {a}x + {b}
        """ 
        self.init_data()

    def f(self, t, x):
        return self.a * x + self.b


class Quadratic1D(DynamicalSystem):
    def __init__(self, a=1, b=0, c=-4):
        self.a = a
        self.b = b
        self.c = c
        self.equations = f"""
        (x)' = {a}x^2 + {b}x + {c}
        """ 
        self.init_data()

    def f(self, t, x):
        return self.a * x**2 + self.b * x + self.c

class Sin1D(DynamicalSystem):
    def __init__(self, a=1):
        self.a = a
        self.equations = f"""
        (x)' = {a} * sin(x)
        """ 
        self.init_data()

    def f(self, t, x):
        return self.a * np.sin(x)


A2 = np.array([[0, 2],
               [-1,0]])

class Linear2D(DynamicalSystem):
    def __init__(self, A = A2):
        self.A = A
        a,b = A[0]
        c,d = A[1]
        self.equations = f"""
        (x)' = {a}x + {b}y
        (y)' = {c}x + {d}y 
        """ 
        self.init_data()

    def f(self, t, X):
        return self.A @ X


B2 = np.array([1,1])
class Affine2D(DynamicalSystem):
    def __init__(self, A = A2, B=B2):
        self.A = A
        a,b = A[0]
        c,d = A[1]
        e,f = B
        self.equations = f"""
        (x)' = {a}x + {b}y + {e}
        (y)' = {c}x + {d}y + {f} 
        """ 
        self.init_data()

    def f(self, t, X):
        return self.A @ X + self.B


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
            -g / l * np.sin(theta) - mu * theta_dot
        ])

class DampedOsc(DynamicalSystem):
    def __init__(self, b=0.5, c=1):
        self.b = b
        self.c = c
        x1 = f'(x)\'     = x_dot'
        x2 = f'(x_dot)\' = -{self.c} * x - {self.b} * x_dot'
        self.equations = stack(x1,x2)
        self.init_data()

    def f(self, t, X):
        x, x_dot = X
        b = self.b
        c = self.c
        return np.array([
            x_dot,
            -c * x - b * x_dot
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
