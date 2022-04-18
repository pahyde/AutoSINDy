import numpy as np


def stack(*args):
    return '\n'.join([*args])


class Exp2D:
    def __init__(self):
        x1 = '(x)\' = 2 * x'
        x2 = '(y)\' = -y'
        self.equations = stack(x1,x2)

    def f(self, t, X):
        x,y = X
        return np.array([
            2 * x,
            -y
        ])


class Lorenz:
    def __init__(self, sigma=10, beta=2.667, rho=28):
        self.sigma = sigma
        self.beta = beta
        self.rho = rho
        x1 = f'(x)\' = {self.sigma}(y - x)'
        x2 = f'(y)\' = x({self.rho} - z) - y'
        x3 = f'(z)\' = xy - {self.beta}z'
        self.equations = stack(x1,x2,x3)

    def f(self, t, X):
        x,y,z = X
        return np.array([
            self.sigma * (y - x),
            x * (self.rho - z) - y,
            x * y - self.beta*z
        ]) 



class SimplePendulum:
    def __init__(self, g=9.8, l=2, mu=0.5):
        self.g = g
        self.l = l
        self.mu = mu
        x1 = f'(θ)\'     = θ_dot'
        x2 = f'(θ_dot)\' = -{self.g}/{self.l} * sin(θ) - {self.mu} * θ_dot'
        self.equations = stack(x1,x2)

    def f(self, t, X):
        theta, theta_dot = X
        g = self.g
        l = self.l
        mu = self.mu
        return np.array([
            theta_dot,
            -g / l * np.sin(theta) - mu * theta_dot
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
