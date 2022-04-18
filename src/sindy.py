import torch
import torch.nn as nn
from sindy_libs import SindyLib
from sindy_algorithms import STLSQ


class SINDy(nn.Module):
    def __init__(
        self,
        input_dim, 
        libs=None, 
        feature_names=None
    ):
        super(SINDy, self).__init__()
        n,m = input_dim

        if feature_names == None:
            feature_names = [f'x{i+1}' for i in range(m)]
        self.feature_names = feature_names

        self.lib = SindyLib(libs, num_features=m, feature_names=feature_names) 
        self.SINDy_layer = nn.Linear(len(self.lib.terms), m, bias=False)

        
    def library(self):
        print('library candidate terms:')
        return self.lib.term_names

    
    def Xi_weights(self):
        params = list(self.parameters())[0]
        return params

    def show(self):
        rows = torch.Tensor(self.Xi_weights()).tolist()
        equations = [[round(coeff, 3) for coeff in row] for row in rows]
        for i,eq in enumerate(equations):
            x   = self.feature_names[i]
            rhs = ' + '.join(f'{coeff} {name}' for coeff,name in zip(eq, self.lib.term_names))
            print(f'({x})\' = {rhs}')


    def forward(self, X):
        theta_X = self.lib.theta(X)
        f_X = self.SINDy_layer(theta_X)
        return f_X


class SINDy_LSQ:
    def __init__(self, libs, feature_names, algo):
        self.libs = libs
        self.feature_names = feature_names
        self.algo = algo
        self.Xi = None

    def fit(self, X, X_dot):
        n,m = X.size()

        if self.feature_names == None:
            self.feature_names = [f'x{i+1}' for i in range(m)]

        self.lib = SindyLib(self.libs, num_features=m, feature_names=self.feature_names)
        self.Xi = self.algo.fit(X, X_dot, self.lib)
        return self.Xi


    def show(self):
        if self.Xi is None:
            print('No model to display. Run model.fit(x,x_dot)')
            return
        rows = self.Xi.T.tolist()
        equations = [[round(coeff, 3) for coeff in row] for row in rows]
        for i,eq in enumerate(equations):
            x   = self.feature_names[i]
            terms = self.lib.term_names
            rhs = ' + '.join(f'{coeff} {name}' for coeff,name in zip(eq, terms) if coeff)
            print(f'({x})\' = {rhs}')



if __name__ == "__main__":
    from dynamics import exp_2d_dynamics 
    from sindy_libs import PolynomialLibrary, TrigLibrary
    import sindy_data

    exp_2d = sindy_data.generate_system_from_dynamics(
        exp_2d_dynamics,
        (1,1),
        t0=0,
        tf=2,
        steps=10000
    )

    X, X_dot = exp_2d.training_data()

    model = SINDy_LSQ(
        libs=[
            PolynomialLibrary(max_degree=3),
            TrigLibrary()
        ],
        feature_names=['x','y'],
        algo=STLSQ(threshold=0.2, reg=1)
    )
    Xi = model.fit(X, X_dot)
    print(Xi.shape)
    loss = torch.linalg.norm((model.lib.theta(X) @ Xi) - X_dot)
    print(loss)
    model.show()
    






        
        
