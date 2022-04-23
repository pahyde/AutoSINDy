import torch
import torch.nn as nn
from sindy_libs import SindyLib
from sindy_algorithms import STLSQ


class AutoSINDy(nn.Module):
    def __init__(
        self,
        input_dim, 
        latent_dim,
        libs=None, 
        feature_names=None
    ):
        super(AutoSINDy, self).__init__()
        n, m = input_dim, latent_dim 
        
        self.feature_names = [f'z{i+1}' for i in range(m)]
        self.lib = SindyLib(libs, num_features=m, feature_names=feature_names) 

        self.encoder = nn.Sequential(
            nn.Linear(n,10),
            nn.ReLU,
            nn.Linear(10,4),
            nn.RelU,
            nn.Linear(4,latent_dim)
        )

        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 4),
            nn.ReLU(),
            nn.Linear(4,10),
            nn.ReLU(),
            nn.Linear(10,n)
        )

        self.SINDy = nn.Linear(len(self.lib.terms), m, bias=False)

    def translate_derivative(X, X_dot, network):
        parameters = []
        for layer in encoder:
            if isinstance(layer, nn.Linear):
                parameters.append((layer.weight, layer.bias))
        l, dl = X, X_dot
        relu = nn.ReLU()
        for i, (w,b) in enumerate(parameters):
            l = l @ w.T + b
            dl = dl @ w.T
            if i < len(parameters) - 1:
                dl = (l > 0).float() * dl
                l = relu(l)
        return dl

    def library(self):
        print('library candidate terms:')
        return self.lib.term_names
    
    def Xi_weights(self):
        params = list(self.parameters())[0]
        return params

    def show(self):
        rows = self.Xi_weights().tolist()
        equations = [[round(coeff, 3) for coeff in row] for row in rows]
        for i,eq in enumerate(equations):
            x   = self.feature_names[i]
            rhs = ' + '.join(f'{coeff} {name}' for coeff,name in zip(eq, self.lib.term_names))
            print(f'({x})\' = {rhs}')

    def forward(self, X, X_dot):
        Z = self.encoder(X)
        X_recon = self.decoder(Z)
        Z_dot = self.translate_derivative(X, X_dot, self.encoder)
        f_Z = self.SINDy(self.lib.theta(Z))
        f_X = self.translate_derivative(Z, f_Z, self.decoder)
        Xi = self.Xi_weights()
        return X_recon, Z_dot, f_Z, f_X, Xi   





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
    






        
        
