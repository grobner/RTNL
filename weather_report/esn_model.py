import numpy as np
import networkx as nx

class ESN:
    def __init__(self, reservoir_size, input_size, density, spectral_radius, seed):
        self.input_size = input_size
        self.reservoir_size = reservoir_size
        self.density = density
        self.spectral_radius = spectral_radius
        self.input_scale = 0.1

        self.win = (2*np.random.rand(reservoir_size, input_size)-1)
        self.W = self._initialize_reservoir(density, spectral_radius, seed)

        self.output_layer = np.random.normal(size=(reservoir_size, input_size))
        self.state = np.zeros(self.reservoir_size)

    def _initialize_reservoir(self, density, spectral_radius, seed):
        m = int(self.reservoir_size*(self.reservoir_size-1)*density/2)
        G = nx.gnm_random_graph(self.reservoir_size, m, seed=seed)
        connection = nx.to_numpy_array(G)
        W = np.array(connection)

        W *= (2*np.random.rand(self.reservoir_size, self.reservoir_size)-1)
        rho_W = np.max(np.abs(np.linalg.eigvals(W)))
        W *= spectral_radius / rho_W
        return W

    def fit(self, x, y):
        x_out = np.zeros((x.shape[0], self.reservoir_size))
        for i in range(x.shape[0]):
            x_out[i] = self._update_reservoir(x[i])

        H_pinv = np.linalg.pinv(x_out)
        self.output_layer = H_pinv @ y

        y_predict = x_out @ self.output_layer
        return y_predict

    def run(self, x):
        test_len = len(x)
        y_pred = np.zeros((x.shape[0], self.input_size))
        y = x[0]
        for n in range(test_len):
            y = self._update_reservoir(y) @ self.output_layer
            y_pred[n] = y

        return y_pred

    def predict(self, x):
        test_len = len(x)
        y_pred = np.zeros((x.shape[0], self.input_size))
        for n in range(test_len):
            y = self._update_reservoir(x[n]) @ self.output_layer
            y_pred[n] = y

        return y_pred

    def _update_reservoir(self, u):
        state = self._sigmoid(self.input_scale * self.win @ u + self.W @ self.state)
        self.state = state
        return state

    def _sigmoid(self, x):
        return np.exp(np.minimum(x, 0)) / (1 + np.exp(- np.abs(x)))
