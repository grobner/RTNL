import numpy as np

class LorenzSystem:
    def __init__(self, sigma, rho, beta):
        self.sigma = sigma
        self.rho = rho
        self.beta = beta

    def dx_dt(self, t, x, y, z):
        return -self.sigma*x + self.sigma*y

    def dy_dt(self, t, x, y, z):
        return -x*z + self.rho*x - y

    def dz_dt(self, t, x, y, z):
        return x*y - self.beta*z

    def compute_derivatives(self, t, state):
        '''
        Calculate the derivatives of the system at a given time and state.
        :param t: Time variable (not used directly in this system).
        :param state: A 3D vector [x, y, z].
        :return: Derivatives [dx/dt, dy/dt, dz/dt].
        '''
        x, y, z = state
        return np.array([
            self.dx_dt(t, x, y, z),
            self.dy_dt(t, x, y, z),
            self.dz_dt(t, x, y, z)
        ])

    def integrate(self, initial_state, total_time, step_size):
        '''
        Numerically integrates the system using the 4th-order Runge-Kutta method.
        :param initial_state: Initial state vector [x, y, z].
        :param total_time: Total integration time.
        :param step_size: Time step for integration.
        :return: Time series data of the system.
        '''
        state = initial_state
        current_time = 0
        trajectory = []

        while current_time < total_time:
            k1 = self.compute_derivatives(current_time, state)
            k2 = self.compute_derivatives(current_time + step_size / 2, state + step_size * k1 / 2)
            k3 = self.compute_derivatives(current_time + step_size / 2, state + step_size * k2 / 2)
            k4 = self.compute_derivatives(current_time + step_size, state + step_size * k3)
            state = state + step_size / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
            trajectory.append(state)
            current_time += step_size

        return np.array(trajectory)
