import numpy as np
# NARMA sequence generator
class NARMA:
    def __init__(self, order, coef1, coef2, coef3, coef4):
        """
        Initialize the NARMA sequence generator.
        :param order: Order of the NARMA model.
        :param coef1: Coefficient for the previous value term.
        :param coef2: Coefficient for the interaction term.
        :param coef3: Coefficient for the input interaction term.
        :param coef4: Bias term.
        """
        self.order = order
        self.coef1 = coef1
        self.coef2 = coef2
        self.coef3 = coef3
        self.coef4 = coef4

    def generate_data(self, length, initial_values, seed=0):
        """
        Generate a NARMA time series.
        :param length: Total number of time steps.
        :param initial_values: List of initial values for the sequence.
        :param seed: Seed for reproducibility of the random input.
        :return: Input sequence and output sequence.
        """
        y = list(initial_values)
        if len(y) < self.order:
            raise ValueError("Initial values must be at least of length equal to the model order.")

        np.random.seed(seed)
        u = np.random.uniform(0, 0.5, length)
        # Generate the sequence
        for n in range(self.order, length):
            y_n = (
                self.coef1 * y[n - 1] +
                self.coef2 * y[n - 1] * np.sum(y[n - self.order:n-1]) +
                self.coef3 * u[n - self.order] * u[n] +
                self.coef4
            )
            y.append(y_n)

        return u, np.array(y)
