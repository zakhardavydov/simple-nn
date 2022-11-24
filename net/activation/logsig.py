import numpy as np

from .activation import Activation


class LogSigActivation(Activation):

    def fun(self, input):
        return 1 / (1 + np.exp(-input))

    def derivative(self, input):
        logsig = self.fun(input)
        return logsig * (1 - logsig)
