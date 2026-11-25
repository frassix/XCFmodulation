"""Fitting helpers: functions written as strings (ROOT style, e.g. "[0] + [1]*x") and an iterative chi2 fit."""
import re

import numpy as np
from scipy.optimize import curve_fit

# Placeholders "[0]", "[1]", ... inside an expression
_PARAMETER = re.compile(r'\[(\d+)\]')

# Names that can be used inside an expression
_MATH = {
    'exp':    np.exp,
    'log':    np.log,
    'log10':  np.log10,
    'sin':    np.sin,
    'cos':    np.cos,
    'tan':    np.tan,
    'arcsin': np.arcsin,
    'arccos': np.arccos,
    'arctan': np.arctan,
    'sinh':   np.sinh,
    'cosh':   np.cosh,
    'tanh':   np.tanh,
    'abs':    np.abs,
    'sqrt':   np.sqrt,
    'pi':     np.pi,
}


class GF1():
    """Function of x defined by a string like "[0] + [1]*x", callable as f(x, p0, p1, ...)."""

    def __init__(self, expression):
        """Parses the expression once and compiles it."""
        self.function = expression
        indexes = [int(i) for i in _PARAMETER.findall(expression)]
        self.nPar = max(indexes) + 1 if indexes else 0
        self._code = compile(_PARAMETER.sub(r'_p\1', expression), '<GF1>', 'eval')

    def __call__(self, x, *parameters):
        """Evaluates the function at x with the given parameters."""
        namespace = dict(_MATH, x=x)
        for i, value in enumerate(parameters):
            namespace[f'_p{i}'] = value.item() if hasattr(value, 'item') else value
        return self.Eval(namespace)

    def Eval(self, namespace):
        """Evaluates the compiled expression in the given namespace."""
        try:
            return eval(self._code, {"__builtins__": None}, namespace)
        except Exception as e:
            raise ValueError(f"Evaluation error in '{self.function}': {e}")

    def GetNumberOfParameters(self):
        """Returns how many parameters ([0], [1], ...) the expression uses."""
        return self.nPar


class GFit():
    """Repeats a curve_fit until the chi2 changes less than delta (or MaxCycle is reached)."""

    def __init__(self, expression, delta=0.001, MaxCycle=5):
        """Builds the fit function from a string expression."""
        print(f'New Function added: {expression}')
        self.GF = GF1(expression)
        self.delta = delta
        self.nMax = MaxCycle
        self.X2 = None

    def __call__(self, x, y, errY=None, params=None, bounds=None):
        """Fits y(x) and returns (parameters, covariance) of the last iteration."""
        print('-'*100 + f'\nRepeated Fit enabled, maximum number of cycles set to {self.nMax}')
        if params is None:
            params = np.ones(self.GF.GetNumberOfParameters())
        if bounds is None:
            bounds = self.InfBounds()
        epsilon = 0.
        n = 0
        while True:
            params, covariance = curve_fit(self.GF, x, y, sigma=errY, p0=params, bounds=bounds)
            print(f'Cycle: {n}, Parameters: {params}')
            Chi2 = self.ChiSquare(x, y, params)
            dof = len(x) - len(params)
            print(f'Chi2: {Chi2} with {dof} dof.\n Reduced Chi2: {Chi2/dof}')
            if np.abs(epsilon - Chi2) < self.delta or n > self.nMax:
                print('-'*100)
                return params, covariance
            epsilon = Chi2
            n += 1

    def ChiSquare(self, x, O_i, params):
        """Returns (and stores) the Pearson chi2: sum((O-E)^2/E)."""
        E_i = np.array(self.GF(x, *params))
        self.X2 = (((O_i - E_i)**2) / E_i).sum()
        return self.X2

    def GetChi2(self):
        """Returns the chi2 of the last fit."""
        return self.X2

    def GetFunction(self):
        """Returns the GF1 function used by the fit."""
        return self.GF

    def InfBounds(self):
        """Returns unbounded limits for every parameter."""
        n_params = self.GF.GetNumberOfParameters()
        return ([-np.inf] * n_params, [np.inf] * n_params)
