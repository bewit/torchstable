# this is a torch implementation of https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.levy_stable.html 
# thanks to the authors of SciPy and of the referenced file and its dependencies in particular
# for docs, please see scipy.stats.levy_stable
# I tried to stay as close to the original numpy implementation as possible (and kind of sensible)
# nonetheless, I'm not that familiar with the computational and mathematical tweaks here, and the implemented solution might appear hacky
# only the piecewise integration works, and as all integrations are carried out using MonteCarlo integration, the results are approximations and the respective errors are not quantified here (this is partly due to me not knowing how to do this and not having the time for researching it right now)
# precision can be increased by choosing more MCIntegration evaluation nodes or increasing the number of repetitions per integration
# according to the WLLN, the results converge in the infinite, but I cannot derive the convergence rate here; and the computation gets heavy quite quickly
# please note that comparison to the scipy implementation still has to be carried out
# if you want to contact me, you can find me via ORCID: https://orcid.org/0000-0002-6861-7301
import math
from numbers import Number
from functools import partial

import torch
from torch import distributions as dist
from torch.distributions import constraints, Distribution
from torch.distributions.utils import broadcast_all
from torchquad import MonteCarlo


__all__ = ["TorchStable"]
torch.set_default_dtype(torch.double)
M_PI = torch.tensor(math.pi)
M_PI_2 = torch.tensor(1.57079632679489661923)
M_1_PI = torch.tensor(0.31830988618379067154)
M_2_PI = torch.tensor(0.63661977236758134308) 

# heavily recommend MonteCarlo here, as the functions are too complex for Simpson/Trapezoid/Boole
integrator = MonteCarlo()
integrator_params = {"N": 20000}
# Parameters for the LBFGS optimization used for determining tighter integration bounds for the CDF computation if alpha > 1.0
LBFGS_EPOCHS = 20
LBFGS_LR = 0.1
# repetitions of all integral computations where alpha != 1.0. the mean over all computations is returned. 1 is sufficient, more increases precision
INTEGRATION_REPETITIONS = 5
# repetitions of all integral computations where alpha == 1.0. the mean over all computations is returned. I suggest at least 10
INTEGRATION_REPETITIONS_ALPHA_1 = 20


class Nolan:
    def __init__(self, alpha: torch.Tensor, beta: torch.Tensor, x0: torch.Tensor):
        self.alpha = alpha
        self.zeta = -beta * torch.tan(M_PI_2 * alpha)

        if alpha != 1.:
            self.xi = torch.atan(-self.zeta) / alpha
            self.zeta_prefactor = torch.pow(torch.pow(self.zeta, 2) + 1., -1. / (2. * (alpha - 1.)))
            self.alpha_exp = alpha / (alpha - 1.)
            self.alpha_xi = torch.atan(-self.zeta)
            self.zeta_offset = x0 - self.zeta

            if alpha < 1.:
                self.c1 = 0.5 - self.xi * M_1_PI
                self.c3 = M_1_PI
            else: # alpha > 1.
                self.c1 = torch.tensor(1.)
                self.c3 = -M_1_PI

            self.c2 = alpha * M_1_PI / torch.abs(alpha - 1.) / (x0 - self.zeta)
            self.g = self.g_alpha_ne_one

        else: # alpha == 1.
            self.xi = M_PI_2
            self.two_beta_div_pi = beta * M_2_PI
            self.pi_div_two_beta = M_PI_2 / beta
            self.x0_div_term = x0 / self.two_beta_div_pi
            self.c1 = torch.tensor(0.)
            self.c2 = 0.5 / torch.abs(beta)
            self.c3 = M_1_PI
            self.g = self.g_alpha_eq_one

    
    def g_alpha_ne_one(self, theta: torch.Tensor) -> torch.Tensor:
        if torch.all(torch.isclose(theta, -self.xi)):
            if self.alpha < 1.:
                return torch.tensor(0.)
            else:
                return torch.tensor(float('inf'))
        if torch.all(torch.isclose(theta, M_PI_2.type(theta.dtype))):
            if self.alpha < 1.:
                return torch.tensor(float('inf'))
            else:
                return torch.tensor(0.)
            
        cos_theta = torch.cos(theta)
        return (
            self.zeta_prefactor
            * torch.pow(
                cos_theta
                / torch.sin(self.alpha_xi + self.alpha * theta)
                * self.zeta_offset, self.alpha_exp)
            * torch.cos(self.alpha_xi + (self.alpha - 1.) * theta)
            / cos_theta
        )
    
    
    def g_alpha_eq_one(self, theta: torch.Tensor) -> torch.Tensor: 
        if torch.all(torch.isclose(theta, -self.xi)):
            return torch.zeros_like(theta)
        if torch.all(theta == M_PI_2.type(theta.dtype)):
            temp = torch.zeros_like(theta)
            temp.fill_(float('inf'))
            return temp
        
        return (
            (1. + theta * self.two_beta_div_pi)
            * torch.exp((self.pi_div_two_beta + theta) * torch.tan(theta) - self.x0_div_term)
            / torch.cos(theta)
        )


def _transform_half_real_line_to_unit_interval(fct, t):
    """Given a function that is defined on [0, +inf), transform it to the domain [0, 1]"""
    return fct(t/(1-t)) / torch.pow(1-t, 2)


def _Phi_Z0(alpha, t):
    return (
        -torch.tan(M_PI * alpha / 2) * (torch.pow(torch.abs(t), (1 - alpha)) - 1)
        if alpha != 1
        else -2.0 * torch.log(torch.abs(t)) / M_PI
    )


def _Phi_Z1(alpha, t):
    return (
        torch.tan(M_PI * alpha / 2)
        if alpha != 1
        else -2.0 * torch.log(torch.abs(t)) / M_PI
    )


def _cf(Phi, t, alpha, beta):
    return torch.exp(
        -(torch.pow(torch.abs(t), alpha) * (1 - 1j * beta * torch.sign(t) * Phi(alpha, t)))
    )


_cf_Z0 = partial(_cf, _Phi_Z0)
_cf_Z1 = partial(_cf, _Phi_Z1)


def _pdf_single_value_cf_integrate(Phi, x, alpha, beta, **kwds):
    # NOTE
    # the direct numeric integration method of stable distribtions fails easily
    # this can be observed using following different approaches:
    # - hermite-gauss quadrature, because the distribution to integrate does not necessarily satisfy all assumptions of hermite-gauss, i.e. it is not of the form x**2
    # - numpy.quad: while yielding good results around the critical mass of the PDF/CDF, it diverges on other areas and is not reliable
    # - torchquad: requires bound integration interval, which is infeasible for this task. transforming the indefinite integral to the unit interval yields diverging results
    # the only reliable approach to the computation of PDF and CDF of stable RVs is the piecewise integration developed by Nolan1999 and implemented by scipy.stats.levy_stable
    import warnings
    warnings.warn("This method is buggy!", RuntimeWarning)

    # split up integral and use change of variables to project to unit interval

    def integrand1(t):
        if torch.all(t == 0):
            return torch.zeros_like(t)
        return torch.exp(-(torch.pow(t, alpha))) * (
            torch.cos(beta * torch.pow(t, alpha) * Phi(alpha, t))
        )
    
    def integrand2(t):
        if torch.all(t == 0):
            return torch.zeros_like(t)
        return torch.exp(-(torch.pow(t, alpha))) * (
            torch.sin(beta * torch.pow(t, alpha) * Phi(alpha, t))
        )

    transformed_integrand1 = partial(_transform_half_real_line_to_unit_interval, integrand1)
    transformed_integrand2 = partial(_transform_half_real_line_to_unit_interval, integrand2)

    max_iter = INTEGRATION_REPETITIONS
    vals1 = torch.empty(size=(max_iter, 1))
    vals2 = torch.empty(size=(max_iter, 1))
    for i in range(max_iter):
        int1 = integrator.integrate(transformed_integrand1, dim=1, integration_domain=[[0, 1]], **integrator_params)
        vals1[i] = int1
        int2 = integrator.integrate(transformed_integrand2, dim=1, integration_domain=[[0, 1]], **integrator_params)
        vals2[i] = int2

    int1 = torch.nanmean(vals1) if not torch.all(torch.isnan(vals1)) else torch.tensor(0.0)
    int2 = torch.nanmean(vals2) if not torch.all(torch.isnan(vals2)) else torch.tensor(0.0)

    return (int1 + int2) / M_PI

_pdf_single_value_cf_integrate_Z0 = partial(_pdf_single_value_cf_integrate, _Phi_Z0)
_pdf_single_value_cf_integrate_Z1 = partial(_pdf_single_value_cf_integrate, _Phi_Z1)


def _nolan_round_x_near_zeta(x0, alpha, zeta, x_tol_near_zeta):
    if torch.abs(x0 - zeta) < x_tol_near_zeta * torch.pow(alpha, (1/alpha)):
        x0 = zeta
    return x0


def _nolan_round_difficult_input(x0, alpha, beta, zeta, x_tol_near_zeta, alpha_tol_near_one):
    if torch.abs(alpha - 1.) < alpha_tol_near_one:
        alpha = torch.tensor(1.0)
    x0 = _nolan_round_x_near_zeta(x0, alpha, zeta, x_tol_near_zeta)
    return x0, alpha, beta


def _pdf_single_value_piecewise_Z1(x, alpha, beta, **kwds):
    zeta = -beta * torch.tan(M_PI * alpha / 2.0)
    x0 = x + zeta if alpha != 1. else x
    
    return _pdf_single_value_piecewise_Z0(x0, alpha, beta, **kwds)


def _pdf_single_value_piecewise_Z0(x0, alpha, beta, **kwds):
    x_tol_near_zeta = kwds.get("piecewise_x_tol_near_zeta", torch.tensor(0.005))
    alpha_tol_near_one = kwds.get("piecewise_alpha_tol_near_one", torch.tensor(0.005))

    zeta = -beta * torch.tan(M_PI * alpha / 2.0)
    x0, alpha, beta = _nolan_round_difficult_input(x0, alpha, beta, zeta, x_tol_near_zeta, alpha_tol_near_one)

    if alpha == 2.0:
        def _norm_pdf(x):
            return torch.exp(-torch.pow(x, 2) / 2.0) / torch.sqrt(2. * M_PI)
        return _norm_pdf(x0 / torch.sqrt(torch.tensor(2.0))) / torch.sqrt(torch.tensor(2.0))
    elif alpha == 0.5 and beta == 1.0:
        _x = x0 + 1.
        if _x <= 0:
            return torch.tensor(0.0)
        return torch.tensor(1.0 / torch.sqrt(2.0 * M_PI * _x) / _x * torch.exp(-1.0 / (2 * _x)))
    # elif alpha == 0.5 and beta == 0.0 and x0 != 0:
        # S, C = sc.fresnel([1 / np.sqrt(2 * np.pi * np.abs(x0))])
        # arg = 1 / (4 * np.abs(x0))
        # return (
        #     np.sin(arg) * (0.5 - S[0]) + np.cos(arg) * (0.5 - C[0])
        # ) / np.sqrt(2 * np.pi * np.abs(x0) ** 3)
        # cannot implement this as fresnel integrals are not available in torch
    elif alpha == 1.0 and beta == 0.0:
        return 1.0 / (1.0 + torch.pow(x0, 2)) / M_PI
    
    return _pdf_single_value_piecewise_post_rounding_Z0(x0, alpha, beta, x_tol_near_zeta)


def _pdf_single_value_piecewise_post_rounding_Z0(x0, alpha, beta, x_tol_near_zeta):
    _nolan = Nolan(alpha, beta, x0)
    zeta = _nolan.zeta
    xi = _nolan.xi
    c2 = _nolan.c2
    g = _nolan.g

    x0 = _nolan_round_x_near_zeta(x0, alpha, zeta, x_tol_near_zeta)

    if torch.isclose(x0, zeta):
        return(
            torch.exp(torch.lgamma(1. + 1./alpha))
            * torch.cos(xi)
            / M_PI
            / (torch.pow(1. + torch.pow(zeta, 2), (1./alpha/2.)))
        )
    elif x0 < zeta:
        return _pdf_single_value_piecewise_post_rounding_Z0(
            -x0, alpha, -beta, x_tol_near_zeta
        )
    
    if torch.isclose(-xi, M_PI/2.):
        return torch.tensor(0.)
    

    def integrand(theta):
        g_1 = g(theta)
        g_1.nan_to_num(posinf=0.0)
        g_1[g_1 < 0] = 0.0
        return g_1 * torch.exp(-g_1)
    
    # bisect and np.quad (which is quasi Fortran's quadpack) are not available
    # instead integrate with MonteCarlo and hope for the best
    if alpha == 1.0:
        # for alpha==1.0, most samples of the MCIntegrator evaluate to 0, hence we cannot compute the integral
        # repeating and averaging this process over a small number of evaluation nodes (like N=1000) helps! but takes time
        # this is a hacky workaround for the edgecase where alpha==1.0, which should happen quite rarely
        # the proper way would to locate the critical mass of the extremely peaked function here and limit the integration region respectively
        # but ATM I have no idea how to do this and sadly need to prioritize other tasks
        max_iter = INTEGRATION_REPETITIONS_ALPHA_1
        vals = torch.empty(size=(max_iter, 1))
        for i in range(max_iter):
            intg = integrator.integrate(integrand, dim=1, N=1000, integration_domain=[[-xi, M_PI/2.]])
            vals[i] = intg
        intg = torch.nanmean(vals) if not torch.all(torch.isnan(vals)) else torch.tensor(0.0)

    else:
        max_iter = INTEGRATION_REPETITIONS
        vals = torch.empty(size=(max_iter, 1))
        for i in range(max_iter):
            intg = integrator.integrate(integrand, dim=1, **integrator_params, integration_domain=[[-xi, M_PI/2.]])
            vals[i] = intg
        intg = torch.nanmean(vals) if not torch.all(torch.isnan(vals)) else torch.tensor(0.0)


    return c2 * intg


def _cdf_single_value_piecewise_Z1(x, alpha, beta, **kwds):
    zeta = -beta * torch.tan(M_PI * alpha / 2.)
    x0 = x + zeta if alpha != 1. else x

    return _cdf_single_value_piecewise_Z0(x0, alpha, beta, **kwds)
    

def _cdf_single_value_piecewise_Z0(x0, alpha, beta, **kwds):
    
    x_tol_near_zeta = kwds.get("piecewise_x_tol_near_zeta", torch.tensor(0.005))
    alpha_tol_near_one = kwds.get("piecewise_alpha_tol_near_one", torch.tensor(0.005))

    zeta = -beta * torch.tan(M_PI * alpha / 2.)
    x0, alpha, beta = _nolan_round_difficult_input(x0, alpha, beta, zeta, x_tol_near_zeta, alpha_tol_near_one)

    if alpha == 2.0:
        dist.Normal(torch.tensor([0.0]), torch.tensor([1.0])).cdf(x0 / torch.sqrt(torch.tensor(2.)))
    elif alpha == 0.5 and beta == 1.0:
        _x = x0 + 1.
        if _x <= 0:
            return torch.tensor(0.)
        return (1. - torch.erf(torch.sqrt(0.5 / _x)))
    elif alpha == 1.0 and beta == 0.0:
        return 0.5 + torch.arctan(x0) / M_PI
    
    return _cdf_single_value_piecewise_post_rounding_Z0(x0, alpha, beta, x_tol_near_zeta)


def _cdf_single_value_piecewise_post_rounding_Z0(x0, alpha, beta, x_tol_near_zeta):
    _nolan = Nolan(alpha, beta, x0)
    zeta = _nolan.zeta
    xi = _nolan.xi
    c1 = _nolan.c1
    c3 = _nolan.c3
    g = _nolan.g

    x0 = _nolan_round_x_near_zeta(x0, alpha, zeta, x_tol_near_zeta)

    if (alpha == 1. and beta < 0.) or x0 < zeta:
        return 1 - _cdf_single_value_piecewise_post_rounding_Z0(-x0, alpha, -beta, x_tol_near_zeta)
    elif torch.isclose(x0, zeta):
        return 0.5 - xi / M_PI
    
    if torch.isclose(-xi, M_PI/2.):
        return c1
    
    def integrand(theta):
        g_1 = g(theta)
        return torch.exp(-g_1)
    
    left_support = -xi
    right_support = M_PI / 2.

    if alpha > 1.:
        # TODO: i'm not really sure if the purpose of the following optimization is just to reduce the interval of integration lateron. if so, we could probably skip this. needs evaluation
        if not torch.isclose(integrand(-xi), torch.tensor(0.)):
            param = torch.clone(-xi).detach().requires_grad_(True)
            opt = torch.optim.LBFGS([param], lr=LBFGS_LR)
            def closure():
                opt.zero_grad()
                loss = integrand(param)
                loss.requires_grad_(True)
                loss.backward()
                return loss
            for epoch in range(LBFGS_EPOCHS):
                opt.step(closure=closure)
            param.clamp(-xi, M_PI/2.)
            left_support = param
    else: # alpha < 1.
        if not torch.isclose(integrand(M_PI/2.), torch.tensor(0.)):
            param = torch.tensor(M_PI/2.).requires_grad_(True)
            opt = torch.optim.LBFGS([param], lr=LBFGS_LR)
            def closure():
                opt.zero_grad()
                loss = integrand(param)
                loss.requires_grad_(True)
                loss.backward()
                return loss
            for epoch in range(LBFGS_EPOCHS):
                opt.step(closure=closure)
            param.clamp(-xi, M_PI/2.)
            right_support = param

    # intg = integrator.integrate(integrand, dim=1, **integrator_params, integration_domain=[[left_support, right_support]])
    if alpha == 1.0:
        # for alpha==1.0, most samples of the MCIntegrator evaluate to 0, hence we cannot compute the integral
        # repeating and averaging this process over a small number of evaluation nodes (like N=1000) helps! but takes time
        # this is a hacky workaround for the edgecase where alpha==1.0, which should happen quite rarely
        # the proper way would to locate the critical mass of the extremely peaked function here and limit the integration region respectively
        # but ATM I have no idea how to do this and sadly need to prioritize other tasks
        max_iter = INTEGRATION_REPETITIONS_ALPHA_1
        vals = torch.empty(size=(max_iter, 1))
        for i in range(max_iter):
            intg = integrator.integrate(integrand, dim=1, N=1000, integration_domain=[[left_support, right_support]])
            vals[i] = intg
        intg = torch.nanmean(vals) if not torch.all(torch.isnan(vals)) else torch.tensor(0.0)
    
    else:
        max_iter = INTEGRATION_REPETITIONS
        vals = torch.empty(size=(max_iter, 1))
        for i in range(max_iter):
            intg = integrator.integrate(integrand, dim=1, **integrator_params, integration_domain=[[left_support, right_support]])
            vals[i] = intg
        intg = torch.nanmean(vals) if not torch.all(torch.isnan(vals)) else torch.tensor(0.0)

    return c1 + c3 * intg


def _rvs_Z1(alpha, beta, TH, aTH, bTH, cosTH, tanTH, W):
    raise NotImplementedError("Sampling not implemented")


def _fitstart_mcculloch_S0(data, epsilon):
    alpha, beta, loc1, scale = _fitstart_mcculloch_S1(data, epsilon=epsilon)

    if alpha != 1.0:
        loc0 = loc1 + beta * scale * torch.tan(M_PI * alpha / 2.0)
    else:
        loc0 = loc1 + 2 * beta * scale * torch.log(scale) / M_PI

    return alpha, beta, loc0, scale


def _fitstart_mcculloch_S1(data, epsilon):
    # McCullock1986 method
    # this is implemented using numpy and scipy, as torch does not offer the RectBivariateSpline needed for the estimtation here
    # as the estimation part only happens once during structure learning, this is fine
    # computation of cdf/pdf can still happen on the gpu
    import numpy as np
    from scipy.interpolate import RectBivariateSpline

    nu_alpha_range = [2.439, 2.5, 2.6, 2.7, 2.8, 3, 3.2, 3.5, 4,
                      5, 6, 8, 10, 15, 25]
    nu_beta_range = [0, 0.1, 0.2, 0.3, 0.5, 0.7, 1]

    alpha_table = np.array([
        [2.000, 2.000, 2.000, 2.000, 2.000, 2.000, 2.000],
        [1.916, 1.924, 1.924, 1.924, 1.924, 1.924, 1.924],
        [1.808, 1.813, 1.829, 1.829, 1.829, 1.829, 1.829],
        [1.729, 1.730, 1.737, 1.745, 1.745, 1.745, 1.745],
        [1.664, 1.663, 1.663, 1.668, 1.676, 1.676, 1.676],
        [1.563, 1.560, 1.553, 1.548, 1.547, 1.547, 1.547],
        [1.484, 1.480, 1.471, 1.460, 1.448, 1.438, 1.438],
        [1.391, 1.386, 1.378, 1.364, 1.337, 1.318, 1.318],
        [1.279, 1.273, 1.266, 1.250, 1.210, 1.184, 1.150],
        [1.128, 1.121, 1.114, 1.101, 1.067, 1.027, 0.973],
        [1.029, 1.021, 1.014, 1.004, 0.974, 0.935, 0.874],
        [0.896, 0.892, 0.884, 0.883, 0.855, 0.823, 0.769],
        [0.818, 0.812, 0.806, 0.801, 0.780, 0.756, 0.691],
        [0.698, 0.695, 0.692, 0.689, 0.676, 0.656, 0.597],
        [0.593, 0.590, 0.588, 0.586, 0.579, 0.563, 0.513]]).T
    
    beta_table = np.array([
        [0, 2.160, 1.000, 1.000, 1.000, 1.000, 1.000],
        [0, 1.592, 3.390, 1.000, 1.000, 1.000, 1.000],
        [0, 0.759, 1.800, 1.000, 1.000, 1.000, 1.000],
        [0, 0.482, 1.048, 1.694, 1.000, 1.000, 1.000],
        [0, 0.360, 0.760, 1.232, 2.229, 1.000, 1.000],
        [0, 0.253, 0.518, 0.823, 1.575, 1.000, 1.000],
        [0, 0.203, 0.410, 0.632, 1.244, 1.906, 1.000],
        [0, 0.165, 0.332, 0.499, 0.943, 1.560, 1.000],
        [0, 0.136, 0.271, 0.404, 0.689, 1.230, 2.195],
        [0, 0.109, 0.216, 0.323, 0.539, 0.827, 1.917],
        [0, 0.096, 0.190, 0.284, 0.472, 0.693, 1.759],
        [0, 0.082, 0.163, 0.243, 0.412, 0.601, 1.596],
        [0, 0.074, 0.147, 0.220, 0.377, 0.546, 1.482],
        [0, 0.064, 0.128, 0.191, 0.330, 0.478, 1.362],
        [0, 0.056, 0.112, 0.167, 0.285, 0.428, 1.274]]).T
    

    alpha_range = [2, 1.9, 1.8, 1.7, 1.6, 1.5, 1.4, 1.3, 1.2, 1.1,
                   1, 0.9, 0.8, 0.7, 0.6, 0.5][::-1]
    beta_range = [0, 0.25, 0.5, 0.75, 1]

    nu_c_table = np.array([
        [1.908, 1.908, 1.908, 1.908, 1.908],
        [1.914, 1.915, 1.916, 1.918, 1.921],
        [1.921, 1.922, 1.927, 1.936, 1.947],
        [1.927, 1.930, 1.943, 1.961, 1.987],
        [1.933, 1.940, 1.962, 1.997, 2.043],
        [1.939, 1.952, 1.988, 2.045, 2.116],
        [1.946, 1.967, 2.022, 2.106, 2.211],
        [1.955, 1.984, 2.067, 2.188, 2.333],
        [1.965, 2.007, 2.125, 2.294, 2.491],
        [1.980, 2.040, 2.205, 2.435, 2.696],
        [2.000, 2.085, 2.311, 2.624, 2.973],
        [2.040, 2.149, 2.461, 2.886, 3.356],
        [2.098, 2.244, 2.676, 3.265, 3.912],
        [2.189, 2.392, 3.004, 3.844, 4.775],
        [2.337, 2.634, 3.542, 4.808, 6.247],
        [2.588, 3.073, 4.534, 6.636, 9.144]])[::-1].T

    nu_zeta_table = np.array([
        [0, 0.000, 0.000, 0.000, 0.000],
        [0, -0.017, -0.032, -0.049, -0.064],
        [0, -0.030, -0.061, -0.092, -0.123],
        [0, -0.043, -0.088, -0.132, -0.179],
        [0, -0.056, -0.111, -0.170, -0.232],
        [0, -0.066, -0.134, -0.206, -0.283],
        [0, -0.075, -0.154, -0.241, -0.335],
        [0, -0.084, -0.173, -0.276, -0.390],
        [0, -0.090, -0.192, -0.310, -0.447],
        [0, -0.095, -0.208, -0.346, -0.508],
        [0, -0.098, -0.223, -0.380, -0.576],
        [0, -0.099, -0.237, -0.424, -0.652],
        [0, -0.096, -0.250, -0.469, -0.742],
        [0, -0.089, -0.262, -0.520, -0.853],
        [0, -0.078, -0.272, -0.581, -0.997],
        [0, -0.061, -0.279, -0.659, -1.198]])[::-1].T
    
    
    psi_1 = RectBivariateSpline(nu_beta_range, nu_alpha_range,
                                alpha_table, kx=1, ky=1, s=0)

    def psi_1_1(nu_beta, nu_alpha):
        return psi_1(nu_beta, nu_alpha) \
            if nu_beta > 0 else psi_1(-nu_beta, nu_alpha)

    psi_2 = RectBivariateSpline(nu_beta_range, nu_alpha_range,
                                beta_table, kx=1, ky=1, s=0)

    def psi_2_1(nu_beta, nu_alpha):
        return psi_2(nu_beta, nu_alpha) \
            if nu_beta > 0 else -psi_2(-nu_beta, nu_alpha)

    phi_3 = RectBivariateSpline(beta_range, alpha_range, nu_c_table,
                                kx=1, ky=1, s=0)

    def phi_3_1(beta, alpha):
        return phi_3(beta, alpha) if beta > 0 else phi_3(-beta, alpha)

    phi_5 = RectBivariateSpline(beta_range, alpha_range, nu_zeta_table,
                                kx=1, ky=1, s=0)

    def phi_5_1(beta, alpha):
        return phi_5(beta, alpha) if beta > 0 else -phi_5(-beta, alpha)

    # quantiles
    p05 = np.percentile(data, 5)
    p50 = np.percentile(data, 50)
    p95 = np.percentile(data, 95)
    p25 = np.percentile(data, 25)
    p75 = np.percentile(data, 75)

    nu_alpha = (p95 - p05) / (p75 - p25)
    nu_beta = (p95 + p05 - 2 * p50) / (p95 - p05)

    if nu_alpha >= 2.439:
        alpha = np.clip(psi_1_1(nu_beta, nu_alpha)[0, 0], epsilon, 2.)
        beta = np.clip(psi_2_1(nu_beta, nu_alpha)[0, 0], -1.0, 1.0)
    else:
        alpha = 2.0
        beta = np.sign(nu_beta)
    c = (p75 - p25) / phi_3_1(beta, alpha)[0, 0]
    zeta = p50 + c * phi_5_1(beta, alpha)[0, 0]
    delta = zeta-beta*c*np.tan(np.pi*alpha/2.) if alpha != 1. else zeta

    return (torch.tensor(alpha), torch.tensor(beta), torch.tensor(delta), torch.tensor(c))
    


class TorchStable(Distribution):
    r"""
    Creates a stable distribution (also called alpha-stable, levy-stable) parametrized by 
    :attr:`alpha`, :attr:`beta`, :attr:`scale`, and :attr:`loc`.
    """
    arg_constraints = {"alpha": constraints.interval(0.0, 2.0), "beta": constraints.interval(-1.0, 1.0), "scale": constraints.positive, "loc": constraints.real}
    support = constraints.real # TODO: actually the support depends on beta (and alpha) e.g. see Levy distribution
    has_rsample = False # TODO: implement if necessary

    # levy_stable parameters
    parametrization = "S1"
    pdf_default_method = "piecewise"
    cdf_default_method = "piecewise"
    piecewise_x_tol_near_zeta = torch.tensor(0.005)
    piecewise_alpha_tol_near_one = torch.tensor(0.005)

    def __init__(self, alpha, beta, loc, scale, validate_args=None):
        self.alpha, self.beta, self.loc, self.scale = broadcast_all(alpha, beta, loc, scale)
        if isinstance(loc, Number) and isinstance(scale, Number):
            batch_shape = torch.Size()
        else:
            batch_shape = self.loc.size()
        super().__init__(batch_shape=batch_shape, validate_args=validate_args)

    def _parametrization(self):
        allowed = ("S0", "S1")
        pz = self.parametrization
        if pz not in allowed:
            raise RuntimeError(f"Parametrization '{pz}' not in supported list: {allowed}")
        return pz
    
    def expand(self, batch_shape, _instance=None):
        raise NotImplementedError("I don't know where is this relevant yet")
    

    def sample(self, sample_shape=torch.Size()):
        raise NotImplementedError("Sampling not implemented")
    

    def rsample(self, sample_shape=torch.Size()):
        raise NotImplementedError("Sampling not implemented")
    

    def pdf(self, value: torch.Tensor) -> torch.Tensor:
        if self._parametrization() == "S0":
            return self._pdf(value, self.alpha, self.beta, self.loc, self.scale)
        elif self._parametrization() == "S1":
            alpha = self.alpha
            beta = self.beta
            loc = self.loc
            scale = self.scale
            if torch.all(torch.reshape(self.alpha, (1, -1))[0, :] != 1.):
                return self._pdf(value, alpha, beta, loc, scale)
            else:
                value_shape = value.shape
                uniq_param_pairs = torch.cat((alpha.reshape(-1, 1), beta.reshape(-1, 1)), dim=-1)

                value, alpha, beta = broadcast_all(value, alpha, beta)
                value = torch.reshape(value, (1, -1))[0, :]
                alpha = torch.reshape(alpha, (1, -1))[0, :]
                beta = torch.reshape(beta, (1, -1))[0, :]
                data_in = torch.dstack((value, alpha, beta))[0]
                data_out = torch.empty(size=(len(data_in), 1))

                for pair in uniq_param_pairs:
                    _alpha, _beta = pair
                    if _alpha == 1.:
                        _delta = loc + 2*_beta*scale * torch.log(scale) / M_PI
                    else:
                        _delta = loc
                    _delta = (loc + 2*_beta*scale * torch.log(scale) / M_PI if _alpha==1.0 else loc)
                    data_mask = torch.all(data_in[:, 1:] == pair, dim=-1)
                    _x = data_in[data_mask, 0]
                    data_out[data_mask] = self._pdf(_x, _alpha, _beta, loc=_delta, scale=scale).reshape(len(_x), 1)

                data_out = torch.reshape(data_out, value_shape) 

                return data_out       


    

    def _pdf(self, value, alpha, beta, loc, scale) -> torch.Tensor:
        if self._parametrization() == "S0":
            _pdf_single_value_piecewise = _pdf_single_value_piecewise_Z0
            _pdf_single_value_cf_integrate = _pdf_single_value_cf_integrate_Z0
            _cf = _cf_Z0
        elif self._parametrization() == "S1":
            _pdf_single_value_piecewise = _pdf_single_value_piecewise_Z1
            _pdf_single_value_cf_integrate = _pdf_single_value_cf_integrate_Z1
            _cf = _cf_Z1

        pdf_default_method_name = self.pdf_default_method
        if pdf_default_method_name in ("piecewise", "best", "zolotarev"):
            pdf_single_value_method = _pdf_single_value_piecewise
        elif pdf_default_method_name in ("dni", "quadrature"):
            pdf_single_value_method = _pdf_single_value_cf_integrate
        else:
            raise ValueError(f"PDF method '{self.pdf_default_method}' not supported")
        
        pdf_single_value_kwds = {
            "piecewise_x_tol_near_zeta": self.piecewise_x_tol_near_zeta,
            "piecewise_alpha_tol_near_one": self.piecewise_alpha_tol_near_one,
        }

        # retrieve original shape for reshaping lateron
        loc, scale, value = broadcast_all(loc, scale, value)
        value_shape = value.shape
        # standardize
        value = (value - loc) / scale
         
        uniq_param_pairs = torch.cat((alpha.reshape(-1, 1), beta.reshape(-1, 1)), dim=-1)
        # print("uniq_param_pairs", uniq_param_pairs.shape)

        value, alpha, beta = broadcast_all(value, alpha, beta)
        # print("value", value.shape)
        # print("alpha", alpha.shape)
        # print("beta", beta.shape)
        value = torch.reshape(value, (1, -1))[0, :]
        alpha = torch.reshape(alpha, (1, -1))[0, :]
        beta = torch.reshape(beta, (1, -1))[0, :]
        # print("value", value.shape)
        # print("alpha", alpha.shape)
        # print("beta", beta.shape)
        data_in = torch.dstack((value, alpha, beta))[0]
        # print("data_in", data_in.shape)
        # print(data_in)
        data_out = torch.empty(size=(len(data_in), 1))

        # uniq_param_pairs = torch.unique(data_in[:, 1:], dim=0)
        for pair in uniq_param_pairs:
            data_mask = torch.all(data_in[:, 1:] == pair, dim=-1)
            data_subset = data_in[data_mask]
            data_out[data_mask] = torch.tensor(
                [
                    pdf_single_value_method(_value, _alpha, _beta, **pdf_single_value_kwds)
                    for _value, _alpha, _beta in data_subset
                ]
            ).reshape(len(data_subset), 1)

        # reshape to original tensor
        data_out = torch.reshape(data_out, value_shape)        
        # account for standardization
        data_out = data_out / scale

        return data_out


    def log_prob(self, value: torch.Tensor) -> torch.Tensor:
        return torch.log(self.pdf(value))
    

    def cdf(self, value: torch.Tensor) -> torch.Tensor:
        if self._parametrization() == "S0":
            return self._cdf(value, self.alpha, self.beta, self.loc, self.scale)
        elif self._parametrization() == "S1":
            alpha = self.alpha
            beta = self.beta
            loc = self.loc
            scale = self.scale
            if torch.all(torch.reshape(self.alpha, (1, -1))[0, :] != 1.):
                return self._cdf(value, alpha, beta, loc, scale)
            else:
                value_shape = value.shape
                uniq_param_pairs = torch.cat((alpha.reshape(-1, 1), beta.reshape(-1, 1)), dim=-1)

                value, alpha, beta = broadcast_all(value, alpha, beta)
                value = torch.reshape(value, (1, -1))[0, :]
                alpha = torch.reshape(alpha, (1, -1))[0, :]
                beta = torch.reshape(beta, (1, -1))[0, :]
                data_in = torch.dstack((value, alpha, beta))[0]
                data_out = torch.empty(size=(len(data_in), 1))

                for pair in uniq_param_pairs:
                    _alpha, _beta = pair
                    if _alpha == 1.:
                        _delta = loc + 2*_beta*scale * torch.log(scale) / M_PI
                    else:
                        _delta = loc
                    _delta = (loc + 2*_beta*scale * torch.log(scale) / M_PI if _alpha==1.0 else loc)
                    data_mask = torch.all(data_in[:, 1:] == pair, dim=-1)
                    _x = data_in[data_mask, 0]
                    data_out[data_mask] = self._cdf(_x, _alpha, _beta, loc=_delta, scale=scale).reshape(len(_x), 1)

                data_out = torch.reshape(data_out, value_shape) 

                return data_out       
    

    def _cdf(self, value, alpha, beta, loc, scale):
        if self._parametrization() == "S0":
            _cdf_single_value_piecewise = _cdf_single_value_piecewise_Z0
            _cf = _cf_Z0
        elif self._parametrization() == "S1":
            _cdf_single_value_piecewise = _cdf_single_value_piecewise_Z1
            _cf = _cf_Z1

        cdf_default_method_name = self.cdf_default_method
        if cdf_default_method_name in ("piecewise", "best", "zolotarev"):
            cdf_single_value_method = _cdf_single_value_piecewise
        else:
            raise ValueError(f"PDF method '{self.pdf_default_method}' not supported")
        
        cdf_single_value_kwds = {
            "piecewise_x_tol_near_zeta": self.piecewise_x_tol_near_zeta,
            "piecewise_alpha_tol_near_one": self.piecewise_alpha_tol_near_one,
        }

        # retrieve original shape for reshaping lateron
        loc, scale, value = broadcast_all(loc, scale, value)
        value_shape = value.shape
        # standardize
        value = (value - loc) / scale
         
        uniq_param_pairs = torch.cat((alpha.reshape(-1, 1), beta.reshape(-1, 1)), dim=-1)

        value, alpha, beta = broadcast_all(value, alpha, beta)
        value = torch.reshape(value, (1, -1))[0, :]
        alpha = torch.reshape(alpha, (1, -1))[0, :]
        beta = torch.reshape(beta, (1, -1))[0, :]
        data_in = torch.dstack((value, alpha, beta))[0]
        data_out = torch.empty(size=(len(data_in), 1))

        for pair in uniq_param_pairs:
            data_mask = torch.all(data_in[:, 1:] == pair, dim=-1)
            data_subset = data_in[data_mask]
            data_out[data_mask] = torch.tensor(
                [
                    cdf_single_value_method(_value, _alpha, _beta, **cdf_single_value_kwds)
                    for _value, _alpha, _beta in data_subset
                ]
            ).reshape(len(data_subset), 1)

        # reshape to original tensor
        data_out = torch.reshape(data_out, value_shape)        

        return data_out
    

    def _fitstart(self, data, epsilon=1e-6):
        # TODO: implemented MLE method as described in Nolan2020
        if self._parametrization() == "S0":
            _fitstart = _fitstart_mcculloch_S0
        elif self._parametrization() == "S1":
            _fitstart = _fitstart_mcculloch_S1
        return _fitstart(data, epsilon=epsilon)


    def icdf(self, value: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError
    

    def entropy(self):
        raise NotImplementedError
    


if __name__ == "__main__":
    import numpy as np
    from scipy.stats import levy_stable
    from tabulate import tabulate

    params_set = [
        {"alpha": 2.0, "beta": 0.0, "loc": 0.0, "scale": 1./np.sqrt(2.)},
        {"alpha": 1.5, "beta": 0.5, "loc": 0.0, "scale": 1.0},
        {"alpha": 1.5, "beta": 0.5, "loc": 1000.0, "scale": 100.0},
        {"alpha": 1.0, "beta": 0.0, "loc": 0.0, "scale": 1.0},
        {"alpha": 1.0, "beta": 1.0, "loc": 4.2, "scale": 1.0},
        {"alpha": 1.0, "beta": -1.0, "loc": 4.2, "scale": 1.0},
        {"alpha": 0.5, "beta": 0.5, "loc": 0.0, "scale": 0.5},
        {"alpha": 0.5, "beta": 1.0, "loc": 0.0, "scale": 0.5},
        {"alpha": 0.5, "beta": -1.0, "loc": 0.0, "scale": 0.5},
    ]

    for params in params_set:
        alpha = params["alpha"]
        beta = params["beta"]
        loc = params["loc"]
        scale = params["scale"]

        torch_stable = TorchStable(alpha=torch.tensor(alpha), beta=torch.tensor(beta), loc=torch.tensor(loc), scale=torch.tensor(scale))
        scipy_stable = levy_stable(alpha=alpha, beta=beta, loc=loc, scale=scale)


        logvals = np.logspace(-5, 5, 11)
        x = list((-1) * logvals) + list(logvals)
        x.sort()
        data = torch.tensor(x)

        print("Params: ")
        print(params)

        results = {"data": data}

        torch_densities = torch_stable.pdf(data)
        # print(torch_densities)
        results["t-PDF"] = torch_densities
        scipy_densities = scipy_stable.pdf(data)
        # print(scipy_densities)
        results["s-PDF"] = scipy_densities
        torch_probs = torch_stable.cdf(data)
        results["t-CDF"] = torch_probs
        # print(torch_probs)
        scipy_probs = scipy_stable.cdf(data)
        results["s-CDF"] = scipy_probs
        # print(scipy_probs)
        print(tabulate(results, headers="keys"))
