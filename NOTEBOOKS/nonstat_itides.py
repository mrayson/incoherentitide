"""

"""

from speccy import bochner, periodogram
from scipy.optimize import minimize
from gptide import cov
import numpy as np

from scipy.special import kv as K_nu
from scipy.special import gamma


def extract_point(ds, xpt, ypt):
    dist = np.abs( (ds.xv.values-xpt) + 1j*(ds.yv.values-ypt))
    idx = np.argwhere(dist==dist.min())[0,0]

    return ds.isel(Nc=idx).drop_vars(['xp','yp','cells'])

# Covariance kernels / ACFs


def logit(p, scale=1.):
    cff = 1/scale
    return np.log(p*cff/(1-p*cff))

def invlogit(x, scale=1.):
    return scale*np.exp(x)/(1+np.exp(x))

def calc_dist(x, xpr, eps=1e-14):
    dx2 = np.power(x-xpr, 2.)
    dx2[dx2<eps] = eps
    return np.sqrt(dx2)

def gamma_exp(x, xpr, gam, l):
    """γ-exponential covariance function"""
    dx = calc_dist(x, xpr)
    return np.exp(-np.power(dx/l, gam))
    
def matern_general(x, xpr, nu, l):
    """General Matern base function"""
    dx = calc_dist(x, xpr)
    
    cff1 = np.sqrt(2*nu)*dx/l
    #K = np.power(eta, 2.) * np.power(2., 1-nu) / gamma(nu)
    K = np.power(2., 1-nu) / gamma(nu)
    K *= np.power(cff1, nu)
    K *= K_nu(nu,cff1)
    
    #K[np.isnan(K)] = np.power(eta, 2.)
    
    return K

def oscillate_1d_matern(x, xpr, params):
    """
    1D Oscillatory kernel
    """
    eta, d, l, nu = params

    return eta**2 * matern_general(x, xpr, nu, d) * cov.cosine(x, xpr, l)

def oscillate_1d_gammaexp(x, xpr, params):
    """
    1D Oscillatory kernel
    """
    eta, d, l, gam = params

    return eta**2 * gamma_exp(x, xpr, gam, d) * cov.cosine(x, xpr, l)


def matern12(x,xpr,l):
    """Matern 1/2 base function"""
    fac2 = np.sqrt((x-xpr)*(x-xpr))
    return np.exp(-fac2/l)


def oscillate_1d(x, xpr, params, itfunc=matern12):
    """
    1D Oscillatory kernel
    """
    eta, d, l = params

    return eta**2 * itfunc(x, xpr, d) * cov.cosine(x, xpr, l)

def oscillate_M2S2(x, xpr, params,
                     lt =[0.517525050851839, 0.5],
                       itfunc=matern12):

    eta1, d1, eta2, d2 = params

    C = oscillate_1d(x, xpr, (eta1, d1, lt[0]), itfunc=itfunc)
    C += oscillate_1d(x, xpr, (eta2, d2, lt[1]), itfunc=itfunc)

    return C

def oscillate_M2S2K1O1(x, xpr, params,
                     lt =[0.517525050851839, 0.5, 0.9972695689985752, 1.0758059026974014],
                       itfunc=matern12):

    eta1, d1, eta2, d2, eta3, d3, eta4, d4 = params

    C = oscillate_1d(x, xpr, (eta1, d1, lt[0]), itfunc=itfunc)
    C += oscillate_1d(x, xpr, (eta2, d2, lt[1]), itfunc=itfunc)
    C += oscillate_1d(x, xpr, (eta3, d3, lt[2]), itfunc=itfunc)
    C += oscillate_1d(x, xpr, (eta4, d4, lt[3]), itfunc=itfunc)

    return C

def oscillate_M2S2K1O1_logeta(x, xpr, params,
                     lt =[0.517525050851839, 0.5, 0.9972695689985752, 1.0758059026974014],
                       itfunc=matern12):

    log_eta1, log_d1, log_eta2, log_d2, log_eta3, log_d3, log_eta4, log_d4 = params

    eta1 = np.exp(log_eta1)
    eta2 = np.exp(log_eta2)
    eta3 = np.exp(log_eta3)
    eta4 = np.exp(log_eta4)

    d1 = np.exp(log_d1)
    d2 = np.exp(log_d2)
    d3 = np.exp(log_d3)
    d4 = np.exp(log_d4)

    C = oscillate_1d(x, xpr, (eta1, d1, lt[0]), itfunc=itfunc)
    C += oscillate_1d(x, xpr, (eta2, d2, lt[1]), itfunc=itfunc)
    C += oscillate_1d(x, xpr, (eta3, d3, lt[2]), itfunc=itfunc)
    C += oscillate_1d(x, xpr, (eta4, d4, lt[3]), itfunc=itfunc)

    return C

def oscillate_S2K1_logeta(x, xpr, params, 
                     lt =[ 0.5, 0.9972695689985752],
                       itfunc=matern12):

    log_eta1, log_d1, log_eta2, log_d2 = params
    
    eta1 = np.exp(log_eta1)
    eta2 = np.exp(log_eta2)
    
    
    d1 = np.exp(log_d1)
    d2 = np.exp(log_d2)

    C = oscillate_1d(x, xpr, (eta1, d1, lt[0]), itfunc=itfunc)
    C += oscillate_1d(x, xpr, (eta2, d2, lt[1]), itfunc=itfunc)

    return C

def oscillate_D2D1_logeta(x, xpr, params, 
                     lt =[ (0.5+0.517525050851839)/2, (0.9972695689985752+1.0758059026974014)/2],
                       itfunc=cov.matern32):

    log_eta1, log_d1, log_eta2, log_d2 = params
    
    eta1 = np.exp(log_eta1)
    eta2 = np.exp(log_eta2)
    
    
    d1 = np.exp(log_d1)
    d2 = np.exp(log_d2)

    C = oscillate_1d(x, xpr, (eta1, d1, lt[0]), itfunc=itfunc)
    C += oscillate_1d(x, xpr, (eta2, d2, lt[1]), itfunc=itfunc)

    return C

def oscillate_D2D1_logeta_constant(x, xpr, params, 
                     lt =[ (0.5+0.517525050851839)/2, (0.9972695689985752+1.0758059026974014)/2],
                       itfunc=cov.expquad):

    log_eta1, log_d1, log_eta2, log_d2, log_constant = params
    
    eta1 = np.exp(log_eta1)
    eta2 = np.exp(log_eta2)
    d1 = np.exp(log_d1)
    d2 = np.exp(log_d2)
    constant = np.exp(log_constant)

    C = oscillate_1d(x, xpr, (eta1, d1, lt[0]), itfunc=itfunc)
    C += oscillate_1d(x, xpr, (eta2, d2, lt[1]), itfunc=itfunc)
    C[0] = C[0] + constant

    return C

def oscillate_D2D1_gmatern(x, xpr, params, 
                     lt =[ (0.5+0.517525050851839)/2, (0.9972695689985752+1.0758059026974014)/2]):

    log_eta1, log_d1, log_eta2, log_d2, log_nu, log_constant = params
    
    eta1 = np.exp(log_eta1)
    eta2 = np.exp(log_eta2)
    d1 = np.exp(log_d1)
    d2 = np.exp(log_d2)
    nu = np.exp(log_nu)
    constant = np.exp(log_constant)

    C = oscillate_1d_matern(x, xpr, (eta1, d1, lt[0], nu))
    C += oscillate_1d_matern(x, xpr, (eta2, d2, lt[1], nu))
    
    C[0] = C[0] + constant

    return C

def oscillate_D2D1_legit(x, xpr, params, 
                     lt =[ (0.5+0.517525050851839)/2, (0.9972695689985752+1.0758059026974014)/2]):

    log_eta1, log_d1, log_eta2, log_d2, logit_gamma, log_constant = params
    
    eta1 = np.exp(log_eta1)
    eta2 = np.exp(log_eta2)
    d1 = np.exp(log_d1)
    d2 = np.exp(log_d2)
    
    gam = invlogit(logit_gamma, scale=2.) # 0 < gamma < 2
    constant = np.exp(log_constant)

    C = oscillate_1d_gammaexp(x, xpr, (eta1, d1, lt[0], gam))
    C += oscillate_1d_gammaexp(x, xpr, (eta2, d2, lt[1], gam))
    
    C[0] = C[0] + constant

    return C

def oscillate_D2D1_legit2(x, xpr, params, 
                     lt =[ (0.5+0.517525050851839)/2, (0.9972695689985752+1.0758059026974014)/2]):

    log_eta1, log_d1, log_eta2, log_d2, logit_gamma1, logit_gamma2, log_constant = params
    
    eta1 = np.exp(log_eta1)
    eta2 = np.exp(log_eta2)
    d1 = np.exp(log_d1)
    d2 = np.exp(log_d2)
    
    gam1 = invlogit(logit_gamma1, scale=2.) # 0 < gamma < 2
    gam2 = invlogit(logit_gamma2, scale=2.) # 0 < gamma < 2
    constant = np.exp(log_constant)

    C = oscillate_1d_gammaexp(x, xpr, (eta1, d1, lt[0], gam1))
    C += oscillate_1d_gammaexp(x, xpr, (eta2, d2, lt[1], gam2))
    
    C[0] = C[0] + constant

    return C

def oscillate_D2D1_gammaexp(x, xpr, params, 
                     lt =[ (0.5+0.517525050851839)/2, (0.9972695689985752+1.0758059026974014)/2]):

    eta1, d1, eta2, d2, gam1, gam2, constant = params
    

    C = oscillate_1d_gammaexp(x, xpr, (eta1, d1, lt[0], gam1))
    C += oscillate_1d_gammaexp(x, xpr, (eta2, d2, lt[1], gam2))
    
    C[0] = C[0] + constant

    return C

def itmodel_matern_log(x, xpr, params, l=0.5):
    
    log_eta, log_d,  logit_nu = params
   
    eta = np.exp(log_eta)
    d = np.exp(log_d)
    nu = invlogit(logit_nu, scale=3.5)

    return oscillate_1d_matern(x, xpr, (eta, d, l, nu) )

def itmodel_gamma_log(x, xpr, params, l=0.5):
    
    log_eta, log_d,  logit_gam = params
   
    eta = np.exp(log_eta)
    d = np.exp(log_d)
    gam = invlogit(logit_gam, scale=2.)

    return oscillate_1d_gammaexp(x, xpr, (eta, d, l, gam) )

def itmodel_matern(x, xpr, params, l=0.5):
    
    eta, d,  logit_nu = params
    
    #nu = nsit.invlogit(logit_nu, scale=3.5)

    return oscillate_1d_matern(x, xpr, (eta, d, l, nu) )

def itmodel_gamma(x, xpr, params, l=0.5):
    
    eta, d,  gam = params
    #gam = nsit.invlogit(logit_gam, scale=2.)

    return oscillate_1d_gammaexp(x, xpr, (eta, d, l, gam) )

def itmodel_gamma_fixed(x, xpr, params, l=0.5, gam=2.):
    
    eta, d = params
    
    return oscillate_1d_gammaexp(x, xpr, (eta, d, l, gam) )

def itmodel_expquad(x, xpr, params, l=0.5):

    #     log_eta, log_d = params

    #     eta = np.exp(log_eta)
    #     d = np.exp(log_d)
    eta, d = params
    
    return oscillate_1d(x, xpr, (eta, d, l), itfunc=cov.expquad )

def itmodel_lorentzian(x, xpr, params, l=0.5):
    
    eta, d = params
    
    return oscillate_1d(x, xpr, (eta, d, l), itfunc=matern12 )

def itmodel_matern(x, xpr, params, l=0.5):
    
    eta, d,  nu = params
   

    return oscillate_1d_matern(x, xpr, (eta, d, l, nu) )

# Some functions that could also go into speccy...
def dwhittle_fast(x, y, ff, I, acffunc, params, delta = 1, h = None, fmin=0, fmax=np.inf):

    ff_boch, S_boch = bochner(acffunc(x, x[0], params), delta = delta, bias = True)

    # Subset frequencies
    idx_boch = (ff_boch > fmin) & (ff_boch<fmax)
    idx = (ff > fmin) & (ff<fmax)

    # HACK: quick fix cause bochner isn't two sided yet
    return - 2 * np.sum(np.log(S_boch[idx_boch]) + I[idx]/S_boch[idx_boch])

def myminfunc(params, priors, X, y, f, I, covfunc,  dt, fmin, fmax):
    ## Add on the priors
    
    sum_prior = 0.
    if priors is not None:
        log_prior = np.array([P.logpdf(val) for P, val in zip(priors, params)])
        if np.any(np.isinf(log_prior)):
            return 1e25
        sum_prior = np.sum(log_prior)
        
    return -dwhittle_fast(X, y, f, I, covfunc, params, delta=dt, fmin=fmin, fmax=fmax) - sum_prior

def estimate_spectral_params_whittle(y, X, covfunc, covparams_ic, fmin, fmax,
            priors=None,
            method='nelder-mead',
            options={'maxiter':5000},
            callback=None,
            bounds=None):

    # Compute the periodogram outside of
    dt = X[1]-X[0]

    f, I = periodogram(y, delta=dt)
    
    soln=minimize(myminfunc,
                  covparams_ic,
                  args=(priors, X, y, f, I, covfunc, dt, fmin, fmax),
                  method=method,
                  bounds=bounds,
                  options=options,
                  callback=callback,
                 )

    return soln['x']

def estimate_spectral_params_whittle_ufunc(y, priors=None, X=None, covfunc=None, covparams_ic=None, fmin=None, fmax=None, 
                                          method='nelder-mead',
                                          options={'maxiter':5000},
                                          callback=None,
                                          bounds=None):
    """
    Function that xarray.apply_func can handle
    """
    return estimate_spectral_params_whittle(y, X, covfunc, covparams_ic, fmin, fmax, 
                                            priors=priors,
                                            method=method, options=options, callback=callback, bounds=bounds)

