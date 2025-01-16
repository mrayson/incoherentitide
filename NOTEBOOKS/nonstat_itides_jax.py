from functools import partial
from jax import numpy as np
import jax 
import optax

###########################
# Speccy jax version
###########################
# Speccy jax stuff


def is_even(x):
    return x % 2 == 0

def n_freq(n):
    return int(np.floor(n/2))

def taus(n, delta):
    return delta * np.arange(n)

def fftshift(x):
    return np.fft.fftshift(x)

def fftfreq(n, delta):
    return fftshift(np.fft.fftfreq(n, delta))

def periodogram(ts, delta = 1, h = None):
    
    n = ts.size

    if h is not None:
        norm = np.sum(h**2)
        scale = np.sqrt(n/norm)
        ts = scale * h * ts

    dft = np.fft.fft(ts)/np.sqrt(n/delta)
    
    I = np.real(dft * np.conj(dft))
    ff = np.fft.fftfreq(n, delta)

    return fftshift(ff), fftshift(I)

def whittle(ts, specfunc, params, delta = 1, h = None):
    
    ff, I = periodogram(ts, delta, h)
    S = specfunc(ff, params)

    ll = - (np.log(S) + I/S)
    idx = (ff != 0) * (ff != -0.5/delta)
    
    return np.sum(ll[idx])

def dwhittle(ts, acffunc, params, delta = 1, h = None):
    
    tt = delta * np.arange(ts.size)
    ff, I = periodogram(ts, delta, h)
    ff_boch, S_boch = bochner(acffunc(tt, params), delta = delta, bias = True)
    # HACK: quick fix cause bochner isn't two sided yet
    return - 2 * np.sum(np.log(S_boch[ff_boch > 0]) + I[ff > 0]/S_boch[ff_boch > 0])

def bochner(acf, delta = 1, bias = True, h = None):

    n = np.size(acf)

    if h is not None:
        
        norm = np.sum(h**2)
        h_conv = (np.convolve(h, h, mode = 'full')/norm)[(n-1):]
        acf = h_conv * acf

    elif bias:

        acf = (1 - np.arange(n)/n) * acf

    ff = fftfreq(n, delta)

    if is_even(n):
        acf = np.concatenate([np.array([acf[0]/2]), acf[1:(n-1)], np.array([acf[-1]/2])])
    else:
        acf = np.concatenate([np.array([acf[0]/2]), acf[1:n]])
    
    psd = 2 * delta * np.real(np.fft.fft(acf))

    return ff, fftshift(psd)

#################################
# Covariance models - Jax version
##################################

def logit(p, scale=1.):
    cff = 1/scale
    return np.log(p*cff/(1-p*cff))

def invlogit(x, scale=1.):
    return scale*np.exp(x)/(1+np.exp(x))

# Covariance kernels / ACFs
def calc_dist(x, xpr, eps=1e-14):
    dx2 = np.power(x-xpr, 2.)
    #dx2[dx2<eps] = eps
    #dx2 = dx2.at[dx2<eps].set(eps)
    dx2 = np.where(dx2 < eps, eps, dx2)
    return np.sqrt(dx2)

def cosine(x, xpr, l):
    """Cosine base function"""
    return np.cos(2*np.pi*np.abs(x-xpr)/l)


def gamma_exp(x, xpr, gam, l):
    """γ-exponential covariance function"""
    dx = calc_dist(x, xpr)
    return np.exp(-np.power(dx/l, gam))

def gamma_exp_1d(x, xpr, params):
    """
    1D Oscillatory kernel
    """
    eta, d, gam = params

    return eta**2 * gamma_exp(x, xpr, gam, d) 
    
def oscillate_1d_matern(x, xpr, params):
    """
    1D Oscillatory kernel
    """
    eta, d, l, nu = params

    return eta**2 * matern_general(x, xpr, nu, d) * cosine(x, xpr, l)

def oscillate_1d_gammaexp(x, xpr, params):
    """
    1D Oscillatory kernel
    """
    eta, d, l, gam = params

    return eta**2 * gamma_exp(x, xpr, gam, d) * cosine(x, xpr, l)

def itmodel_gamma(x, xpr, params, l=0.5):
    
    eta, d,  gam = params
    #gam = nsit.invlogit(logit_gam, scale=2.)

    return oscillate_1d_gammaexp(x, xpr, (eta, d, l, gam) )

def oscillate_D2D1_gammaexp(x, xpr, params, 
                     lt =[ (0.5+0.517525050851839)/2, (0.9972695689985752+1.0758059026974014)/2]):

    eta1, d1, eta2, d2, gam1, gam2 = params
    

    C = oscillate_1d_gammaexp(x, xpr, (eta1, d1, lt[0], gam1))
    C += oscillate_1d_gammaexp(x, xpr, (eta2, d2, lt[1], gam2))

    return C

def oscillate_D2D1_gammaexp_fixed(x, xpr, params, 
                     lt =[ (0.5+0.517525050851839)/2, (0.9972695689985752+1.0758059026974014)/2],
                                 gam1=1.5, gam2=1.5):

    eta1, d1, eta2, d2, constant = params
    

    C = oscillate_1d_gammaexp(x, xpr, (eta1, d1, lt[0], gam1))
    C += oscillate_1d_gammaexp(x, xpr, (eta2, d2, lt[1], gam2))
    
    #C[0] = C[0] + constant
    C = C.at[0].add(constant)

    return C

def oscillate_fD2D1_gammaexp(x, xpr, params, 
                     f_cor=2,
                     lt =[ (0.5+0.517525050851839)/2, (0.9972695689985752+1.0758059026974014)/2],
                                 gam=1.5):

    eta1, d1, eta2, d2, eta3, d3 = params
    
    C = oscillate_1d_gammaexp(x, xpr, (eta1, d1, f_cor, gam))
    C += oscillate_1d_gammaexp(x, xpr, (eta2, d2, lt[0], gam))
    C += oscillate_1d_gammaexp(x, xpr, (eta3, d3, lt[1], gam))

    return C


def itide_fD2D1_meso_gammaexp(x, xpr, params, **kwargs):
    eta_m, l_m, gam_m, eta1, d1,  eta2, d2, eta3, d3 = params

    C = eta_m**2 * gamma_exp(x, xpr, gam_m, l_m)
    C += oscillate_fD2D1_gammaexp(x, xpr, (eta1, d1, eta2, d2, eta3, d3), **kwargs)
    return C

def itide_D2_meso_gammaexp(x, xpr, params, 
                     lt = 0.5):

    eta_m, l_m, gam_m, eta2, d2, gam1 = params
    
    C = eta_m**2 * gamma_exp(x, xpr, gam_m, l_m)
    C += oscillate_1d_gammaexp(x, xpr, (eta2, d2, lt, gam1))

    return C

def oscillate_D2D1_gammaexp(x, xpr, params, 
                     f_cor=2,
                     lt =[ (0.5+0.517525050851839)/2, (0.9972695689985752+1.0758059026974014)/2],
                                 gam=1.5):

    eta2, d2, eta3, d3 = params
    
    C = oscillate_1d_gammaexp(x, xpr, (eta2, d2, lt[0], gam))
    C += oscillate_1d_gammaexp(x, xpr, (eta3, d3, lt[1], gam))

    return C


def itide_meso_nof_gammaexp(x, xpr, params, **kwargs):
    eta_m, l_m, gam_m, eta2, d2, eta3, d3 = params

    C = eta_m**2 * gamma_exp(x, xpr, gam_m, l_m)
    C += oscillate_D2D1_gammaexp(x, xpr, (eta2, d2, eta3, d3), **kwargs)
    return C

def itide_D2_meso_gammaexp_fixed(x, xpr, params, 
                     lt = 0.5, gam1 = 1.0):

    eta_m, l_m, gam_m, eta2, d2 = params
    
    C = eta_m**2 * gamma_exp(x, xpr, gam_m, l_m)
    C += oscillate_1d_gammaexp(x, xpr, (eta2, d2, lt, gam1))

    return C

def itide_M2S2_meso_gammaexp(x, xpr, params, 
                     lt = [0.5175, 0.5] ):

    eta_m, l_m, gam_m, etaM2, etaS2, tauM2, tauS2, gammaM2, gammaS2 = params
    
    C = eta_m**2 * gamma_exp(x, xpr, gam_m, l_m)
    C += oscillate_1d_gammaexp(x, xpr, (etaM2, tauM2, lt[0], gammaM2))
    C += oscillate_1d_gammaexp(x, xpr, (etaS2, tauS2, lt[1], gammaS2))
    return C

def itide_M2S2_meso_gammaexp_fixed(x, xpr, params, 
                     lt = [0.5175, 0.5], gamma=1 ):

    eta_m, l_m, gam_m, etaM2, etaS2, tauM2, tauS2 = params
    
    C = eta_m**2 * gamma_exp(x, xpr, gam_m, l_m)
    C += oscillate_1d_gammaexp(x, xpr, (etaM2, tauM2, lt[0], gamma))
    C += oscillate_1d_gammaexp(x, xpr, (etaS2, tauS2, lt[1], gamma))
    return C
#################################################
# Jax parameter estimation/optimisation routines
#################################################
# Transformation of optimiser parameters
class LogTransformer:
    def __init__(self,params):
        self.params = params

    def __call__(self):
        return np.log(self.params)
    
    def out(self, tparams):
        return np.exp(tparams)

class LinearTransformer:
    def __init__(self,params, scale=1., offset=0.):
        self.scale = scale
        self.offset = offset
        self.params = params

    def __call__(self):
        return (self.params-offset)/scale
    
    def out(self, tparams):
        return self.params*scale + offset

class CustomTransformer:
    def __init__(self,params):
        self.params = params

    def __call__(self):
        params_t = np.log(self.params)
        params_t = params_t.at[2].set(invlogit(self.params[2],scale=2))
        params_t = params_t.at[5].set(invlogit(self.params[5],scale=2))
        return params_t
        
    def out(self, tparams):
        params = np.exp(tparams)
        #params[2] = nsjax.logit(tparams[2],scale=2)
        #params[5] = nsjax.logit(tparams[5],scale=2)
        params = params.at[2].set(logit(tparams[2],scale=2))
        params = params.at[5].set(logit(tparams[5],scale=2))
        return params
        
###
# Loss functions
def dwhittle_fast(x, y, ff, I, acffunc, params, fidx, delta = 1, h = None, fmin=0, fmax=np.inf, acf_kwargs={}):
    ff_boch, S_boch = bochner(acffunc(x, x[0], params, **acf_kwargs), delta = delta, bias = True)
    # Subset frequencies
    #idx = (ff > fmin) & (ff<fmax)
    whit = np.log(S_boch) + I/S_boch
    return -2* np.where(fidx, whit, 0).sum()

@jax.value_and_grad
@partial(jax.jit, static_argnums=(6,7,8))
def loss(logparams,  X, y, f, I, fidx, covfunc,  dt,  Transformer, acf_kwargs):
    params = Transformer.out(logparams)
    return -dwhittle_fast(X, y, f, I, covfunc, params, fidx, delta=dt, acf_kwargs=acf_kwargs) 

#####
# Main optimisation routine
#####
def estimate_jax(y, X, covfunc, covparams_ic, fmin, fmax,
                fidx=None,
                cov_kwargs={},
                window=None,
                verbose=True,
                maxiter=500,
                ftol=1e-2,
                opt= optax.sgd(learning_rate=3e-4),
                transformer=LogTransformer,
                f=None,
                I=None):

    dt = X[1]-X[0]
    if f is None or I is None:
        f, I = periodogram(y, delta=dt, h=window)
        
    if fidx is None:
        fidx = (f > fmin) & (f<fmax)
    
    # def dwhittle_jax(params, acffunc):
    #     ff_boch, S_boch = bochner(acffunc(x, x[0], params))
    #     whit = np.log(S_boch) + I/S_boch
    #     return -2* np.where(idx, whit, 0).sum()
    
    # @jax.value_and_grad
    # @partial(jax.jit, static_argnums=(1))
    # def loss2(logparams, covfunc):
    #     params = np.exp(logparams)
    #     return -dwhittle_jax(params, covfunc) 

    T = transformer(np.array(covparams_ic))
    logparams = T()
    
    opt_state = opt.init(logparams)
    loss_val = np.inf
    for i in range(maxiter):
        loss_val_new, grads = loss(logparams, np.array(X), np.array(y), f, I, fidx, 
                                   covfunc,  dt, T, cov_kwargs)
        #loss_val_new, grads = loss2(logparams, covfunc)
        updates, opt_state = opt.update(grads, opt_state)
        logparams = optax.apply_updates(logparams, updates)
        
        if i % 25 == 0:
            if verbose:
                print(f'step {i}, loss: {loss_val_new}')
                print(np.exp(logparams))
        if np.abs(loss_val_new-loss_val) < ftol:
            if verbose:
                print(f'step {i}, loss: {loss_val}')
            break
    
        loss_val = 1*loss_val_new
        
    return T.out(logparams), loss_val
    #return np.concatenate([T.out(logparams), np.array([loss_val])])
