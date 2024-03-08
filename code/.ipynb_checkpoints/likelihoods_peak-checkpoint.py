import numpyro
import numpyro.distributions as dist
import jax.numpy as jnp
from jax.scipy.special import erf
from jax import vmap
import random

from scipy.io import loadmat
import numpy as np

from custom_distributions import *
from gwBackground import v, dEdf
from constants import *
#from gwBackground import *

logit_std = 2.5
tmp_max = 100.
tmp_min = 2.
ln_m1_grid = jnp.linspace(jnp.log(tmp_min),jnp.log(tmp_max),100)
m1_grid = jnp.exp(ln_m1_grid)
d_ln_m1 = ln_m1_grid[1] - ln_m1_grid[0]

##################################################
#######            Pre-computing           #######
##################################################

N = 20000

m1s_drawn = np.random.uniform(tmp_min, tmp_max, size=N)

c_m2s = np.random.uniform(size=int(N))
m2s_drawn = tmp_min**(1.)+c_m2s*(m1s_drawn**(1.)-tmp_min**(1.))
    
zs_drawn = np.random.uniform(0,4,size=N)

# Computing dEdfs
freqs = stochasticDict['freqs']

dEdfs = np.array([dEdf(m1s_drawn[ii]+m2s_drawn[ii],freqs*(1+zs_drawn[ii]),eta=m2s_drawn[ii]/m1s_drawn[ii]/(1+m2s_drawn[ii]/m1s_drawn[ii])**2) for ii in range(N)])

p_m1_old = 1/(tmp_max-tmp_min)*np.ones(N)
p_z_old = 1/(4-0)*np.ones(N)
p_m2_old = 1/(m1s_drawn-tmp_min)


##################################################
#######            Likelihoods             #######
##################################################
all_zs = np.linspace(0,4,200)

def truncatedNormal(samples,mu,sigma,lowCutoff,highCutoff):

    """
    Jax-enabled truncated normal distribution
    
    Parameters
    ----------
    samples : `jax.numpy.array` or float
        Locations at which to evaluate probability density
    mu : float
        Mean of truncated normal
    sigma : float
        Standard deviation of truncated normal
    lowCutoff : float
        Lower truncation bound
    highCutoff : float
        Upper truncation bound

    Returns
    -------
    ps : jax.numpy.array or float
        Probability density at the locations of `samples`
    """

    a = (lowCutoff-mu)/jnp.sqrt(2*sigma**2)
    b = (highCutoff-mu)/jnp.sqrt(2*sigma**2)
    norm = jnp.sqrt(sigma**2*np.pi/2)*(-erf(a) + erf(b))
    ps = jnp.exp(-(samples-mu)**2/(2.*sigma**2))/norm
    return ps


@jax.jit
def sigmoid(low, delta, width, middle, zs):
    
    """
    Sigmoid function for varying parameters with a low value and high value which is delta
    away from that low value.
    
    Parameters
    ----------
    low : float
        The low value of the sigmoid function.
    delta : float
        The difference between the high value and the low value.
    width : float
        The width of the sigmoid function.
    middle : float
        The middle point of the sigmoid function.
    zs : array_like
        The input values at which to evaluate the sigmoid function.
        
    Returns
    -------
    array_like
        The values of the sigmoid function evaluated at the given input values.    
    """

    return delta / (1 + jnp.exp(-(1/width)*(zs - middle))) + low


@jax.jit
def sigmoid_initial_final_no_delta(low, high, width, middle, zs):
    
    """
    Sigmoid function for varying parameters with a low value and a high value.
    
    Parameters
    ----------
    low : float
        The low value of the sigmoid function.
    high : float
        The high value of the sigmoid function.
    width : float
        The width of the sigmoid function.
    middle : float
        The middle point of the sigmoid function.
    zs : array_like
        The input values at which to evaluate the sigmoid function.
        
    Returns
    -------
    array_like
        The values of the sigmoid function evaluated at the given input values.
    """

    return (high - low) / (1 + jnp.exp(-(1/width)*(zs - middle))) + low


@jax.jit
def merger_rate(alpha_z, beta_z, zp, zs):
    
    """
    Computes the merger rate as a function of redshift (zs) using the given parameters.
    
    Parameters
    ----------
    alpha_z : float
        low redshift exponent parameter determining the redshift dependence.
    beta_z : float
        high redshifts exponent parameter determining the redshift dependence.
    zp : float
        The peak redshift at which the merger rate shifts exponent.
    zs : array_like
        The redshift values at which to evaluate the merger rate.
        
    Returns
    -------
    array_like
        The merger rate evaluated at the given redshift values.
    """

    return (1+zs)**alpha_z/(1+((1+zs)/(1+zp))**(alpha_z+beta_z))


@jax.jit
def massModel_variation_all_m1(m1, alpha_ref, mu_m1, delta_mu, width_mu, middle_mu,
                               sig_m1, high_sig, width_sig, middle_sig,
                               log_f_peak, log_high_f_peak, width_f_peak, middle_f_peak,
                               mMax, mMin, dmMax, dmMin, zs):

    """
    Baseline primary mass model, described as a mixture between a power law
    and gaussian, with exponential tapering functions at high and low masses.

    Parameters
    ----------
    m1 : array_like or float
        Primary masses at which to evaluate probability densities.
    alpha_ref : float
        Power-law index.
    mu_m1 : float
        Location of possible Gaussian peak.
    delta_mu : float
        Change in location of the Gaussian peak with redshift.
    width_mu : float
        Width parameter of the sigmoid function for the Gaussian peak location.
    middle_mu : float
        Middle parameter of the sigmoid function for the Gaussian peak location.
    sig_m1 : float
        Standard deviation of possible Gaussian peak.
    high_sig : float
        High value of the standard deviation of the Gaussian peak with redshift.
    width_sig : float
        Width parameter of the sigmoid function for the Gaussian peak standard deviation.
    middle_sig : float
        Middle parameter of the sigmoid function for the Gaussian peak standard deviation.
    log_f_peak : float
        Logarithm of the approximate fraction of events contained within Gaussian peak.
    log_high_f_peak : float
        Logarithm of the high value of the fraction of events within Gaussian peak with redshift.
    width_f_peak : float
        Width parameter of the sigmoid function for the fraction of events within Gaussian peak.
    middle_f_peak : float
        Middle parameter of the sigmoid function for the fraction of events within Gaussian peak.
    mMax : float
        Location at which high-mass tapering begins.
    mMin : float
        Location at which low-mass tapering begins.
    dmMax : float
        Scale width of high-mass tapering function.
    dmMin : float
        Scale width of low-mass tapering function.
    zs : array_like
        Redshift values at which to evaluate the mass model.

    Returns
    -------
    array_like
        Unnormalized array of probability densities.
    """

    p_m1_pl = (1.+alpha_ref)*m1**(alpha_ref)/(tmp_max**(1.+alpha_ref) - tmp_min**(1.+alpha_ref))

    new_mu_m1 = sigmoid(mu_m1, delta_mu, width_mu, middle_mu, zs)
    new_sig_m1 = sigmoid_initial_final_no_delta(sig_m1, high_sig, width_sig, middle_sig, zs)

    p_m1_peak = jnp.exp(-(m1-new_mu_m1)**2/(2.*new_sig_m1**2))/jnp.sqrt(2.*np.pi*new_sig_m1**2)

    # Compute low- and high-mass filters
    low_filter = jnp.exp(-(m1-mMin)**2/(2.*dmMin**2))
    low_filter = jnp.where(m1<mMin,low_filter,1.)
    high_filter = jnp.exp(-(m1-mMax)**2/(2.*dmMax**2))
    high_filter = jnp.where(m1>mMax,high_filter,1.)

    new_f_peak = sigmoid_initial_final_no_delta(log_f_peak, log_high_f_peak, width_f_peak, middle_f_peak, zs)
    actual_f_peak = 10.**(new_f_peak)

    combined_p = jnp.array((actual_f_peak*p_m1_peak + (1. - actual_f_peak)*p_m1_pl)*low_filter*high_filter)
    return combined_p


def get_value_from_logit(logit_x,x_min,x_max):
    exp_logit = jnp.exp(logit_x)
    x = (exp_logit*x_max + x_min)/(1.+exp_logit)
    dlogit_dx = 1./(x-x_min) + 1./(x_max-x)
    return x,dlogit_dx


def cumsum(total,new_element):
    phi,w = new_element
    total = phi*total+w
    return total,total


def combined_pop_gwb_cbc_redshift_mass(sampleDict,injectionDict):
    """
    Implementation of a Gaussian effective spin distribution for inference within `numpyro`

    Parameters
    ----------
    sampleDict : dict
        Precomputed dictionary containing posterior samples for each event in our catalog
    injectionDict : dict
        Precomputed dictionary containing successfully recovered injections
    """
    
    # Sample our hyperparameters
    # alpha: Power-law index on primary mass distribution
    # mu_m1: Location of gaussian peak in primary mass distribution
    # sig_m1: Width of gaussian peak
    # f_peak: Fraction of events comprising gaussian peak
    # mMax: Location at which BBH mass distribution tapers off
    # mMin: Lower boundary at which BBH mass distribution tapers off
    # dmMax: Taper width above maximum mass
    # dmMin: Taper width below minimum mass
    # bq: Power-law index on the conditional secondary mass distribution p(m2|m1)
    # mu: Mean of the chi-effective distribution
    # logsig_chi: Log10 of the chi-effective distribution's standard deviation
    
    # I will be adding a moving mass distribution in z. 
    # First start with changing alpha in z, with a new parameter dalpha/dz and then sample it, and expand linearly in z for alpha. 
    # Take alpha_zero fixed and let dalpha/dz change with redshift. 
    # First without stochastic part, only CBC for the moment. 

    logR20 = numpyro.sample("logR20",dist.Uniform(-2,1))
    alpha_ref = numpyro.sample("alpha_ref",dist.Normal(-2,3))
    # dalpha_dz = numpyro.sample("dalpha_dz", dist.Normal(0, 5))

    mu_m1 = numpyro.sample("mu_m1",dist.Uniform(20,50))    
    delta_mu = numpyro.sample("delta_mu", dist.Normal(0, 5))
    log_width_mu = numpyro.sample("log_width_mu", dist.Uniform(-1, 1)) # cannot let this be negative, I think
    width_mu = numpyro.deterministic("width_mu", 10.**log_width_mu)
    middle_z_mu = numpyro.sample("middle_z_mu", dist.Uniform(0,4))
    
    mMin = numpyro.sample("mMin",dist.Uniform(5,15))

    bq = numpyro.sample("bq",dist.Normal(0,3))

    # kappa = numpyro.sample("kappa",dist.Uniform(-25,25))
    R20 = numpyro.deterministic("R20",10.**logR20)

    # alpha_z = numpyro.sample("alpha_z",dist.Uniform(-25,25))
    alpha_z = numpyro.sample("alpha_z",dist.Normal(0,4))
    beta_z = numpyro.sample("beta_z",dist.Uniform(0,10))

    zp = numpyro.sample("zp",dist.Uniform(0,4))
    
    # sig_m1 = numpyro.sample("sig_m1",TransformedUniform(1.5,15.))
    logit_sig_m1 = numpyro.sample("logit_sig_m1",dist.Normal(0,logit_std))
    sig_m1,jac_sig_m1 = get_value_from_logit(logit_sig_m1,1.5 ,15.)
    numpyro.deterministic("sig_m1",sig_m1)
    numpyro.factor("p_sig_m1",logit_sig_m1**2/(2.*logit_std**2)-jnp.log(jac_sig_m1))
    
    high_sig = numpyro.sample("high_sig", dist.Uniform(1.5, 15))
    log_width_sig = numpyro.sample("log_width_sig", dist.Uniform(-1, 1)) # cannot let this be negative, I think
    width_sig = numpyro.deterministic("width_sig", 10.**log_width_sig)
    middle_z_sig = numpyro.sample("middle_z_sig", dist.Uniform(0,4))
    # dsig_m1_dz = numpyro.sample("dsig_m1_dz", dist.Normal(0, 3))

    # log_f_peak = numpyro.sample("log_f_peak",TransformedUniform(-3.,0.))
    logit_log_f_peak = numpyro.sample("logit_log_f_peak",dist.Normal(0,logit_std))
    log_f_peak,jac_log_f_peak = get_value_from_logit(logit_log_f_peak,-5. ,0.) # -5 was -3
    numpyro.deterministic("log_f_peak",log_f_peak)
    numpyro.factor("p_log_f_peak",logit_log_f_peak**2/(2.*logit_std**2)-jnp.log(jac_log_f_peak))
    
    log_high_f_peak = numpyro.sample("log_high_f_peak", dist.Uniform(-5, 0))
    log_width_f_peak = numpyro.sample("log_width_f_peak", dist.Uniform(-1, 1)) # cannot let this be negative, I think
    width_f_peak = numpyro.deterministic("width_f_peak", 10.**log_width_f_peak)
    middle_z_f_peak = numpyro.sample("middle_z_f_peak", dist.Uniform(0,4))
    # log_dfpeak_dz = numpyro.sample("log_df_peak_dz", dist.Normal(0,0.5)) # 1 or .1

    # mMax = numpyro.sample("mMax",TransformedUniform(50.,100.))
    logit_mMax = numpyro.sample("logit_mMax",dist.Normal(0,logit_std))
    mMax,jac_mMax = get_value_from_logit(logit_mMax,50. ,100.)
    numpyro.deterministic("mMax",mMax)
    numpyro.factor("p_mMax",logit_mMax**2/(2.*logit_std**2)-jnp.log(jac_mMax))

    # log_dmMin = numpyro.sample("log_dmMin",TransformedUniform(-1.,0.5))
    logit_log_dmMin = numpyro.sample("logit_log_dmMin",dist.Normal(0,logit_std))
    log_dmMin,jac_log_dmMin = get_value_from_logit(logit_log_dmMin, -1. ,0.5)
    numpyro.deterministic("log_dmMin",log_dmMin)
    numpyro.factor("p_log_dmMin",logit_log_dmMin**2/(2.*logit_std**2)-jnp.log(jac_log_dmMin))

    # log_dmMax = numpyro.sample("log_dmMax",TransformedUniform(0.5,1.5))
    logit_log_dmMax = numpyro.sample("logit_log_dmMax",dist.Normal(0,logit_std))
    log_dmMax,jac_log_dmMax = get_value_from_logit(logit_log_dmMax,0.5 ,1.5)
    numpyro.deterministic("log_dmMax",log_dmMax)
    numpyro.factor("p_log_dmMax",logit_log_dmMax**2/(2.*logit_std**2)-jnp.log(jac_log_dmMax))

    # mu_chi = numpyro.sample("mu_chi",TransformedUniform(0.,1.))
    logit_mu_chi= numpyro.sample("logit_mu_chi",dist.Normal(0,logit_std))
    mu_chi,jac_mu_chi = get_value_from_logit(logit_mu_chi,0. ,1.)
    numpyro.deterministic("mu_chi",mu_chi)
    numpyro.factor("p_mu_chi",logit_mu_chi**2/(2.*logit_std**2)-jnp.log(jac_mu_chi))

    # logsig_chi = numpyro.sample("logsig_chi",TransformedUniform(-1.,0.))
    logit_logsig_chi = numpyro.sample("logit_logsig_chi",dist.Normal(0,logit_std))
    logsig_chi,jac_logsig_chi = get_value_from_logit(logit_logsig_chi,-1. ,0.)
    numpyro.deterministic("logsig_chi",logsig_chi)
    numpyro.factor("p_logsig_chi",logit_logsig_chi**2/(2.*logit_std**2)-jnp.log(jac_logsig_chi))

    # sig_cost = numpyro.sample("sig_cost",TransformedUniform(0.3,2.))
    logit_sig_cost = numpyro.sample("logit_sig_cost",dist.Normal(0,logit_std))
    sig_cost,jac_sig_cost = get_value_from_logit(logit_sig_cost,0.3 ,2.)
    numpyro.deterministic("sig_cost",sig_cost)
    numpyro.factor("p_sig_cost",logit_sig_cost**2/(2.*logit_std**2)-jnp.log(jac_sig_cost))

    # Fixed params
    mu_cost = 1.

    # Normalization
    alpha_for_norm = alpha_ref
    p_m1_norm = massModel_variation_all_m1(jnp.array([20]), alpha_ref, mu_m1, delta_mu, width_mu, middle_z_mu, 
                                           sig_m1, high_sig, width_sig, middle_z_sig,
                                           log_f_peak, log_high_f_peak, width_f_peak, middle_z_f_peak,
                                           mMax, mMin, 10.**log_dmMax,10.**log_dmMin, jnp.array([0.2]))
    p_z_norm = merger_rate(alpha_z, beta_z, zp, 0.2)

    # Read out found injections
    # Note that `pop_reweight` is the inverse of the draw weights for each event
    a1_det = injectionDict['a1']
    a2_det = injectionDict['a2']
    cost1_det = injectionDict['cost1']
    cost2_det = injectionDict['cost2']
    m1_det = injectionDict['m1']
    m2_det = injectionDict['m2']
    z_det = injectionDict['z']
    dVdz_det = injectionDict['dVdz']
    p_draw = injectionDict['p_draw_m1m2z']*injectionDict['p_draw_a1a2cost1cost2']

    # Compute proposed population weights
    p_m1_det =  massModel_variation_all_m1(m1_det,  alpha_ref, mu_m1, delta_mu, width_mu, middle_z_mu, 
                                           sig_m1, high_sig, width_sig, middle_z_sig,
                                           log_f_peak, log_high_f_peak, width_f_peak, middle_z_f_peak,
                                           mMax, mMin, 10.**log_dmMax,10.**log_dmMin, z_det)/p_m1_norm
    p_m2_det = (1.+bq)*m2_det**bq/(m1_det**(1.+bq)-tmp_min**(1.+bq))
    p_a1_det = truncatedNormal(a1_det,mu_chi,10.**logsig_chi,0,1)
    p_a2_det = truncatedNormal(a2_det,mu_chi,10.**logsig_chi,0,1)
    p_cost1_det = truncatedNormal(cost1_det,mu_cost,sig_cost,-1,1)
    p_cost2_det = truncatedNormal(cost2_det,mu_cost,sig_cost,-1,1)

    rate_det = merger_rate(alpha_z, beta_z, zp, z_det)
    p_z_det = dVdz_det/(1.+z_det)*rate_det/p_z_norm 
    R_pop_det = R20*p_m1_det*p_m2_det*p_z_det*p_a1_det*p_a2_det*p_cost1_det*p_cost2_det

    # Form ratio of proposed weights over draw weights
    inj_weights = R_pop_det/(p_draw/2.)
    
    # As a fit diagnostic, compute effective number of injections
    nEff_inj = jnp.sum(inj_weights)**2/jnp.sum(inj_weights**2)
    nObs = 1.0*len(sampleDict)
    numpyro.deterministic("nEff_inj_per_event",nEff_inj/nObs)

    # Compute net detection efficiency and add to log-likelihood
    Nexp = jnp.sum(inj_weights)/injectionDict['nTrials']
    numpyro.factor("rate",-Nexp)

    # This function defines the per-event log-likelihood
    # m1_sample: Primary mass posterior samples
    # m2_sample: Secondary mass posterior samples
    # z_sample: Redshift posterior samples
    # dVdz_sample: Differential comoving volume at each sample location
    # Xeff_sample: Effective spin posterior samples
    # priors: PE priors on each sample
    def logp_cbc(m1_sample,m2_sample,z_sample,dVdz_sample,a1_sample,a2_sample,cost1_sample,cost2_sample,priors):

        # Compute proposed population weights
        
        p_m1 =  massModel_variation_all_m1(m1_sample,  alpha_ref, mu_m1, delta_mu, width_mu, middle_z_mu, 
                                           sig_m1, high_sig, width_sig, middle_z_sig,
                                           log_f_peak, log_high_f_peak, width_f_peak, middle_z_f_peak,
                                           mMax, mMin, 10.**log_dmMax,10.**log_dmMin, z_sample)/p_m1_norm
        p_m2 = (1.+bq)*m2_sample**bq/(m1_sample**(1.+bq)-tmp_min**(1.+bq))
        p_a1 = truncatedNormal(a1_sample,mu_chi,10.**logsig_chi,0,1)
        p_a2 = truncatedNormal(a2_sample,mu_chi,10.**logsig_chi,0,1)
        p_cost1 = truncatedNormal(cost1_sample,mu_cost,sig_cost,-1,1)
        p_cost2 = truncatedNormal(cost2_sample,mu_cost,sig_cost,-1,1)

        rate = merger_rate(alpha_z, beta_z, zp, z_sample)
        p_z = dVdz_sample/(1.+z_sample)*rate/p_z_norm
        R_pop = R20*p_m1*p_m2*p_z*p_a1*p_a2*p_cost1*p_cost2
        mc_weights = R_pop/priors
        
        # Compute effective number of samples and return log-likelihood
        n_eff = jnp.sum(mc_weights)**2/jnp.sum(mc_weights**2)     
        return jnp.log(jnp.mean(mc_weights)),n_eff
    
    # Map the log-likelihood function over each event in our catalog
    log_ps,n_effs = vmap(logp_cbc)(
                       jnp.array([sampleDict[k]['m1'] for k in sampleDict]),
                       jnp.array([sampleDict[k]['m2'] for k in sampleDict]),
                       jnp.array([sampleDict[k]['z'] for k in sampleDict]), 
                       jnp.array([sampleDict[k]['dVc_dz'] for k in sampleDict]),
                       jnp.array([sampleDict[k]['a1'] for k in sampleDict]),
                       jnp.array([sampleDict[k]['a2'] for k in sampleDict]),
                       jnp.array([sampleDict[k]['cost1'] for k in sampleDict]),
                       jnp.array([sampleDict[k]['cost2'] for k in sampleDict]),
                       jnp.array([sampleDict[k]['z_prior'] for k in sampleDict]))
    # As a diagnostic, save minimum number of effective samples across all events
    numpyro.deterministic('min_log_neff',jnp.min(jnp.log10(n_effs)))

    # Tally log-likelihoods across our catalog
    numpyro.factor("logp_cbc",jnp.sum(log_ps))
