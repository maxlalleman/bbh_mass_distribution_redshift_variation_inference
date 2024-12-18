import numpyro
import numpyro.distributions as dist
import jax.numpy as jnp
from jax.scipy.special import erf
from jax import vmap
import random

from scipy.io import loadmat
import numpy as np

from custom_distributions import *
from constants import *


logit_std = 2.5
tmp_min = 2.
tmp_max = 100.

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

    logR20 = numpyro.sample("logR20",dist.Uniform(-2,1))
    alpha_ref = numpyro.sample("alpha_ref",dist.Normal(-2,3))
    high_alpha = numpyro.sample("high_alpha", dist.Normal(-2, 3))
    log_width_alpha = numpyro.sample("log_width_alpha", dist.Uniform(-1, 1))
    width_alpha = numpyro.deterministic("width_alpha", 10.**log_width_alpha)
    middle_z_alpha = numpyro.sample("middle_z_alpha", dist.Uniform(0,0.8))

    mu_m1 = numpyro.sample("mu_m1",dist.Uniform(15,60))
    mMin = numpyro.sample("mMin",dist.Uniform(5,15))
    
    high_mMin = numpyro.sample("high_mMin", dist.Uniform(5,15))
    # delta_mMax = numpyro.sample("delta_mMax", dist.Normal(0, 5))
    log_width_mMin = numpyro.sample("log_width_mMin", dist.Uniform(-1, 1))
    width_mMin = numpyro.deterministic("width_mMin", 10.**log_width_mMin)
    middle_z_mMin = numpyro.sample("middle_z_mMin", dist.Uniform(0,0.8))

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

    # log_f_peak = numpyro.sample("log_f_peak",TransformedUniform(-3.,0.))
    logit_log_f_peak = numpyro.sample("logit_log_f_peak",dist.Normal(0,logit_std))
    log_f_peak,jac_log_f_peak = get_value_from_logit(logit_log_f_peak,-6. ,0.) # -6 was -5 was -3
    numpyro.deterministic("log_f_peak",log_f_peak)
    numpyro.factor("p_log_f_peak",logit_log_f_peak**2/(2.*logit_std**2)-jnp.log(jac_log_f_peak))

    log_high_f_peak = numpyro.sample("log_high_f_peak", dist.Uniform(-6, 0))
    log_width_f_peak = numpyro.sample("log_width_f_peak", dist.Uniform(-1, 1))
    width_f_peak = numpyro.deterministic("width_f_peak", 10.**log_width_f_peak)
    middle_z_f_peak = numpyro.sample("middle_z_f_peak", dist.Uniform(0,0.8))

    # mMax = numpyro.sample("mMax",TransformedUniform(50.,100.))
    logit_mMax = numpyro.sample("logit_mMax",dist.Normal(0,logit_std))
    mMax,jac_mMax = get_value_from_logit(logit_mMax,50. ,100.)
    numpyro.deterministic("mMax",mMax)
    numpyro.factor("p_mMax",logit_mMax**2/(2.*logit_std**2)-jnp.log(jac_mMax))

    high_mMax = numpyro.sample("high_mMax", dist.Uniform(50,100))
    # delta_mMax = numpyro.sample("delta_mMax", dist.Normal(0, 5))
    log_width_mMax = numpyro.sample("log_width_mMax", dist.Uniform(-1, 1))
    width_mMax = numpyro.deterministic("width_mMax", 10.**log_width_mMax)
    middle_z_mMax = numpyro.sample("middle_z_mMax", dist.Uniform(0,0.8))

    # log_dmMin = numpyro.sample("log_dmMin",TransformedUniform(-1.,0.5))
    logit_log_dmMin = numpyro.sample("logit_log_dmMin",dist.Normal(0,logit_std))
    log_dmMin,jac_log_dmMin = get_value_from_logit(logit_log_dmMin, -1. ,0.5)
    numpyro.deterministic("log_dmMin",log_dmMin)
    numpyro.factor("p_log_dmMin",logit_log_dmMin**2/(2.*logit_std**2)-jnp.log(jac_log_dmMin))
    
    log_high_dmMin = numpyro.sample("log_high_dmMin", dist.Uniform(-1, 0.5))
    log_width_dmMin = numpyro.sample('log_width_dmMin', dist.Uniform(-1, 1))
    width_dmMin = numpyro.deterministic("width_dmMin", 10.**log_width_dmMin)
    middle_z_dmMin = numpyro.sample("middle_z_dmMin", dist.Uniform(0, 0.8))

    # log_dmMax = numpyro.sample("log_dmMax",TransformedUniform(0.5,1.5))
    logit_log_dmMax = numpyro.sample("logit_log_dmMax",dist.Normal(0,logit_std))
    log_dmMax,jac_log_dmMax = get_value_from_logit(logit_log_dmMax,0.5 ,1.5)
    numpyro.deterministic("log_dmMax",log_dmMax)
    numpyro.factor("p_log_dmMax",logit_log_dmMax**2/(2.*logit_std**2)-jnp.log(jac_log_dmMax))

    log_high_dmMax = numpyro.sample("log_high_dmMax", dist.Uniform(0.5, 1.5))
    log_width_dmMax = numpyro.sample('log_width_dmMax', dist.Uniform(-1, 1))
    width_dmMax = numpyro.deterministic("width_dmMax", 10.**log_width_dmMax)
    middle_z_dmMax = numpyro.sample("middle_z_dmMax", dist.Uniform(0, 0.8))

    # mu_chi = numpyro.sample("mu_chi",TransformedUniform(0.,1.))
    logit_mu_chi = numpyro.sample("logit_mu_chi",dist.Normal(0,logit_std))
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
    p_m1_norm = massModel_variation_all_m1_power_law(jnp.array([20]), alpha_ref, high_alpha, width_alpha, middle_z_alpha,
                                           mu_m1, sig_m1, log_f_peak, log_high_f_peak, width_f_peak, middle_z_f_peak,
                                           mMax, high_mMax, width_mMax, middle_z_mMax, 
                                           mMin, high_mMin, width_mMin, middle_z_mMin,
                                           10.**log_dmMax, 10.**log_high_dmMax, width_dmMax, middle_z_dmMax, 
                                                     10.**log_dmMin, 10.**log_high_dmMin, width_dmMin, middle_z_dmMin, jnp.array([0.2]))
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
    p_m1_det =  massModel_variation_all_m1_power_law(m1_det, alpha_ref, high_alpha, width_alpha, middle_z_alpha,
                                           mu_m1, sig_m1, log_f_peak, log_high_f_peak, width_f_peak, middle_z_f_peak,
                                           mMax, high_mMax, width_mMax, middle_z_mMax, 
                                           mMin, high_mMin, width_mMin, middle_z_mMin,
                                           10.**log_dmMax, 10.**log_high_dmMax, width_dmMax, middle_z_dmMax, 
                                                     10.**log_dmMin, 10.**log_high_dmMin, width_dmMin, middle_z_dmMin, z_det)/p_m1_norm
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
        
        p_m1 =  massModel_variation_all_m1_power_law(m1_sample, alpha_ref, high_alpha, width_alpha, middle_z_alpha,
                                           mu_m1, sig_m1, log_f_peak, log_high_f_peak, width_f_peak, middle_z_f_peak,
                                           mMax, high_mMax, width_mMax, middle_z_mMax, 
                                           mMin, high_mMin, width_mMin, middle_z_mMin,
                                           10.**log_dmMax, 10.**log_high_dmMax, width_dmMax, middle_z_dmMax, 
                                                     10.**log_dmMin, 10.**log_high_dmMin, width_dmMin, middle_z_dmMin, z_sample)/p_m1_norm
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
