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


logit_std = 2.5
tmp_min = 2.


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

    mu_m1 = numpyro.sample("mu_m1",dist.Uniform(15,60))
    
    mMin = numpyro.sample("mMin",dist.Uniform(5,15))

    bq = numpyro.sample("bq",dist.Normal(0,3))

    R20 = numpyro.deterministic("R20",10.**logR20)

    alpha_z = numpyro.sample("alpha_z",dist.Normal(0,4))
    high_alpha_z = numpyro.sample("high_alpha_z", dist.Normal(0,4))
    log_width_alpha_z = numpyro.sample("log_width_alpha_z", dist.Uniform(-1, 1))
    # logit_width_alpha_z = numpyro.sample("logit_log_width_alpha_z",dist.Normal(0,logit_std))
    # log_width_alpha_z,jac_log_width_alpha_z = get_value_from_logit(logit_width_alpha_z,-1. ,1.)
    # numpyro.deterministic("log_width_alpha_z",log_width_alpha_z)
    # numpyro.factor("p_log_width_alpha_z",logit_width_alpha_z**2/(2.*logit_std**2)-jnp.log(jac_log_width_alpha_z))
    width_alpha_z = numpyro.deterministic("width_alpha_z", 10.**log_width_alpha_z)
    # middle_m_alpha_z = numpyro.sample("middle_m_alpha_z", dist.Uniform(20, 90))
    logit_middle_m_alpha_z = numpyro.sample("logit_middle_m_alpha_z",dist.Normal(0,logit_std))
    middle_m_alpha_z,jac_middle_m_alpha_z = get_value_from_logit(logit_middle_m_alpha_z,20. ,75.)
    numpyro.deterministic("middle_m_alpha_z",middle_m_alpha_z)
    numpyro.factor("p_middle_m_alpha_z",logit_middle_m_alpha_z**2/(2.*logit_std**2)-jnp.log(jac_middle_m_alpha_z))
    
    beta_z = numpyro.sample("beta_z",dist.Uniform(0,10))
    high_beta_z = numpyro.sample("high_beta_z", dist.Uniform(0,10))
    log_width_beta_z = numpyro.sample("log_width_beta_z", dist.Uniform(-1, 1))
    # logit_width_beta_z = numpyro.sample("logit_log_width_beta_z",dist.Normal(0,logit_std))
    # log_width_beta_z,jac_log_width_beta_z = get_value_from_logit(logit_width_beta_z,-1. ,1.)
    # numpyro.deterministic("log_width_beta_z",log_width_beta_z)
    # numpyro.factor("p_log_width_beta_z",logit_width_beta_z**2/(2.*logit_std**2)-jnp.log(jac_log_width_beta_z))
    width_beta_z = numpyro.deterministic("width_beta_z", 10.**log_width_beta_z)
    # middle_m_beta_z = numpyro.sample("middle_m_beta_z", dist.Uniform(20, 90))
    logit_middle_m_beta_z = numpyro.sample("logit_middle_m_beta_z",dist.Normal(0,logit_std))
    middle_m_beta_z,jac_middle_m_beta_z = get_value_from_logit(logit_middle_m_beta_z,20. ,75.)
    numpyro.deterministic("middle_m_beta_z",middle_m_beta_z)
    numpyro.factor("p_middle_m_beta_z",logit_middle_m_beta_z**2/(2.*logit_std**2)-jnp.log(jac_middle_m_beta_z))

    low_zp = numpyro.sample("low_zp", dist.Uniform(0.2, 4)) 
    high_zp = numpyro.sample("high_zp", dist.Uniform(0.2, 4))
    log_width_zp = numpyro.sample("log_width_zp", dist.Uniform(-1, 1))
    # logit_width_zp = numpyro.sample("logit_log_width_zp",dist.Normal(0,logit_std))
    # log_width_zp,jac_log_width_zp = get_value_from_logit(logit_width_zp,-1. ,1.)
    # numpyro.deterministic("log_width_zp",log_width_zp)
    # numpyro.factor("p_log_width_zp",logit_width_zp**2/(2.*logit_std**2)-jnp.log(jac_log_width_zp))
    width_zp = numpyro.deterministic("width_zp", 10.**log_width_zp)
    # middle_m_zp = numpyro.sample("middle_m_zp", dist.Uniform(20, 90))
    logit_middle_m_zp = numpyro.sample("logit_middle_m_zp",dist.Normal(0,logit_std))
    middle_m_zp,jac_middle_m_zp = get_value_from_logit(logit_middle_m_zp,20. ,75.)
    numpyro.deterministic("middle_m_zp",middle_m_zp)
    numpyro.factor("p_middle_m_zp",logit_middle_m_zp**2/(2.*logit_std**2)-jnp.log(jac_middle_m_zp))

    logit_sig_m1 = numpyro.sample("logit_sig_m1",dist.Normal(0,logit_std))
    sig_m1,jac_sig_m1 = get_value_from_logit(logit_sig_m1,1.5 ,15.)
    numpyro.deterministic("sig_m1",sig_m1)
    numpyro.factor("p_sig_m1",logit_sig_m1**2/(2.*logit_std**2)-jnp.log(jac_sig_m1))

    logit_log_f_peak = numpyro.sample("logit_log_f_peak",dist.Normal(0,logit_std))
    log_f_peak,jac_log_f_peak = get_value_from_logit(logit_log_f_peak,-6. ,0.)
    numpyro.deterministic("log_f_peak",log_f_peak)
    numpyro.factor("p_log_f_peak",logit_log_f_peak**2/(2.*logit_std**2)-jnp.log(jac_log_f_peak))
    f_peak= numpyro.deterministic("f_peak",10.**log_f_peak)

    logit_mMax = numpyro.sample("logit_mMax",dist.Normal(0,logit_std))
    mMax,jac_mMax = get_value_from_logit(logit_mMax,50. ,100.)
    numpyro.deterministic("mMax",mMax)
    numpyro.factor("p_mMax",logit_mMax**2/(2.*logit_std**2)-jnp.log(jac_mMax))

    logit_log_dmMin = numpyro.sample("logit_log_dmMin",dist.Normal(0,logit_std))
    log_dmMin,jac_log_dmMin = get_value_from_logit(logit_log_dmMin, -1. ,0.5)
    numpyro.deterministic("log_dmMin",log_dmMin)
    numpyro.factor("p_log_dmMin",logit_log_dmMin**2/(2.*logit_std**2)-jnp.log(jac_log_dmMin))

    logit_log_dmMax = numpyro.sample("logit_log_dmMax",dist.Normal(0,logit_std))
    log_dmMax,jac_log_dmMax = get_value_from_logit(logit_log_dmMax,0.5 ,1.5)
    numpyro.deterministic("log_dmMax",log_dmMax)
    numpyro.factor("p_log_dmMax",logit_log_dmMax**2/(2.*logit_std**2)-jnp.log(jac_log_dmMax))

    logit_mu_chi= numpyro.sample("logit_mu_chi",dist.Normal(0,logit_std))
    mu_chi,jac_mu_chi = get_value_from_logit(logit_mu_chi,0. ,1.)
    numpyro.deterministic("mu_chi",mu_chi)
    numpyro.factor("p_mu_chi",logit_mu_chi**2/(2.*logit_std**2)-jnp.log(jac_mu_chi))

    logit_logsig_chi = numpyro.sample("logit_logsig_chi",dist.Normal(0,logit_std))
    logsig_chi,jac_logsig_chi = get_value_from_logit(logit_logsig_chi,-1. ,0.)
    numpyro.deterministic("logsig_chi",logsig_chi)
    numpyro.factor("p_logsig_chi",logit_logsig_chi**2/(2.*logit_std**2)-jnp.log(jac_logsig_chi))

    logit_sig_cost = numpyro.sample("logit_sig_cost",dist.Normal(0,logit_std))
    sig_cost,jac_sig_cost = get_value_from_logit(logit_sig_cost,0.3 ,2.)
    numpyro.deterministic("sig_cost",sig_cost)
    numpyro.factor("p_sig_cost",logit_sig_cost**2/(2.*logit_std**2)-jnp.log(jac_sig_cost))

    # Fixed params
    mu_cost = 1.

    # Normalization
    alpha_for_norm = alpha_ref
    p_m1_norm = massModel_no_variation(20.,alpha_for_norm,mu_m1,sig_m1,10.**log_f_peak,mMax,mMin,10.**log_dmMax,10.**log_dmMin)
    p_z_norm = merger_rate_zp_sigmoid(alpha_z, high_alpha_z, width_alpha_z, middle_m_alpha_z,
                           beta_z, high_beta_z, width_beta_z, middle_m_beta_z,
                           low_zp, high_zp, width_zp, middle_m_zp, 20., 0.2)

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
    p_m1_det = massModel_no_variation(m1_det, alpha_ref, mu_m1,sig_m1,10.**log_f_peak,mMax,mMin,10.**log_dmMax,10.**log_dmMin)/p_m1_norm
    p_m2_det = (1.+bq)*m2_det**bq/(m1_det**(1.+bq)-tmp_min**(1.+bq))
    p_a1_det = truncatedNormal(a1_det,mu_chi,10.**logsig_chi,0,1)
    p_a2_det = truncatedNormal(a2_det,mu_chi,10.**logsig_chi,0,1)
    p_cost1_det = truncatedNormal(cost1_det,mu_cost,sig_cost,-1,1)
    p_cost2_det = truncatedNormal(cost2_det,mu_cost,sig_cost,-1,1)

    rate_det = merger_rate_zp_sigmoid(alpha_z, high_alpha_z, width_alpha_z, middle_m_alpha_z,
                           beta_z, high_beta_z, width_beta_z, middle_m_beta_z,
                           low_zp, high_zp, width_zp, middle_m_zp, m1_det, z_det)

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
        
        p_m1 = massModel_no_variation(m1_sample,alpha_ref, mu_m1,sig_m1,10.**log_f_peak,mMax,mMin,10.**log_dmMax,10.**log_dmMin)/p_m1_norm
        p_m2 = (1.+bq)*m2_sample**bq/(m1_sample**(1.+bq)-tmp_min**(1.+bq))
        p_a1 = truncatedNormal(a1_sample,mu_chi,10.**logsig_chi,0,1)
        p_a2 = truncatedNormal(a2_sample,mu_chi,10.**logsig_chi,0,1)
        p_cost1 = truncatedNormal(cost1_sample,mu_cost,sig_cost,-1,1)
        p_cost2 = truncatedNormal(cost2_sample,mu_cost,sig_cost,-1,1)

        rate = merger_rate_zp_sigmoid(alpha_z, high_alpha_z, width_alpha_z, middle_m_alpha_z,
                           beta_z, high_beta_z, width_beta_z, middle_m_beta_z,
                           low_zp, high_zp, width_zp, middle_m_zp, m1_sample, z_sample)
        #merger_rate_varied_zp(alpha_z, beta_z, zp, dzp_dm, m1_sample, z_sample)
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
