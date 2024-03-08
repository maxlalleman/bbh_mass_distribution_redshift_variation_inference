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

# def merger_rate_varied_zp(alpha_z, beta_z, zp, dzp_dm, masses, zs):
  #  new_zp = zp + dzp_dm*(masses - 20)
    # new_zp[new_zp < 0.2] = 0.2
   # full_arr = jnp.full(len(new_zp), 0.2)
    # print(full_arr, new_zp)
    # new_zp = jnp.maximum(full_arr, new_zp)
    # new_zp = jnp.where(new_zp > 0.2, new_zp, 0.2)
    # print(new_zp)
    # return (1+zs)**alpha_z/(1+((1+zs)/(1+new_zp))**(alpha_z+beta_z))
    
def sigmoid_zp(low, high, width, middle, mass):
    return (high - low) / (1 + jnp.exp(-(1/width)*(mass - middle))) + low

def sigmoid_delta(low, delta, width, middle, mass):
    return (delta) / (1 + jnp.exp(-(1/width)*(mass - middle))) + low

def merger_rate_zp_sigmoid(alpha_z, delta_alpha_z, width_alpha_z, middle_m_alpha_z,
                           beta_z, high_beta_z, width_beta_z, middle_m_beta_z,
                           low_zp, high_zp, width_m, m_middle, masses, z_s):
    new_zp = sigmoid_zp(low_zp, high_zp, width_m, m_middle, masses)
    new_alpha = sigmoid_delta(alpha_z, delta_alpha_z, width_alpha_z, middle_m_alpha_z, masses)
    new_beta = sigmoid_zp(beta_z, high_beta_z, width_beta_z, middle_m_beta_z, masses)
    return (1+z_s)**new_alpha/(1+((1+z_s)/(1+new_zp))**(new_alpha+new_beta))

@jax.jit
def massModel_no_variation(m1, alpha_ref, mu_m1, sig_m1, f_peak, mMax, mMin, dmMax, dmMin):

    """
    Baseline primary mass model, described as a mixture between a power law
    and gaussian, with exponential tapering functions at high and low masses

    Parameters
    ----------
    m1 : array or float
        Primary masses at which to evaluate probability densities
    alpha : float
        Power-law index
    mu_m1 : float
        Location of possible Gaussian peak
    sig_m1 : float
        Stanard deviation of possible Gaussian peak
    f_peak : float
        Approximate fraction of events contained within Gaussian peak (not exact due to tapering)
    mMax : float
        Location at which high-mass tapering begins
    mMin : float
        Location at which low-mass tapering begins
    dmMax : float
        Scale width of high-mass tapering function
    dmMin : float
        Scale width of low-mass tapering function

    Returns
    -------
    p_m1s : jax.numpy.array
        Unnormalized array of probability densities
    """
    
    p_m1_pl = (1.+alpha_ref)*m1**(alpha_ref)/(tmp_max**(1.+alpha_ref) - tmp_min**(1.+alpha_ref))
        
    #get rid of if and else statements. trouble in compil + run. Instead call on a single number -> 1-element matrix jax array
    # "pre-expand" zs and m1s, define beforehand -> already have 3-D matrices. 
    # input of massmodel_with_z has to be 3-D array.
    
    #16:18

    #combined = f_of_m1[ : ,np.newaxis]*g_of_z[np.newaxis, : ] 
    # new axis makes it more like a "matrix". 
    #combined[ i, j] = f_of_m1[ i ] * g_of_z[j]

    p_m1_peak = jnp.exp(-(m1-mu_m1)**2/(2.*sig_m1**2))/jnp.sqrt(2.*np.pi*sig_m1**2)

    # Compute low- and high-mass filters
    low_filter = jnp.exp(-(m1-mMin)**2/(2.*dmMin**2))
    low_filter = jnp.where(m1<mMin,low_filter,1.)
    high_filter = jnp.exp(-(m1-mMax)**2/(2.*dmMax**2))
    high_filter = jnp.where(m1>mMax,high_filter,1.)

    #tryout = f_peak*p_m1_peak[0] + (1.-f_peak)*p_m1_pl[0]*low_filter[0]*high_filter[0]
    combined_p = jnp.array((f_peak*p_m1_peak + (1. - f_peak)*p_m1_pl)*low_filter*high_filter)
    #combined_p = [f_peak*m_peak + (1.-f_peak)*p_m1_pl[index]*low_filter[index]*high_filter[index] for index, m_peak in enumerate(p_m1_peak)]
    #index like normal, but if index changes jax can't compile. Jax freaks out in a for loop. 
    #print(np.shape(combined_p[0]))
    return combined_p #(f_peak*p_m1_peak + (1.-f_peak)*p_m1_pl)*low_filter*high_filter


def get_value_from_logit(logit_x,x_min,x_max):

    exp_logit = jnp.exp(logit_x)
    x = (exp_logit*x_max + x_min)/(1.+exp_logit)
    dlogit_dx = 1./(x-x_min) + 1./(x_max-x)

    return x,dlogit_dx

def cumsum(total,new_element):
    phi,w = new_element
    total = phi*total+w
    return total,total

##################################################
#######            Pre-computing           #######
##################################################

N = 20000

m1s_drawn = np.random.uniform(tmp_min, tmp_max, size=N)

c_m2s = np.random.uniform(size=int(N))
m2s_drawn = tmp_min**(1.)+c_m2s*(m1s_drawn**(1.)-tmp_min**(1.))
    
zs_drawn = np.random.uniform(0,10,size=N)

# Loading in stochastic dict
matdata = loadmat("/home/max.lalleman/CBC_Stoch_Search/gwbcbcmodeling/gwbcbcmodeling/New_Code_New_sample/full_combined_results_alpha0.mat")
Cf = np.array(matdata['ptEst_ff']).reshape(-1)
sigmas = np.array(matdata['sigma_ff']).reshape(-1)
freqs = np.array(matdata['freq']).reshape(-1)

# Select frequencies below 200 Hz
lowFreqs = freqs<200.
freqs = freqs[lowFreqs]
Cf = Cf[lowFreqs]
sigma2s = sigmas[lowFreqs]**2.

# loaded_data = np.load("/home/max.lalleman/CBC_Stoch_Search/gwbcbcmodeling/gwbcbcmodeling/New_Code_New_sample/Varying_z_peak/Data_for_O5_run_above_new_prior.npz")
# Cf = np.array(loaded_data["omega_f"])
# var_f = np.array(loaded_data["var_f"])
# freqs = np.array(loaded_data["freqs"])
# 
# # Select frequencies below 200 Hz
# lowFreqs = freqs<200.
# freqs = freqs[lowFreqs]
# Cf = Cf[lowFreqs]
# sigma2s = var_f[lowFreqs]

# Select only frequencies with data
# This step removes frequency bins that have been notched due to the presence of
# loud or unsafe lines

goodInds = np.where(Cf==Cf)
freqs = freqs[goodInds]
Cf = Cf[goodInds]
sigma2s = sigma2s[goodInds]

stochasticDict = {'freqs': freqs, 'Cf': np.real(Cf), 'sigma2s': sigma2s}

# Computing dEdfs

freqs = stochasticDict['freqs']

dEdfs = np.array([dEdf(m1s_drawn[ii]+m2s_drawn[ii],freqs*(1+zs_drawn[ii]),eta=m2s_drawn[ii]/m1s_drawn[ii]/(1+m2s_drawn[ii]/m1s_drawn[ii])**2) for ii in range(N)])

p_m1_old = 1/(tmp_max-tmp_min)*np.ones(N)
p_z_old = 1/(10-0)*np.ones(N)
p_m2_old = 1/(m1s_drawn-tmp_min)


##################################################
#######            Likelihoods             #######
##################################################
all_zs = np.linspace(0,10,200) # Should be 1000, just took it 10 for testing
# omg = OmegaGW_BBH(2.1,100.,all_zs)

# Note: mMin was 2, but this caused issues with tmp_min

def combined_pop_gwb_cbc_redshift_mass(sampleDict,injectionDict, stochasticProds, run_stoch=True):
    """
    Implementation of a Gaussian effective spin distribution for inference within `numpyro`

    Parameters
    ----------
    sampleDict : dict
        Precomputed dictionary containing posterior samples for each event in our catalog
    injectionDict : dict
        Precomputed dictionary containing successfully recovered injections
    stochasticProds: dict
        Dictionary with arrays of frequencies ('freqs'), point estimate spectrum ('Cf') and 
        variance spectrum ('sigma2s') from stochastic search
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

    mu_m1 = numpyro.sample("mu_m1",dist.Uniform(20,50))
    
    mMin = numpyro.sample("mMin",dist.Uniform(5,15))

    bq = numpyro.sample("bq",dist.Normal(0,3))

    # kappa = numpyro.sample("kappa",dist.Uniform(-25,25))
    R20 = numpyro.deterministic("R20",10.**logR20)

    # alpha_z = numpyro.sample("alpha_z",dist.Uniform(-25,25))
    alpha_z = numpyro.sample("alpha_z",dist.Normal(0,4))
    delta_alpha_z = numpyro.sample("delta_alpha_z", dist.Normal(0,1))
    log_width_alpha_z = numpyro.sample("log_width_alpha_z", dist.Uniform(-1, 1))
    width_alpha_z = numpyro.deterministic("width_alpha_z", 10.**log_width_alpha_z)
    middle_m_alpha_z = numpyro.sample("middle_m_alpha_z", dist.Uniform(30, 75))
    
    beta_z = numpyro.sample("beta_z",dist.Uniform(0,10))
    high_beta_z = numpyro.sample("high_beta_z", dist.Uniform(0,10))
    log_width_beta_z = numpyro.sample("log_width_beta_z", dist.Uniform(-1, 1))
    width_beta_z = numpyro.deterministic("width_beta_z", 10.**log_width_beta_z)
    middle_m_beta_z = numpyro.sample("middle_m_beta_z", dist.Uniform(30, 75))

    low_zp = numpyro.sample("low_zp", dist.Uniform(0.2, 4)) # dist.Normal(2,5e-1))
    high_zp = numpyro.sample("high_zp", dist.Uniform(0.2, 4))
    # high_zp = numpyro.sample("high_zp", dist.Normal(5,1))
    log_width_zp = numpyro.sample("log_width_zp", dist.Uniform(-1, 1)) # cannot let this be negative, I think
    width_zp = numpyro.deterministic("width_zp", 10.**log_width_zp)
    middle_m = numpyro.sample("middle_m", dist.Uniform(30, 75)) # 15, 85
    # zp = numpyro.sample("zp",dist.Uniform(0.2,4))
    # dzp_dm = numpyro.sample("dzp_dm", dist.Normal(0,5e-2))

    # logit_log_steep_zp = numpyro.sample("logit_log_steep_zp",dist.Normal(0,logit_std))
    # log_steep_zp,jac_log_steep_zp = get_value_from_logit(logit_log_steep_zp,-1. ,2.)
    # numpyro.deterministic("log_steep_zp",log_steep_zp)
    # numpyro.factor("p_log_steep_zp",logit_log_steep_zp**2/(2.*logit_std**2)-jnp.log(jac_log_steep_zp))
    # steep_zp = numpyro.deterministic("steep_zp",10.**log_steep_zp)

    # sig_m1 = numpyro.sample("sig_m1",TransformedUniform(1.5,15.))
    logit_sig_m1 = numpyro.sample("logit_sig_m1",dist.Normal(0,logit_std))
    sig_m1,jac_sig_m1 = get_value_from_logit(logit_sig_m1,1.5 ,15.)
    numpyro.deterministic("sig_m1",sig_m1)
    numpyro.factor("p_sig_m1",logit_sig_m1**2/(2.*logit_std**2)-jnp.log(jac_sig_m1))

    # log_f_peak = numpyro.sample("log_f_peak",TransformedUniform(-3.,0.))
    logit_log_f_peak = numpyro.sample("logit_log_f_peak",dist.Normal(0,logit_std))
    log_f_peak,jac_log_f_peak = get_value_from_logit(logit_log_f_peak,-5. ,0.) # -5 was -3
    numpyro.deterministic("log_f_peak",log_f_peak)
    numpyro.factor("p_log_f_peak",logit_log_f_peak**2/(2.*logit_std**2)-jnp.log(jac_log_f_peak))
    f_peak= numpyro.deterministic("f_peak",10.**log_f_peak)

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
    p_m1_norm = massModel_no_variation(20.,alpha_for_norm,mu_m1,sig_m1,10.**log_f_peak,mMax,mMin,10.**log_dmMax,10.**log_dmMin)
    p_z_norm = merger_rate_zp_sigmoid(alpha_z, delta_alpha_z, width_alpha_z, middle_m_alpha_z,
                           beta_z, high_beta_z, width_beta_z, middle_m_beta_z,
                           low_zp, high_zp, width_zp, middle_m, 20., 0.2)
    #merger_rate_varied_zp(alpha_z, beta_z, zp, dzp_dm, jnp.array([20]), jnp.array([0.2]))

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

    rate_det = merger_rate_zp_sigmoid(alpha_z, delta_alpha_z, width_alpha_z, middle_m_alpha_z,
                           beta_z, high_beta_z, width_beta_z, middle_m_beta_z,
                           low_zp, high_zp, width_zp, middle_m, m1_det, z_det)
    #merger_rate_varied_zp(alpha_z, beta_z, zp, dzp_dm, m1_det, z_det)
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

        rate = merger_rate_zp_sigmoid(alpha_z, delta_alpha_z, width_alpha_z, middle_m_alpha_z,
                           beta_z, high_beta_z, width_beta_z, middle_m_beta_z,
                           low_zp, high_zp, width_zp, middle_m, m1_sample, z_sample)
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

    if run_stoch:
        def logp_gwb(freqs, Cf, sigma2s):
            p_m2_new = ((1.+bq)*jnp.power(m2s_drawn,bq)/(jnp.power(m1s_drawn,1.+bq)-tmp_min**(1.+bq)))
            p_m1_new = massModel_no_variation(m1s_drawn,alpha_ref, mu_m1,sig_m1,10.**log_f_peak, mMax, mMin, 10.**log_dmMax,10.**log_dmMin)
            rate = merger_rate_zp_sigmoid(alpha_z, delta_alpha_z, width_alpha_z, middle_m_alpha_z,
                           beta_z, high_beta_z, width_beta_z, middle_m_beta_z,
                           low_zp, high_zp, width_zp, middle_m, m1s_drawn, zs_drawn)
            #merger_rate_varied_zp(alpha_z, beta_z, zp, dzp_dm, m1s_drawn, zs_drawn)
            p_z_new = rate/((1.+zs_drawn)*jnp.sqrt(OmgM*(1.+zs_drawn)**3.+OmgL))
            
            w_i = p_m1_new*p_m2_new*p_z_new/(p_m1_old*p_m2_old*p_z_old)
            Omega_spectra_NEW = (freqs)*(jnp.einsum("if,i->if",dEdfs,w_i))
            Omega_spectra_NEW_avg = 1/rhoC/H0*R20/1e9/year/merger_rate_zp_sigmoid(alpha_z, delta_alpha_z, width_alpha_z, middle_m_alpha_z,
                           beta_z, high_beta_z, width_beta_z, middle_m_beta_z,
                           low_zp, high_zp, width_zp, middle_m, jnp.array([20]), jnp.array([0.2]))/massModel_no_variation(20, alpha_ref, mu_m1,sig_m1,10.**log_f_peak,mMax, mMin, 10.**log_dmMax,10.**log_dmMin)*jnp.mean(Omega_spectra_NEW, axis=0)        
            
            Omega_f = Omega_spectra_NEW_avg
            diff = Omega_f-Cf
            log_stoch = -0.5*jnp.sum((diff**2/sigma2s))
            return log_stoch
    
    
        freqs = stochasticProds['freqs']
        Cf = stochasticProds['Cf']
        sigma2s = stochasticProds['sigma2s']
    
        logps_gwb = logp_gwb(freqs, Cf, sigma2s)
    
        numpyro.factor("logp_gwb", logps_gwb)
