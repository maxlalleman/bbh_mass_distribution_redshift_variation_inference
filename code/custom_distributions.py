import jax
import jax.numpy as jnp


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
def massModel_variation_all_m1_peak(m1, alpha_ref, mu_m1, delta_mu, width_mu, middle_mu,
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


@jax.jit
def massModel_variation_all_m1_power_law(m1, alpha_ref, delta_alpha, width_alpha, middle_alpha,
                               mu_m1, sig_m1, log_f_peak, log_high_f_peak, width_f_peak, middle_f_peak,
                               mMax, high_mMax, width_mMax, middle_mMax,
                               mMin, dmMax, high_dmMax, width_dm, middle_dm, dmMin, zs):

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
    alpha_new = sigmoid(alpha_ref, delta_alpha, width_alpha, middle_alpha, zs)
    p_m1_pl = (1.+alpha_new)*m1**(alpha_new)/(tmp_max**(1.+alpha_new) - tmp_min**(1.+alpha_new))

    p_m1_peak = jnp.exp(-(m1-mu_m1)**2/(2.*sig_m1**2))/jnp.sqrt(2.*np.pi*sig_m1**2)
    
    new_mMax = sigmoid_initial_final_no_delta(mMax, high_mMax, width_mMax, middle_mMax, zs)
    new_dmMax = sigmoid_initial_final_no_delta(dmMax, high_dmMax, width_dm, middle_dm, zs)

    # Compute low- and high-mass filters
    low_filter = jnp.exp(-(m1-mMin)**2/(2.*dmMin**2))
    low_filter = jnp.where(m1<mMin,low_filter,1.)
    high_filter = jnp.exp(-(m1-new_mMax)**2/(2.*new_dmMax**2))
    high_filter = jnp.where(m1>new_mMax,high_filter,1.)

    new_f_peak = sigmoid_initial_final_no_delta(log_f_peak, log_high_f_peak, width_f_peak, middle_f_peak, zs)
    actual_f_peak = 10.**(new_f_peak)
    combined_p = jnp.array((actual_f_peak*p_m1_peak + (1. - actual_f_peak)*p_m1_pl)*low_filter*high_filter)
    return combined_p


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
    p_m1_peak = jnp.exp(-(m1-mu_m1)**2/(2.*sig_m1**2))/jnp.sqrt(2.*np.pi*sig_m1**2)

    # Compute low- and high-mass filters
    low_filter = jnp.exp(-(m1-mMin)**2/(2.*dmMin**2))
    low_filter = jnp.where(m1<mMin,low_filter,1.)
    high_filter = jnp.exp(-(m1-mMax)**2/(2.*dmMax**2))
    high_filter = jnp.where(m1>mMax,high_filter,1.)

    combined_p = jnp.array((f_peak*p_m1_peak + (1. - f_peak)*p_m1_pl)*low_filter*high_filter)
    return combined_p


def merger_rate_zp_sigmoid(alpha_z, delta_alpha_z, width_alpha_z, middle_m_alpha_z,
                           beta_z, high_beta_z, width_beta_z, middle_m_beta_z,
                           low_zp, high_zp, width_m, m_middle, masses, z_s):
    """
    Calculate the merger rate using a sigmoidal function for parameters related to redshift.

    Parameters:
    alpha_z : float
        Parameter controlling the first increasing redshift index of the merger rate.
    delta_alpha_z : float
        Parameter controlling the shift of the sigmoid function for alpha_z.
    width_alpha_z : float
        Width of the sigmoid function for alpha_z.
    middle_m_alpha_z : float
        Middle point of the sigmoid function for alpha_z.
    beta_z : float
        Parameter controlling the second decreasing redshift index of the merger rate.
    high_beta_z : float
        High value parameter for the sigmoid function for beta_z.
    width_beta_z : float
        Width of the sigmoid function for beta_z.
    middle_m_beta_z : float
        Middle point of the sigmoid function for beta_z.
    low_zp : float
        Low value parameter for the sigmoid function for the peak point.
    high_zp : float
        High value parameter for the sigmoid function for the peak point.
    width_m : float
        Width of the sigmoid function for the masses.
    m_middle : float
        Middle point of the sigmoid function for the masses.
    masses : array_like
        Array of masses.
    z_s : float
        Redshift value.

    Returns:
    float
        Merger rate calculated using the sigmoidal function.
    """
    new_zp = sigmoid_initial_final_no_delta(low_zp, high_zp, width_m, m_middle, masses)
    new_alpha = sigmoid(alpha_z, delta_alpha_z, width_alpha_z, middle_m_alpha_z, masses)
    new_beta = sigmoid_initial_final_no_delta(beta_z, high_beta_z, width_beta_z, middle_m_beta_z, masses)
    return (1+z_s)**new_alpha/(1+((1+z_s)/(1+new_zp))**(new_alpha+new_beta))


def get_value_from_logit(logit_x,x_min,x_max):
    """
    Transform a value from logit space to the original space.

    Parameters:
    logit_x : float
        The value in logit space.
    x_min : float
        The minimum value in the original space.
    x_max : float
        The maximum value in the original space.

    Returns:
    x : float
        The transformed value in the original space.
    dlogit_dx : float
        The derivative of the transformation with respect to the original space.
    """
    exp_logit = jnp.exp(logit_x)
    x = (exp_logit*x_max + x_min)/(1.+exp_logit)
    dlogit_dx = 1./(x-x_min) + 1./(x_max-x)
    return x,dlogit_dx
