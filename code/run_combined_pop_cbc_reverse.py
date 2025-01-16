import numpyro
import numpy as np
ndevice = 1
numpyro.set_host_device_count(ndevice)
numpyro.set_platform('gpu')
from numpyro.infer import NUTS,MCMC
from jax import random
import arviz as az

from jax.config import config
config.update("jax_enable_x64", True)

from likelihoods_reverse import combined_pop_gwb_cbc_redshift_mass
from getData import *
from get_cosmo import *
from scipy.io import loadmat

# Get dictionaries holding injections and posterior samples
injectionDict = getInjections(reweight=False)
sampleDict = getSamples(sample_limit=2000,reweight=False)

# Set up NUTS sampler over our likelihood
kernel = NUTS(combined_pop_gwb_cbc_redshift_mass, 
        target_accept_prob = 0.95
             )
mcmc = MCMC(kernel,num_warmup=2000,num_samples=3000,num_chains=1)

# Choose a random key and run over our model
rng_key = random.PRNGKey(110)
rng_key,rng_key_ = random.split(rng_key)
mcmc.run(rng_key_,sampleDict,injectionDict)
mcmc.print_summary()

# Save out data
data = az.from_numpyro(mcmc)
save_path = "../data/mass_variation_analysis.cdf"
az.to_netcdf(data, save_path)

