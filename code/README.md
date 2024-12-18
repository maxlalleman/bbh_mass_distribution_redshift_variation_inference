# Code

This subdirectory contains the actual analysis code used to produce the results of all the different sector analyses of the paper. We divided the analysis into three different parts. The peak sector script to perform the peak sector analysis is `run_combined_pop_peak.py`. Other are similarly named for the power-law and the redshift merger rate in mass variation sector.

These run scripts take need functions from `likelihoods.py` scripts in which the utilised mass distributions and inference likelihoods are defined.

Additionally, this folder contains a script that reads in data, like selection effects and the actual gravitational-wave catalog, `getData.py`, a script that defines some custom distribution to get rid of certain divergence effects, `custom_distributions.py`, a script that defines cosmological constants, `constants.py`, and cosmological properties, `get_cosmo.py`.

*Note: to run the code in this subdirectory, it is likely that you first need to download the data,
as specified in the `/data` subdirectory.*

