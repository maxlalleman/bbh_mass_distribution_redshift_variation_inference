# bbh_mass_distribution_redshift_variation_inference

This repository contains the code and necessary material to reproduce the results of

"**Titel paper**"

In this paper, we infer the variation in redshift of the binary black hole mass distribution from gravitational-wave (GW) data, using individual binary black hole mergers. 
We consider the GW events from LIGO-Virgo-KAGRA's third observing run. We discuss the variation in redshift of the mass distribution in three different sectors.
The peak sector, containing all parameters relevant to the peak part of the mass distribution, the power-law sector where the parameters of the power-law sector are varied, and the third 'reverse' sector, where we vary the redshift dependent part of the merger rate with mass. This last one can be seen as the opposite analysis, but gives the same results.

The data can be downloaded from Zenodo . A pre-print of the paper can be found .

## Getting started

The code relies on several packages, which can easily be downloaded in a conda environment using the provided yml file in the repository.
To create a conda environment from the yml file, execute the following command `conda env create -f environment.yml`. To activate
the environment, simply call `conda activate gpu_env`.

## Structure of the repository

The repository contains several folders, each with a specific purpose:

- **code**: Contains the code to run the analysis which produced the results in this paper.
- **data**: Contains the script to download from Zenodo , the data used in this paper, as well as the results produced in this paper.
- **figures**: Contains several notebooks to reproduce the figures of the paper. Note that all the data should have been downloaded
    from Zenodo in order to run the notebooks in this folder.