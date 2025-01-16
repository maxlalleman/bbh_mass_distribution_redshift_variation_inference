#!/bin/bash

# Run on O3 BBH data - peak analysis

python3 process_cdf.py -inputfile ./peak_analysis.cdf -outputfile ./peak_analysis.hdf -which_analysis peak

# Run on O3 BBH data - power-law analysis

python3 process_cdf.py -inputfile ./power_law_analysis.cdf -outputfile ./power_law_analysis.hdf -which_analysis power-law

# Run on O3 BBH data - mass-variation analysis

python3 process_cdf.py -inputfile ./mass_variation_analysis.cdf -outputfile ./mass_variation_analysis.hdf -which_analysis mass-variation

# Run on O3 BBH data - all-varying analysis

python3 process_cdf.py -inputfile ./all_varied.cdf -outputfile ./all_varied.hdf -which_analysis all