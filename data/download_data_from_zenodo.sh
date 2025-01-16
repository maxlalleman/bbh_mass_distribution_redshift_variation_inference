#!/bin/bash

# Download and unzip
curl https://zenodo.org/records/14671139/redshift_evolution_CBC_O3.zip --output "redshift_evolution_O3.zip"
unzip redshift_evolution_O3.zip

# Move input data to ../input/
mv redshift_evolution_O3/sampleDict_FAR_1_in_1_yr.pickle ../input/.
mv redshift_evolution_O3/injectionDict_FAR_1_in_1.pickle ../input/.

# Move input data to ../
mv redshift_evolution_O3/peak_analysis.hdf .
mv redshift_evolution_O3/power_law_analysis.hdf .
mv redshift_evolution_O3/mass_variation_analysis.hdf .
mv redshift_evolution_O3/all_varied.hdf .

# Remove original zip files and directory
rm redshift_evolution_O3.zip
rm -r redshift_evolution_O3/