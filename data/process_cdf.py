import arviz as az
import numpy as np
import h5py
import argparse


dict_variables = {
    "all_analyses":[
        "R20",
        "alpha_ref",
        "mu_m1",
        "sig_m1",
        "log_f_peak",
        "mMin",
        "mMax",        
        "log_dmMax",
        "log_dmMin",
        "bq",
        "alpha_z",
        "beta_z",
        "zp",
        "mu_chi",
        "logsig_chi",
        "sig_cost",
        "nEff_inj_per_event",
        "min_log_neff"
    ],
    "peak":[
        "high_mu",
        "width_mu",
        "middle_z_mu",
        "high_sig",
        "width_sig",
        "middle_z_sig",
        "log_high_f_peak",
        "width_f_peak",
        "middle_z_f_peak",
    ],
    "power-law":[
        "high_alpha",
        "width_alpha",
        "middle_z_alpha",
        "high_mMin",
        "width_mMin",
        "middle_z_mMin",
        "high_mMax",
        "width_mMax",
        "middle_z_mMax",
        "log_high_dmMax",
        "width_dmMax",
        "middle_z_dmMax",
        "log_high_dmMin",
        "width_dmMin",
        "middle_z_dmMin",
        "log_high_f_peak",
        "width_f_peak",
        "middle_z_f_peak",
    ],
    "mass-variation":[
        "high_alpha_z",
        "width_alpha_z",
        "middle_m_alpha_z",
        "high_beta_z",
        "width_beta_z",
        "middle_m_beta_z",
        "high_zp",
        "width_zp",
        "middle_m_zp",
    ],
}

dict_variables["all"] = dict_variables["peak"] + dict_variables["power-law"]

def process(inputfile, outputfile, which_analysis = "peak"):
    # Load inference results
    inference_data = az.from_netcdf(inputfile)
    var_names = dict_variables["all_analyses"] + dict_variables[which_analysis]
    samps = inference_data.posterior.stack(draws=("chain", "draw"))

    R_20 = samps.R20.values
    alpha_ref = samps.alpha_ref.values
    mu_m1 = samps.mu_m1.values
    sig_m1 = samps.sig_m1.values
    log_f_peak = samps.log_f_peak.values
    mMin = samps.mMin.values
    mMax = samps.mMax.values
    log_dmMin = samps.log_dmMin.values
    log_dmMax = samps.log_dmMax.values
    bq = samps.bq.values
    alpha_z = samps.alpha_z.values
    beta_z = samps.beta_z.values
    mu_chi = samps.mu_chi.values
    logsig_chi = samps.logsig_chi.values
    sig_cost = samps.sig_cost.values
    nEff_inj_per_event = samps.nEff_inj_per_event.values
    min_log_neff = samps.min_log_neff.values
    
    # Different inference results depending on analysis
    if which_analysis == "peak" or which_analysis == "all":
        high_mu = samps.high_mu.values
        width_mu = samps.width_mu.values
        middle_z_mu = samps.middle_z_mu.values
        high_sig = samps.high_sig.values
        width_sig = samps.width_sig.values
        middle_z_sig = samps.middle_z_sig.values
    
    if which_analysis == "peak" or which_analysis == "power-law" or which_analysis == "all":
        log_high_f_peak = samps.log_high_f_peak.values
        width_f_peak = samps.width_f_peak.values
        middle_z_f_peak = samps.middle_z_f_peak.values
        zp = samps.zp.values
        
    if which_analysis == "power-law" or which_analysis == "all":
        high_alpha = samps.high_alpha.values
        width_alpha = samps.width_alpha.values
        middle_z_alpha = samps.middle_z_alpha.values
        high_mMin = samps.high_mMin.values
        width_mMin = samps.width_mMin.values
        middle_z_mMin = samps.middle_z_mMin.values
        high_mMax = samps.high_mMax.values
        width_mMax = samps.width_mMax.values
        middle_z_mMax = samps.middle_z_mMax.values
        log_high_dmMax = samps.log_high_dmMax.values
        width_dmMax = samps.width_dmMax.values
        middle_z_dmMax = samps.middle_z_dmMax.values
        log_high_dmMin = samps.log_high_dmMin.values
        width_dmMin = samps.width_dmMin.values
        middle_z_dmMin = samps.middle_z_dmMin.values
        
    if which_analysis == "mass-variation":
        high_alpha_z = samps.high_alpha_z.values
        width_alpha_z = samps.width_alpha_z.values
        middle_m_alpha_z = samps.middle_m_alpha_z.values
        high_beta_z = samps.high_beta_z.values
        width_beta_z = samps.width_beta_z.values
        middle_m_beta_z = samps.middle_m_beta_z.values
        low_zp = samps.low_zp.values
        high_zp = samps.high_zp.values
        width_zp = samps.width_zp.values
        middle_m_zp = samps.middle_m_zp.values

    # Create hdf5 file and write posterior samples
    hfile = h5py.File(outputfile,'w')
    posterior = hfile.create_group('posterior')
    posterior.create_dataset('R_20',data=R_20)
    posterior.create_dataset('alpha_ref',data=alpha_ref)
    posterior.create_dataset('mu_m1',data=mu_m1)
    posterior.create_dataset('sig_m1',data=sig_m1)
    posterior.create_dataset('log_f_peak',data=log_f_peak)
    posterior.create_dataset('mMin',data=mMin)
    posterior.create_dataset('mMax',data=mMax)
    posterior.create_dataset('log_dmMin',data=log_dmMin)
    posterior.create_dataset('log_dmMax',data=log_dmMax)
    posterior.create_dataset('bq',data=bq)
    posterior.create_dataset('alpha_z', data=alpha_z)
    posterior.create_dataset('beta_z', data=beta_z)
    posterior.create_dataset('mu_chi',data=mu_chi)
    posterior.create_dataset('logsig_chi',data=logsig_chi)
    posterior.create_dataset('sig_cost',data=sig_cost)
    posterior.create_dataset('nEff_inj_per_event',data=nEff_inj_per_event)
    posterior.create_dataset('min_log_neff',data=min_log_neff)
    
    # Differentiate again per analysis
    if which_analysis == "peak" or which_analysis == "all":
        posterior.create_dataset('high_mu',data = high_mu)
        posterior.create_dataset('width_mu',data = width_mu)
        posterior.create_dataset('middle_z_mu',data = middle_z_mu)
        posterior.create_dataset('high_sig',data = high_sig)
        posterior.create_dataset('width_sig',data = width_sig)
        posterior.create_dataset('middle_z_sig',data = middle_z_sig)
    
    if which_analysis == "peak" or which_analysis == "power-law" or which_analysis == "all":
        posterior.create_dataset('log_high_f_peak',data = log_high_f_peak)
        posterior.create_dataset('width_f_peak',data = width_f_peak)
        posterior.create_dataset('middle_z_f_peak',data = middle_z_f_peak)
        posterior.create_dataset('zp', data=zp)
        
    if which_analysis == "power-law" or which_analysis == "all":
        posterior.create_dataset('high_alpha',data = high_alpha)
        posterior.create_dataset('width_alpha',data = width_alpha)
        posterior.create_dataset('middle_z_alpha',data = middle_z_alpha)
        posterior.create_dataset('high_mMin',data = high_mMin)
        posterior.create_dataset('width_mMin',data = width_mMin)
        posterior.create_dataset('middle_z_mMin',data = middle_z_mMin)
        posterior.create_dataset('high_mMax',data = high_mMax)
        posterior.create_dataset('width_mMax',data = width_mMax)
        posterior.create_dataset('middle_z_mMax',data = middle_z_mMax)
        posterior.create_dataset('log_high_dmMax',data = log_high_dmMax)
        posterior.create_dataset('width_dmMax',data = width_dmMax)
        posterior.create_dataset('middle_z_dmMax',data = middle_z_dmMax)
        posterior.create_dataset('log_high_dmMin',data = log_high_dmMin)
        posterior.create_dataset('width_dmMin',data = width_dmMin)
        posterior.create_dataset('middle_z_dmMin',data = middle_z_dmMin)
        
    if which_analysis == "mass-variation":
        posterior.create_dataset('high_alpha_z',data = high_alpha_z)
        posterior.create_dataset('width_alpha_z',data = width_alpha_z)
        posterior.create_dataset('middle_m_alpha_z',data = middle_m_alpha_z)
        posterior.create_dataset('high_beta_z',data = high_beta_z)
        posterior.create_dataset('width_beta_z',data = width_beta_z)
        posterior.create_dataset('middle_m_beta_z',data = middle_m_beta_z)
        posterior.create_dataset('low_zp', data = low_zp)
        posterior.create_dataset('high_zp',data = high_zp)
        posterior.create_dataset('width_zp',data = width_zp)
        posterior.create_dataset('middle_m_zp',data = middle_m_zp)

    hfile.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-inputfile', help = "Input CDF file",action="store", type=str)
    parser.add_argument('-outputfile', help = "Output HDF5 file",action="store", type=str)
    parser.add_argument('-which_analysis', help = "Which analysis is converted?", action = "store", type = str)
    
    args = parser.parse_args()

    process(args.inputfile, args.outputfile, which_analysis = args.which_analysis)