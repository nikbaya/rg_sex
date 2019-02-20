#! /usr/bin/env python

###########################################


########
#
# Genetic correlation analysis
# male vs. female GWAS of same trait in UKBB
#
####

# Setup:
#


# get arguments for phenotype file, parallelization
import argparse
parser = argparse.ArgumentParser(add_help=False)
parser.add_argument('--phen', type=str, required=True, help="phenotype")
parser.add_argument('--n_chunks', type=int, required=True, help="number of chunks used for meta-analysis")
parser.add_argument('--batch', type=int, required=True, help="which of the batches to run")
parser.add_argument('--variant_set', type=str, required=True, help="set of variants to use")
parser.add_argument('--no_intercept', type=str,required=False, help="whether to run ldsc with intercept (if running simulated data, set to True")
#parser.add_argument('--numphens', type=int, required=True, help="total number of phenotype split pairs to run")
#parser.add_argument('--parsplit', type=int, required=True, help="number of parallel batches")


args = parser.parse_args()
    
phen = args.phen
batch = args.batch
n_chunks = args.n_chunks
variant_set = args.variant_set
no_intercept = bool(args.no_intercept)



#phenfile = str(phen)+'.h2part.n'+str(n_chunks)+'_batch_'+str(batch)+'.tsv'                   ############ CHECK ############
#phenfile = 'qc_pos.'+str(phen)+'.h2part.n'+str(n_chunks)+'_batch_'+str(batch)+'.tsv'                   ############ CHECK ############
phenfile = variant_set+'.'+phen+'.h2part.nchunks'+str(n_chunks)+'_batch_'+str(batch)+'.tsv'                   ############ CHECK ############
#phenfile = 'hm3.sex_strat.h2part.nchunks2_batch_1.tsv'
#phenfile = 'blood_pressure_phens.sex_strat.h2part.tsv' #only used for blood pressure phens

# local working dir
wd = '/home/mtag'
# 
# directory of UKBB sumstat files
#gs_sumstat_dir = 'gs://ukb31063-mega-gwas/hail-0.1/ldsc-sumstats-tsvs' # OUTDATED
#gs_sumstat_dir = 'gs://ukb31063-mega-gwas/ldsc-export/sumstats-files'
gs_sumstat_dir = 'gs://nbaya/rg_sex' #added by Nik
gs_phenfile_bucket = 'gs://nbaya/rg_sex'
gs_ss_path1 = gs_phenfile_bucket + '/' + phenfile

#phen_summary = 'gs://nbaya/split/test/height.split.tsv' # for height
#phen_summary = 'gs://nbaya/split/test/phenotypes.split.tsv'              ############ CHECK ############


# where to save 
#out_bucket = 'gs://ukb31063-ldsc-results/rg_sex/batches' # OUTDATED
out_bucket = 'gs://nbaya/rg_sex/batches'
#out_bucket = 'gs://nbaya/rg_sex/allphens'
#out_bucket = 'gs://nbaya/rg_sex/sex_strat'
#

# parallelization
num_proc = 6
# 
# 
# compute parallel split
#idx = range(args.batch-1, args.numphens, args.parsplit)
idx = range(100) # default: 100
#idx = range(1)
#rg_outname = 'ukbb31063.rg_sex.'+phen+'.n'+str(n_chunks)+'.batch_'+str(batch)+'.tsv.gz' # used for hm3 variants
rg_outname = 'ukbb31063.rg_sex.'+variant_set+'.'+phen+'.nchunks'+str(n_chunks)+'.batch_'+str(batch)+('.no_intercept' if no_intercept else '')+'.int_gcov_0.tsv.gz'
#rg_outname = 'ukbb31063.rg_sex.'+variant_set+'.'+phen+'.tsv.gz' #used only for blood pressure phens

########

print('\n####################')
print('Phenotype: '+phen)
print('Variant Set: '+variant_set)
print('Batch: '+str(batch))
print('no_intercept: '+str(no_intercept))
print('####################')

###
# load packages
print("Loading packages...")
###

import numpy as np
import pandas as pd
import os
import sys
import subprocess
import itertools
import datetime
from io import StringIO
from argparse import Namespace
from scipy import stats
from multiprocessing import Process, Pool
from functools import partial


###
# Install packages (MTAG, fix google-compute-engine)
print("Installing requirements...")
print('Time: {:%H:%M:%S (%Y-%b-%d)}'.format(datetime.datetime.now()))
###


# Install joblib (for MTAG)
subprocess.call(['/opt/conda/bin/pip','install','joblib'])

# for ldsc
subprocess.call(['/opt/conda/bin/pip','install','bitarray'])

# Download MTAG for ref panel, logger
if not os.path.isdir(wd+'/mtag-master'):
    subprocess.call(['wget', '--quiet', '-P', wd+'/', 'https://github.com/omeed-maghzian/mtag/archive/master.zip'])
    subprocess.call(['unzip', '-q', wd+'/master.zip', '-d', wd])
    subprocess.call(['rm',wd+'/master.zip'])

# Download ldsc
if not os.path.isdir(wd+'/ldsc-master'):
    subprocess.call(['wget', '--quiet', '-P', wd+'/', 'https://github.com/bulik/ldsc/archive/master.zip'])
    subprocess.call(['unzip', '-q', wd+'/master.zip', '-d', wd])    
    subprocess.call(['rm',wd+'/master.zip'])
    
# load MTAG and ldsc
sys.path.insert(0, wd+'/ldsc-master/')
sys.path.insert(0, wd+'/mtag-master/')
import ldscore.ldscore as ld
import ldscore.sumstats as sumstats
from mtag import Logger_to_Logging

# reference path
ld_ref_panel = wd+'/mtag-master/ld_ref_panel/eur_w_ld_chr/'


loc_path1 = wd + '/'
subprocess.call(['gsutil','cp',gs_ss_path1,loc_path1])
#subprocess.call(['ls',wd])



#####
# Prep sumstat files
#####

print("Preparing files...")
print('Time: {:%H:%M:%S (%Y-%b-%d)}'.format(datetime.datetime.now()))

### Get list of UKBB sumstats

# read phenotype file
print(wd+'/'+phenfile)
phens = pd.read_table(wd+'/'+phenfile)

# split to chunks
ph_sub = phens.loc[idx, :]


#####
# core function to run rg
#   on list of ukbb phenotypes
print("Preparing rg function...")
print('Time: {:%H:%M:%S (%Y-%b-%d)}'.format(datetime.datetime.now()))
#####

def ldsc_rg_pair(args, **kwargs):
    """
    Args is a list with elements:
    - args[0] = phenotype name
    - args[1] = phenotype description
    - args[2] = file name for phenotype 1
    - args[3] = file name for phenotype 2
    - args[4] = N for phenotype 1
    - args[5] = N_cases for phenotype 1
    - args[6] = N_controls for phenotype 1
    - args[7] = N for phenotype 2
    - args[8] = N_cases for phenotype 2
    - args[9] = N_controls for phenotype 2
    
    Assumes keyword args for:
    wd
    gs_sumstat_dir
    ld_ref_panel
    """
    
    # handle args
    phname = str(args[0])
    phdesc = str(args[1])
    f1 = str(args[2])
    f2 = str(args[3])
    n1 = int(args[4])
    ncas1 = float(args[5])
    ncon1 = float(args[6])
    n2 = int(args[7])
    ncas2 = float(args[8])
    ncon2 = float(args[9])
    
    # log
    print("Starting phenotype: "+str(phname))
    
    # download sumstats for phens
    gs_ss_path1 = gs_sumstat_dir + '/' + str(f1)
    loc_ss_path1 = wd + '/' + str(f1)
    subprocess.call(['gsutil','cp',gs_ss_path1,loc_ss_path1])
    
    gs_ss_path2 = gs_sumstat_dir + '/' + str(f2)
    loc_ss_path2 = wd + '/' + str(f2)
    subprocess.call(['gsutil','cp',gs_ss_path2,loc_ss_path2])  

    # list of files
    rg_file_list = ','.join([loc_ss_path1,loc_ss_path2])

    # list of names
    rg_name_list = [str(f1), str(f2)]

    # dummy output name
    rg_out = wd + '/' + 'rg.summary'

    # args for ldsc
    args_ldsc_rg =  Namespace(out=rg_out, 
                              bfile=None,
                              l2=None,
                              extract=None,
                              keep=None,
                              ld_wind_snps=None,
                              ld_wind_kb=None,
                              ld_wind_cm=None,
                              print_snps=None,
                              annot=None,
                              thin_annot=False,
                              cts_bin=None,
                              cts_break=None,
                              cts_names=None,
                              per_allele=False,
                              pq_exp=None,
                              no_print_annot=False,
                              maf=None,
                              h2=None,
                              rg=rg_file_list,
                              ref_ld=None,
                              ref_ld_chr=ld_ref_panel,
                              w_ld=None,
                              w_ld_chr=ld_ref_panel,
                              overlap_annot=False,
                              no_intercept=no_intercept,                               ###### CHECK WHEN RUNNING SIMULATED DATA ######
                              intercept_h2=None, 
                              intercept_gencov='0,0',#None,
                              M=None,
                              two_step=None,# if no_intercept else 99999,                  ###### CHECK WHEN RUNNING SIMULATED DATA, should be None ######
                              chisq_max=99999,
                              print_cov=False,
                              print_delete_vals=False,
                              chunk_size=50,
                              pickle=False,
                              invert_anyway=False,
                              yes_really=False,
                              n_blocks=200,
                              not_M_5_50=False,
                              return_silly_things=False,
                              no_check_alleles=False,
                              print_coefficients=False,
                              samp_prev=None,
                              pop_prev=None,
                              frqfile=None,
                              h2_cts=None,
                              frqfile_chr=None,
                              print_all_cts=False)

    # run rg
    rg_out = sumstats.estimate_rg(args_ldsc_rg, Logger_to_Logging())

    
    # get basic rg summary table
    rg_tab_txt = sumstats._get_rg_table(rg_name_list, rg_out, args_ldsc_rg)
    rg_df = pd.read_csv(StringIO(rg_tab_txt), delim_whitespace=True)
    
    # rename h2, int columns so we can add h2/int for phenotype 1
    rg_df.rename({
            'h2_obs' : 'ph2_h2_obs',
            'h2_obs_se' : 'ph2_h2_obs_se',
            'h2_int' : 'ph2_h2_int',
            'h2_int_se' : 'ph2_h2_int_se'
        }, axis='columns', inplace=True)
    
    # add h2/int for phenotype 1
    t = lambda attr: lambda obj: getattr(obj, attr, 'NA')
    rg_df['ph1_h2_int'] = map(t('intercept'), map(t('hsq1'), rg_out))
    rg_df['ph1_h2_int_se'] = map(t('intercept_se'), map(t('hsq1'), rg_out))
    rg_df['ph1_h2_obs'] = map(t('tot'), map(t('hsq1'), rg_out))
    rg_df['ph1_h2_obs_se']= map(t('tot_se'), map(t('hsq1'), rg_out))
    
    # add phenotype info 
    rg_df.insert(0,'description',str(phdesc))
    rg_df.insert(0,'phenotype',str(phname))
    
    # add sample size info
    rg_df['ph1_n'] = n1
    rg_df['ph1_n_case'] = ncas1
    rg_df['ph1_n_control'] = ncon1
    rg_df['ph2_n'] = n2
    rg_df['ph2_n_case'] = ncas2
    rg_df['ph2_n_control'] = ncon2
    
    return rg_df

# zip arguments to map
iter_args = itertools.izip(ph_sub['phen'], 
                           ph_sub['desc'], 
                           ph_sub['female_file'],
                           ph_sub['male_file'],
                           ph_sub['female_n'],
                           ph_sub['female_n_cas'],
                           ph_sub['female_n_con'],
                           ph_sub['male_n'],
                           ph_sub['male_n_cas'],
                           ph_sub['male_n_con'])

 # bake in globals
ldsc_rg_map = partial(ldsc_rg_pair,
                      wd=wd,
                      gs_sumstat_dir=gs_sumstat_dir,
                      ld_ref_panel=ld_ref_panel)


# dispatch rg
print("Starting ldsc...")
print('Time: {:%H:%M:%S (%Y-%b-%d)}'.format(datetime.datetime.now()))
pool = Pool(num_proc)
results = pool.imap_unordered(ldsc_rg_map, iter_args)
#results = pool.imap(ldsc_rg_map, iter_args)
pool.close()
pool.join()

# collect results
print("Processing results...")
print('Time: {:%H:%M:%S (%Y-%b-%d)}'.format(datetime.datetime.now()))

dat = pd.concat(results)

####
# write results to file
print("Saving...")
print('Time: {:%H:%M:%S (%Y-%b-%d)}'.format(datetime.datetime.now()))
####
rg_local = wd+'/'+rg_outname
rg_cloud = out_bucket+'/'+rg_outname

# write local
dat.to_csv(rg_local, sep='\t', compression='gzip', index=False, na_rep="NA")

# move to cloud
subprocess.call(['gsutil','cp','file://'+rg_local,rg_cloud])

print("################")
print("Finished!!")
print('Time: {:%H:%M:%S (%Y-%b-%d)}'.format(datetime.datetime.now()))
print("Results output to:")
print(str(rg_cloud))
print("################")

# eof