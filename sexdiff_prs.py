#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep  9 18:01:18 2019

For a given phenotype:
1) Randomly remove 10k males, 10k females from UKB white British with phenotype defined.
2) Run GWAS in males with 10k removed, females with 10k removed for phenotype (using HM3 SNPs).
3) Run GWAS in combined set of males+females (with previously defined 10k males, 10k females removed)
4) Meta-analyze (male, female GWAS?) with MTAG, assuming rg=1
5) Meta-analyze with MTAG without assuming rg=1
6) Create PRS using P+T and each of the 3 GWAS results.
7) Compute difference in PRS-phenotype R2 using PRS from combined GWAS and PRS male GWAS, female GWAS


@author: nbaya
"""

import hail as hl
import numpy as np
from hail.utils.java import Env
import requests
import subprocess
import os
import pandas as pd
url = 'https://raw.githubusercontent.com/nikbaya/split/master/gwas.py'
r = requests.get(url).text
exec(r)
gwas=gwas


wd= 'gs://nbaya/sexdiff/prs/'

def get_mt(phen, variant_set='hm3'):
    mt0 = hl.read_matrix_table(f'gs://nbaya/split/ukb31063.{variant_set}_variants.gwas_samples_repart.mt')
    
    print(f'\n... Reading UKB phenotype "{phen_dict[phen][0]}" (code: {phen}) ...')
    phen_tb0 = hl.import_table('gs://ukb31063/ukb31063.PHESANT_January_2019.both_sexes.tsv.bgz',
                               missing='',impute=True,types={'s': hl.tstr}, key='s')
    phen_tb = phen_tb0.select(phen).rename({phen:'phen'})

    mt1 = mt0.annotate_cols(phen_str = hl.str(phen_tb[mt0.s]['phen']).replace('\"',''))
    mt1 = mt1.filter_cols(mt1.phen_str == '',keep=False)

    if phen_tb.phen.dtype == hl.dtype('bool'):
        mt1 = mt1.annotate_cols(phen = hl.bool(mt1.phen_str)).drop('phen_str')
    else:
        mt1 = mt1.annotate_cols(phen = hl.float64(mt1.phen_str)).drop('phen_str')

    #Remove withdrawn samples
    withdrawn = hl.import_table('gs://nbaya/w31063_20181016.csv',missing='',no_header=True)
    withdrawn_set = set(withdrawn.f0.take(withdrawn.count()))
    mt1 = mt1.filter_cols(hl.literal(withdrawn_set).contains(mt1['s']),keep=False)
    mt1 = mt1.key_cols_by('s')
    
    return mt1

def remove_n_individuals(mt, n_remove_per_sex, phen, sexes = 'fm', seed=None):
    r'''
    Removes n_remove_per_sex individuals from each specified sex (default is to remove from both
    females and males, i.e. sexes='fm').
    '''
    assert 'f' in sexes or 'm' in sexes, "sexes must have 'f' or 'm'"
    n_remove_per_sex = int(n_remove_per_sex)
    print(f'\n... Removing {n_remove_per_sex} per sex from UKB phenotype ({phen}) ...\n')
    seed = seed if seed is not None else int(str(Env.next_seed())[:5])
    print(f'\n... seed: {seed} ...\n')
    mt_cols = mt.cols()
    initial_ct = mt_cols.count()
    print(f'\n\n... Initial combined mt count ({phen}): {initial_ct} ...\n')
    tb_sex_ls = [None, None]
    for idx, sex in enumerate(sexes):
        filter_arg = mt_cols.isFemale if sex=='f' else (~mt_cols.isFemale if sex=='m' else None)
        mt_cols_sex = mt_cols.filter(filter_arg) 
        mt_cols_sex_ct = mt_cols_sex.count()
        print(f'\n\n... Initial {sex} mt count ({phen}): {mt_cols_sex_ct} ...\n')
        n_sex = mt_cols_sex.count()
        tb1 = mt_cols_sex.add_index('idx_tmp')
        tb2 = tb1.key_by('idx_tmp')
        remove = [1]*(n_remove_per_sex)+[0]*(n_sex-n_remove_per_sex)
        randstate = np.random.RandomState(int(seed)) 
        randstate.shuffle(remove)
        tb3 = tb2.annotate(remove = hl.literal(remove)[hl.int32(tb2.idx_tmp)])
        tb4 = tb3.filter(tb3.remove == 1, keep=True) #remove samples we wish to discard from original mt
        tb5 = tb4.key_by('s')
        tb_sex_ls[idx] = tb5
    mt_both = mt
    for tb_sex in tb_sex_ls:
        mt_both = mt_both.anti_join_cols(tb_sex)
    mt_f = mt.filter_cols(mt.isFemale).anti_join_cols(tb_sex_ls[sexes.index('f')]) if 'f' in sexes else None
    mt_m = mt.filter_cols(~mt.isFemale).anti_join_cols(tb_sex_ls[sexes.index('m')]) if 'm' in sexes else None
    mt_sex_ls = [mt_f, mt_m]
    mt_both_ct = mt_both.count_cols()
    print(f'\n\n... Final combined mt count ({phen}): {mt_both_ct} ...\n')
    for idx, sex in enumerate('fm'):
        if sex in sexes:
            mt_sex_ct = mt_sex_ls[idx].count_cols()
            print(f'\n... Final {sex} mt count ({phen}): {mt_sex_ct} ...\n')
    return mt_both, mt_f, mt_m, seed
    
def gwas(mt, x, y, cov_list=[], with_intercept=True, pass_through=[], path_to_save=None, 
         normalize_x=False, is_std_cov_list=False):
    r'''Runs GWAS'''
    
    mt = mt._annotate_all(col_exprs={'__y':y},
                           entry_exprs={'__x':x})
    
    print('... Calculating allele frequency ...')
    mt_freq_rows = mt.annotate_rows(freq = hl.agg.mean(mt.dosage)/2).rows() #frequency of alternate allele
    mt_freq_rows = mt_freq_rows.key_by('rsid')
    
    if normalize_x:
        mt = mt.annotate_rows(__gt_stats = hl.agg.stats(mt.__x))
        mt = mt.annotate_entries(__x= (mt.__x-mt.__gt_stats.mean)/mt.__gt_stats.stdev) 
        mt = mt.drop('__gt_stats')
    
    if is_std_cov_list:
        cov_list = ['isFemale','age','age_squared','age_isFemale',
                    'age_squared_isFemale']+['PC{:}'.format(i) for i in range(1, 21)]
        
    if str in list(map(lambda x: type(x),cov_list)):
        cov_list = list(map(lambda x: mt[x] if type(x) is str else x,cov_list))
        
    cov_list = ([1] if with_intercept else [])+cov_list
    
    print(f'pass through: {pass_through}')

    gwas_ht = hl.linear_regression_rows(y=mt.__y,
                                        x=mt.__x,
                                        covariates=cov_list,
                                        pass_through = ['rsid']+pass_through)
    
    gwas_ht = gwas_ht.annotate_globals(with_intercept = with_intercept)
        
    gwas_ht = gwas_ht.key_by('rsid')
    
    ss_template = hl.read_table('gs://nbaya/rg_sex/hm3.sumstats_template.ht') # sumstats template as a hail table
    ss_template = ss_template.key_by('SNP')
        
    ss = ss_template.annotate(chr = gwas_ht[ss_template.SNP].locus.contig,
                              bpos = gwas_ht[ss_template.SNP].locus.position,
                              freq = mt_freq_rows[ss_template.SNP].freq,
                              beta = gwas_ht[ss_template.SNP].beta,
                              z = gwas_ht[ss_template.SNP].t_stat,
                              pval = gwas_ht[ss_template.SNP].p_value,
                              n = gwas_ht[ss_template.SNP].n)
    ss = ss.drop('N')
    ss = ss.rename({'SNP':'snpid',
                    'A1':'a1',
                    'A2':'a2'})
    
    print(ss.describe())
    
    
    if path_to_save is not None:
        ss.export(path_to_save)
        
    return ss

def get_freq(mt, sex, n_remove, seed):
    r'''
    Get allele frequencies and other SNP information (needed to fix previously 
    created sumstats files)
    '''
    
    print('... Calculating allele frequency ...')
    mt = mt.annotate_rows(freq = hl.agg.mean(mt.dosage)/2) #frequency of alternate allele
    mt_rows = mt.rows()
    mt_rows = mt_rows.key_by('rsid')
    mt_rows = mt_rows.annotate(chr = mt_rows.locus.contig,
                               bpos = mt_rows.locus.position)

    
    ss = hl.import_table(wd+f'{phen}.gwas.{sex}.n_remove_{n_remove_per_sex}.seed_{seed}.old.tsv.bgz',
                         impute=True,
                         key='SNP')
    
    ss = ss.annotate(chr = mt_rows[ss.SNP].chr,
                     bpos = mt_rows[ss.SNP].bpos,
                     freq = mt_rows[ss.SNP].freq,
                     z = ((-1)*(ss.beta<0)*hl.abs(hl.qnorm(ss.p_value/2))+
                          (ss.beta>0)*hl.abs(hl.qnorm(ss.p_value/2)))
                     )
                     

    if 'N' in ss.row:
        if 'n' not in ss.row:
            ss = ss.annotate(n = ss.N)
        ss = ss.drop('N')
        
    ss = ss.rename({'SNP':'snpid',
                    'A1':'a1',
                    'A2':'a2',
                    'p_value':'pval'})
    
    ss = ss.key_by()
    ss = ss.select('snpid','chr','bpos','a1','a2','freq','beta','z','pval','n')
    ss = ss.key_by('snpid')
    
    ss.export(wd+f'{phen}.gwas.{sex}.n_remove_{n_remove_per_sex}.seed_{seed}.tsv.bgz')
    
def get_freq_alt(mt, sex, n_remove, seed):
    
    ss = hl.import_table(wd+f'{phen}.gwas.{sex}.n_remove_{n_remove_per_sex}.seed_{seed}.tsv.bgz',
                         impute=True,
                         key='snpid')
    ss = ss.annotate(z = ((-1)*(ss.beta<0)*hl.abs(hl.qnorm(ss.pval/2))+
                          (ss.beta>0)*hl.abs(hl.qnorm(ss.pval/2)))
                     )
    
    if 'N' in ss.row:
        if 'n' not in ss.row:
            ss = ss.annotate(n = ss.N)
        ss = ss.drop('N')
    
    ss = ss.key_by()
    ss = ss.select('snpid','chr','bpos','a1','a2','freq','beta','z','pval','n')
    ss = ss.key_by('snpid')
    
    ss.export(wd+f'{phen}.gwas.{sex}.n_remove_{n_remove_per_sex}.seed_{seed}.tsv.bgz')
        
def prs(mt, phen, sex, n_remove, prune, threshold, seed):
    r'''
    Calculate PRS using betas from both sexes and sex-stratified GWAS, as well
    as MTAG meta-analyzed betas.
    '''
    assert sex in ['both_sexes','female','male'], f'WARNING: sex={sex} not allowed. sex must be one of the following: both_sexes, female, male'
    threshold_str = '{:.4e}'.format(threshold)
    corr_path = (wd+f'corr.{phen}.{sex}.n_remove_{n_remove_per_sex}.seed_{seed}.\
                 {"" if prune else "not_"}pruned.threshold_{threshold_str}.tsv')
    try:
        subprocess.check_output([f'gsutil', 'ls', corr_path]) != None
    except:
        if prune:
            print('\n... Pruning SNPs ...')
            # define the set of SNPs
            pruned_snps_file = 'gs://nbaya/risk_gradients/ukb_imp_v3_pruned.bim' #from Robert Maier (pruning threshold=0.2, random 10k sample)
            variants = hl.import_table(pruned_snps_file, delimiter='\t', no_header=True, impute=True)
            print(f'\n... Pruning to variants in {pruned_snps_file}...')
            variants = variants.rename(
                {'f0': 'chr', 'f1': 'rsid', 'f3': 'pos'}).key_by('rsid')
            mt = mt.key_rows_by('rsid')
            # filter to variants defined in variants table
            mt = mt.filter_rows(hl.is_defined(variants[mt.rsid]))
            ct_rows = mt.count_rows()
            print(f'\n... row count after pruning filter: {ct_rows} ...\n')
        else:
            print(f'\n... Not pruning because prune={prune} ...\n')
            
        # "def" uses the MTAG results created by using the default settings
        # "rg1" uses the MTAG results created by using the --perfect-gencov flag
        gwas_versions = ['unadjusted',f'mtag_{"rg1" if sex is "both_sexes" else "def"}'] 
        
        r_ls, gwas_version_ls = [], []
        
        for gwas_version in gwas_versions:
            print(f'\n... Calculating PRS-phenotype R for "{phen_dict[phen][0]}" {sex} {gwas_version} ...')
            gwas_version_suffix = "" if gwas_version=='unadjusted' else '.'+gwas_version
            gwas_path = (wd+f'{phen}.gwas.{sex}.n_remove_{n_remove_per_sex}.seed_{seed}{gwas_version_suffix}.tsv.bgz')
    
            ss=hl.import_table(gwas_path,impute=True,key='snpid' if gwas_version is 'unadjusted' else 'SNP')
            
            ss = ss.filter(ss.pval<threshold)
            
            mt = mt.annotate(beta = ss[mt.rsid]['beta' if gwas_version is 'unadjusted' else 'mtag_beta'])
            
            mt = mt.aggregate_cols(prs = hl.agg.sum(mt.dosage*mt.beta))
            
            mt_cols = mt.cols()
            
            r = mt_cols.aggregate(hl.agg.corr(mt_cols.phen, mt_cols.prs))
            print(f'\n\n... PRS-phenotype R for "{phen_dict[phen][0]}" {sex} {gwas_version}\
                                                  GWAS ({"" if prune else "not_"}pruned,\
                                                  pval<{threshold_str}) ...\nR = {r}\
                                                  \nR^2 = {r**2}\n\n')
            r_ls.append(r)
            gwas_version_ls.append(gwas_version)
            
        df = pd.DataFrame(data=list(zip([phen]*len(r_ls), [sex]*len(r_ls), gwas_version_ls, r_ls)),
                              columns=['phen', 'sex', 'gwas_version', 'r'])
        print(df)
    
        hl.Table.from_pandas(df).export(corr_path)



if __name__ == "__main__":
    phen_dict = {
                '50_irnt':['Standing height', 73178],
                '23105_irnt':['Basal metabolic rate', 35705],
#                '23106_irnt':['Impedance of the whole body', 73701],
#                '2217_irnt':['Age started wearing contact lenses or glasses', 73178],
            '23127_irnt':['Trunk fat percentage', 73178]
#                '1100':['Drive faster than motorway speed limit', 73178],
#                '1757':['Facial ageing', 35705],
#            '6159_3':['Pain type(s) experienced in last month: Neck or shoulder',None],
#            '894':['Duration of moderate activity',None],
#            '1598':['Average weekly spirits intake',None]
            }
    
    n_remove_per_sex = 10e3
    
    for phen, phen_desc in phen_dict.items():

        mt = get_mt(phen)
        mt_both, mt_f, mt_m, seed = remove_n_individuals(mt=mt, n_remove_per_sex=n_remove_per_sex, 
                                                         phen=phen,sexes = 'fm', seed=phen_dict[phen][1])
        for mt_tmp, sex in [(mt_f,'female'), (mt_m,'male'), (mt_both,'both_sexes')]:            
            path = wd+f'{phen}.gwas.{sex}.n_remove_{n_remove_per_sex}.seed_{seed}.tsv.bgz'
            try:
                subprocess.check_output([f'gsutil', 'ls', path]) != None
                print(f'\n... "{phen_dict[phen][0]}" GWAS of {sex} already complete! ...\n')
                prs(mt=mt_tmp,phen=phen,sex=sex,n_remove=n_remove_per_sex,prune=True,threshold=1,seed=seed)
            except:
                old_path = wd+f'{phen}.gwas.{sex}.n_remove_{n_remove_per_sex}.seed_{seed}.old.tsv.bgz'
                try:
                    subprocess.check_output([f'gsutil', 'ls', old_path]) != None
                    print(f'\n... "{phen_dict[phen][0]}" GWAS of {sex} already complete but needs more fields! ...\n')
                    get_freq(mt=mt_tmp, sex=sex, n_remove=n_remove_per_sex, seed=seed)
                except:
                    print(f'\n... Running {sex} GWAS on "{phen_dict[phen][0]}" (code: {phen}) ...\n')
                    gwas(mt=mt_tmp, 
                         x=mt_tmp.dosage, 
                         y=mt_tmp['phen'], 
                         path_to_save=path,
                         is_std_cov_list=True)
#            
