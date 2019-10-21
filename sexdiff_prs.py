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
import subprocess
import pandas as pd
from scipy import stats
from datetime import datetime as dt
#import requests
#url = 'https://raw.githubusercontent.com/nikbaya/split/master/gwas.py'
#r = requests.get(url).text
#exec(r)
#gwas=gwas
hl.init(log='/tmp/hail.log')


wd= 'gs://nbaya/sexdiff/prs/'

def get_mt(mt0, phen_tb0, phen):
    print(f'\n... Reading UKB phenotype "{phen_dict[phen][0]}" (code: {phen}) ...')

    phen_tb = phen_tb0.select(phen).rename({phen:'phen'})

    mt1 = mt0.annotate_cols(phen_str = hl.str(phen_tb[mt0.s]['phen']).replace('\"',''))
    mt1 = mt1.filter_cols(mt1.phen_str == '',keep=False)

    if phen_tb.phen.dtype == hl.dtype('bool'):
        mt1 = mt1.annotate_cols(phen = hl.bool(mt1.phen_str)).drop('phen_str')
    else:
        mt1 = mt1.annotate_cols(phen = hl.float64(mt1.phen_str)).drop('phen_str')
    
    return mt1

def remove_n_individuals(mt, n_remove_per_sex, phen, sexes = 'fm', seed=None):
    r'''
    Removes n_remove_per_sex individuals from each specified sex (default is to remove from both
    females and males, i.e. sexes='fm').
    Saves tables with phenotype data for each sex and the sexes combined.
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
        tb_sex_path = wd+f'{phen}.{sex}.n_remove_{n_remove_per_sex}.seed_{seed}.ht'
        try:
            subprocess.check_output([f'gsutil', 'ls', tb_sex_path+'/_SUCCESS']) is not None
            print(f'\n... {phen} table for {sex} already written! ...\n')
        except:
            print(f'\n... Starting to write {phen} table for {sex} ...\n')
            filter_arg = mt_cols.isFemale if sex=='f' else (~mt_cols.isFemale if sex=='m' else None)
            mt_cols_sex = mt_cols.filter(filter_arg) 
            n_sex = mt_cols_sex.count()
            print(f'\n\n... Initial {sex} mt count ({phen}): {n_sex} ...\n')
            tb1 = mt_cols_sex.add_index('idx_tmp')
            tb2 = tb1.key_by('idx_tmp')
            remove = [1]*(n_remove_per_sex)+[0]*(n_sex-n_remove_per_sex)
            randstate = np.random.RandomState(int(seed)) 
            randstate.shuffle(remove)
            tb3 = tb2.annotate(remove = hl.literal(remove)[hl.int32(tb2.idx_tmp)])
            tb4 = tb3.filter(tb3.remove == 1, keep=True) #keep samples we wish to discard from original mt
    #         tb4 = tb3.filter(tb3.remove == 1, keep=False) #remove samples we wish to discard from original mt
            tb5 = tb4.key_by('s')
            tb5.select('phen').write(tb_sex_path) # write out table with samples of single sex we wish to discard from original mt
        tb_sex_ls[idx] = hl.read_table(tb_sex_path)
    if len(set(sexes).union('fm')) == 2: #if sexes has both sexes
        tb_both_path = wd+f'{phen}.both_sexes.n_remove_{n_remove_per_sex}.seed_{seed}.ht'
        try:
            subprocess.check_output([f'gsutil', 'ls', tb_both_path+'/_SUCCESS']) is not None
            print(f'\n... {phen} table for both_sexes already written! ...\n')
        except:
            print(f'\n... Starting to write {phen} table for both_sexes ...\n')
            tb_both = mt_cols
            tb_both = tb_both.anti_join(tb_sex_ls[0])
            tb_both = tb_both.anti_join(tb_sex_ls[1])
            tb_both = mt_cols.anti_join(tb_both)
            tb_both.select('phen').write(tb_both_path) # write out table containing all m/f individuals we wish to discard from original mt
        tb_both = hl.read_table(tb_both_path)
        mt_both = mt.anti_join_cols(tb_both)
        mt_both_ct = mt_both.count_cols()
        print(f'\n\n... Final both_sexes mt count ({phen}): {mt_both_ct} ...\n')
    else: 
        mt_both=None
    mt_f = mt.filter_cols(mt.isFemale) if 'f' in sexes else None
    mt_f = mt_f.anti_join_cols(tb_sex_ls[sexes.index('f')]) if mt_f is not None else None
    mt_m = mt.filter_cols(~mt.isFemale) if 'm' in sexes else None
    mt_m = mt_m.anti_join_cols(tb_sex_ls[sexes.index('m')]) if mt_m is not None else None
    mt_sex_ls = [mt_f, mt_m]
    for idx, sex in enumerate('fm'):
        if sex in sexes:
            mt_sex_ct = mt_sex_ls[idx].count_cols()
            print(f'\n\n... Final {sex} mt count ({phen}): {mt_sex_ct} ...\n')

    return mt_both, mt_f, mt_m, seed
    
def gwas(mt, x, y, cov_list=[], with_intercept=True, pass_through=[], path_to_save=None, 
         normalize_x=False, is_std_cov_list=False):
    r'''Runs GWAS'''
    
    mt = mt._annotate_all(col_exprs={'__y':y},
                           entry_exprs={'__x':x})
    
    print('\n... Calculating allele frequency ...')
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
    
        
        
def prs(mt_all, phen, sex, n_remove, prune, percentiles, seed):
    r'''
    Calculate PRS using betas from both sexes and sex-stratified GWAS, as well
    as MTAG meta-analyzed betas.
    P-value thresholds are determined by percentile
    '''
    assert sex in ['both_sexes','female','male'], f'WARNING: sex={sex} not allowed. sex must be one of the following: both_sexes, female, male'

        
#    mt = get_test_mt(mt_all, sex=('fe' if sex=='male' else '')+'male', seed=seed)
#    mt = get_test_mt(mt_all, sex=sex, seed=seed)
    mt = get_test_mt(mt_all, sex='both_sexes', seed=seed)
    
    # "def" uses the MTAG results created by using the default settings
    # "rg1" uses the MTAG results created by using the --perfect-gencov flag
    gwas_versions = ['unadjusted',f'mtag_{"rg1" if sex is "both_sexes" else "def"}'] 
    
#        r_ls, rpval_ls, gwas_version_ls, percentile_ls, threshold_ls, snps_ls = [], [], [], [], [], []
    
    for gwas_version in gwas_versions:
        print(f'\n... Calculating PRS for "{phen_dict[phen][0]}" {sex} {gwas_version} ...\n')
        gwas_version_suffix = "" if gwas_version=='unadjusted' else '.'+gwas_version
        gwas_path = (wd+f'{phen}.gwas.{sex}.n_remove_{n_remove_per_sex}.seed_{seed}{gwas_version_suffix}.tsv.{"b" if gwas_version=="unadjusted" else ""}gz')

        ss=hl.import_table(gwas_path,
                           impute=True,
                           key='snpid' if gwas_version is 'unadjusted' else 'SNP',
                           force=True)
        
        if prune:
            print('\n... Pruning SNPs ...\n')
            # define the set of SNPs
            pruned_snps_file = 'gs://nbaya/risk_gradients/ukb_imp_v3_pruned.bim' #from Robert Maier (pruning threshold=0.2, random 10k UKB sample)
            variants = hl.import_table(pruned_snps_file, delimiter='\t', no_header=True, impute=True)
            print(f'\n... Pruning to variants in {pruned_snps_file} ...\n')
            variants = variants.rename(
                {'f0': 'chr', 'f1': 'rsid', 'f3': 'pos'}).key_by('rsid')
#            mt = mt.key_rows_by('rsid')
            # filter to variants defined in variants table
            ss = ss.filter(hl.is_defined(variants[ss['snpid' if gwas_version is 'unadjusted' else 'SNP']]))
            ct_rows = ss.count()
            print(f'\n\n... SNP count after pruning filter: {ct_rows} ...\n')
        else:
            print(f'\n... Not pruning because prune={prune} ...\n')
            
        
        for percentile in percentiles:

            prs_path_without_threshold = (wd+f'prs.{phen}.{sex}.n_remove_{int(n_remove_per_sex)}.seed_{seed}.{gwas_version}.{"" if prune else "not_"}pruned.pval_thresh_*.perc_{percentile}.tsv')
#            prs_path_without_threshold = (wd+f'prs.{phen}.{sex}.n_remove_{int(n_remove_per_sex)}.seed_{seed}.{gwas_version}.{"" if prune else "not_"}pruned.pval_thresh_*.perc_{percentile}.opposite_sex.tsv')

            try:
                subprocess.check_output(['gsutil', 'ls', prs_path_without_threshold]) != None
                print(f'\n\n... Calculation of PRS for "{phen_dict[phen][0]}" {sex} {gwas_version} for percentile {percentile} already completed! ...\n')
        
            except:
                start = dt.now()
                
                threshold = ss.aggregate(hl.agg.approx_quantiles(ss[('' if gwas_version is 'unadjusted' else 'mtag_')+'pval'],percentile))
                threshold_str = '{:.4e}'.format(threshold)
                prs_path = wd+f'prs.{phen}.{sex}.n_remove_{int(n_remove_per_sex)}.seed_{seed}.{gwas_version}.{"" if prune else "not_"}pruned.pval_thresh_{threshold_str}.perc_{percentile}.tsv'
#                prs_path = wd+f'prs.{phen}.{sex}.n_remove_{int(n_remove_per_sex)}.seed_{seed}.{gwas_version}.{"" if prune else "not_"}pruned.pval_thresh_{threshold_str}.perc_{percentile}.opposite_sex.tsv'

                print(f'\n\n... Using p-value threshold of {threshold} for percentile {percentile} ...\n')
                
                ss = ss.filter(ss[('' if gwas_version is 'unadjusted' else 'mtag_')+'pval']<=threshold)
                                
                mt = mt.annotate_rows(beta = ss[mt.rsid]['beta' if gwas_version is 'unadjusted' else 'mtag_beta'])
    
                threshold_ct = mt.filter_rows(hl.is_defined(mt.beta)).count_rows()
                
                print(f'\n\n... Variants remaining after thresholding filter: {threshold_ct} ...\n')
                
                mt = mt.annotate_cols(prs = hl.agg.sum(mt.dosage*mt.beta))
                
                mt_cols_ct = mt.filter_cols(hl.is_defined(mt.prs)).count_cols()
                
                print(f'\n\n... Samples with PRS: {mt_cols_ct} ...\n')
    
                mt.cols().select('phen','prs').export(prs_path) #to_pandas()
                
                elapsed = dt.now()-start
                
                print(f'\n\n... Completed calculation of PRS for "{phen_dict[phen][0]}" {sex} {gwas_version} ...')
                print(f'\n... Elapsed time: {round(elapsed.seconds/60, 2)} min ...\n')

def get_test_mt(mt_all, sex, seed):
    assert sex in ['both_sexes','female','male'], f'WARNING: sex={sex} not allowed. sex must be one of the following: both_sexes, female, male'
    sex_dict = {'both_sexes':'both_sexes','female':'f','male':'m'}
    path = wd+f'{phen}.{sex_dict[sex]}.n_remove_{int(n_remove_per_sex)}.seed_{seed}.ht'
    ht_sex = hl.read_table(path)
    return mt_all.semi_join_cols(ht_sex)

def prs_phen_reg(mt_all, phen, sex, n_remove, prune, percentiles, seed):
    test_mt = get_test_mt(mt_all, 'both_sexes', seed=phen_dict[phen][1])
    test_ht = test_mt.cols()
    

    reg_path = wd+f'prs_phen_reg.{phen}.{sex}.n_remove_{int(n_remove_per_sex)}.seed_{seed}.{"" if prune else "not_"}pruned.tsv'

    
    try:
        subprocess.check_output(['gsutil', 'ls', reg_path]) != None
        print(f'... Phen ~ PRS + covariates regression already complete for all gwas versions & percentiles of {phen} {sex} ! ...')
    except:
        
        row_struct_ls = []
#        gwas_version_ls = []
#        percentile_ls = []
#        pval_thresh_ls = []
#        test_sex_ls = []
#        r2_ls = []
#        r2_pval_ls = []
        
        gwas_versions = ['unadjusted', f'mtag_{"def" if sex!="both_sexes" else "rg1"}']
        
        for gwas_version in gwas_versions:
            for percentile in percentiles:
                prs_path_without_threshold = (wd+f'prs.{phen}.{sex}.n_remove_{int(n_remove_per_sex)}.seed_{seed}.{gwas_version}.{"" if prune else "not_"}pruned*.perc_{percentile}{".comb" if sex!="both_sexes" else ""}.tsv')
                print(prs_path_without_threshold)
                process = subprocess.Popen(['gsutil','ls',prs_path_without_threshold], stdout=subprocess.PIPE)
                stdout, stderr = process.communicate()
                prs_path = stdout.decode('ascii').splitlines()[0]
                if sex=='both_sexes':
                    path_w_thresh = prs_path
                else:
                    original_prs_path = wd+f'prs.{phen}.{sex}.n_remove_{int(n_remove_per_sex)}.seed_{seed}.{gwas_version}.{"" if prune else "not_"}pruned.pval*.perc_{percentile}.tsv'
                    process = subprocess.Popen(['gsutil','ls',original_prs_path], stdout=subprocess.PIPE)
                    stdout, stderr = process.communicate()
                    path_w_thresh = stdout.decode('ascii').splitlines()[0]
                pval_thresh = path_w_thresh.split('pval_thresh_')[1].split('.perc')[0] #previously used for the both_sexes prs
                print(f'... {phen} {sex} {gwas_version} percentile={percentile} ...')
                print(f'... using {prs_path} ...')
                print(f'... pval threshold: {pval_thresh} ...')
                prs_ht = hl.import_table(prs_path,impute=True, key='s',types={'s': hl.tstr})
                test_ht = test_ht.annotate(prs = prs_ht[test_ht.s].prs)
                
                cov_list = ['prs','age','age_squared']+['PC{:}'.format(i) for i in range(1, 21)]
                for isFemale in [0,1]:
                    test_ht_sex = test_ht.filter(test_ht.isFemale==isFemale)
                    reg = test_ht_sex.aggregate(hl.agg.linreg(y=test_ht_sex.phen, 
                                                              x=[1]+list(map(lambda x: test_ht_sex[x] if type(x) is str else x,cov_list))))
                    print(f'\n\n... {phen} {sex} {gwas_version} percentile={percentile} applied to {"fe" if isFemale else ""}males ...\n'+
                          f'\n... multiple R^2: {reg.multiple_r_squared} ...'+
                          f'\n... pval for multiple R^2: {reg.multiple_p_value} ...'+
                          f'\n... adjusted R^2: {reg.adjusted_r_squared} ...')
                    row_struct_ls.append({'phen':phen, 'gwas_sex':sex, 'gwas_version':gwas_version,
                                          'percentile':str(percentile), 'pval_threshold':pval_thresh,
                                          'sex_tested_on':f'{"fe" if isFemale else ""}males',
                                          'multiple_r2':str(reg.multiple_r_squared),
                                          'multiple_r2_pval':str(reg.multiple_p_value),
                                          'adjusted_r2':str(reg.adjusted_r_squared)})
#                    percentile_ls.append(percentile)
#                    pval_thresh_ls.append(pval_thresh)
#                    r2_ls.append(reg.multiple_r_squared)
#                    r2_pval_ls.append(reg.multiple_p_value)
                
#        df = pd.DataFrame(data=list(zip([phen]*len(r2_ls), [sex]*len(r2_ls), 
#                                gwas_version_ls, percentile_ls, pval_thresh_ls, 
#                                test_sex_ls,r2_ls, r2_pval_ls)),
#                      columns=['phen', 'gwas_sex', 'gwas_version','percentile',
#                               'pval_threshold', 'sex_tested_on','r2','r2_pval']) 
#        print(df)
        ht = hl.Table.parallelize(hl.literal(row_struct_ls, 'array<struct{phen: str, gwas_sex: str, gwas_version: str, percentile: str, pval_threshold: str, sex_tested_on: str, multiple_r2: str, multiple_r2_pval: str, adjusted_r2: str}>'))
        ht = ht.annotate(percentile = hl.float(ht.percentile),
                         pval_threshold = hl.float(ht.pval_threshold),
                         multiple_r2 = hl.float(ht.multiple_r2),
                         multiple_r2_pval = hl.float(ht.multiple_r2_pval),
                         adjusted_r2 = hl.float(ht.adjusted_r2))
        ht.show(12)
#        hl.Table.parallelize(hl.literal(row_ls))
    
#        hl.Table.from_pandas(df).export(reg_path)  
        ht.export(reg_path)  

def prs_corr(phen,sex,n_remove,prune,percentiles,seed):
    pass
#    for 
#    threshold_str = '{:.4e}'.format(threshold)
#    corr_path = (wd+f'corr.{phen}.{sex}.n_remove_{n_remove_per_sex}.seed_{seed}.{"" if prune else "not_"}pruned.threshold_{threshold_str}.tsv')
#    corr_path = (wd+f'corr.{phen}.{sex}.n_remove_{n_remove_per_sex}.seed_{seed}.{"" if prune else "not_"}pruned.tsv')
                
    #            mt_cols.select('phen','prs').export()
                
#                r, rpval = stats.pearsonr(df_cols.phen, df_cols.prs)
#                print(f'\n\n... PRS-phenotype R for "{phen_dict[phen][0]}" {sex} {gwas_version} GWAS ({"" if prune else "not_"}pruned, pval≤{threshold}) ...\nR: {r}\nR^2: {r**2}\npval: {rpval}\n\n')
#                gwas_version_ls.append(gwas_version)
#                r_ls.append(r)
#                rpval_ls.append(rpval)
#                percentile_ls.append(percentiles[threshold_idx])
#                threshold_ls.append(threshold)
#                snps_ls.append(threshold_ct)
#            
#        df = pd.DataFrame(data=list(zip([phen]*len(r_ls), [sex]*len(r_ls), 
#                                        gwas_version_ls, [prune]*len(r_ls),
#                                        percentile_ls, threshold_ls, snps_ls, 
#                                        r_ls, rpval_ls)),
#                              columns=['phen', 'sex', 'gwas_version','pruned',
#                                       'percentile','pval_threshold','n_snps', 
#                                       'r','r_pval'])
#        print(df)
#    
#        hl.Table.from_pandas(df).export(corr_path)
        




if __name__ == "__main__":
    phen_dict = {
#                        '50_irnt':['Standing height', 73178],
#                        '23105_irnt':['Basal metabolic rate', 35705],
#                        '23106_irnt':['Impedance of the whole body', 73701],
#                '2217_irnt':['Age started wearing contact lenses or glasses', 73178],
                        '23127_irnt':['Trunk fat percentage', 73178],
                '1100':['Drive faster than motorway speed limit', 73178],
                '1757':['Facial ageing', 35705],
#                        '6159_3':['Pain type(s) experienced in last month: Neck or shoulder',73178],
#                '894':['Duration of moderate activity',73178],
#                '1598':['Average weekly spirits intake',35705]
            }
    
    variant_set = 'hm3'
    n_remove_per_sex = 10e3
    prune=True
    percentiles = [1,0.50,0.10]
    
    mt0 = hl.read_matrix_table(f'gs://nbaya/split/ukb31063.{variant_set}_variants.gwas_samples_repart.mt')
    #Remove withdrawn samples
    withdrawn = hl.import_table('gs://nbaya/w31063_20181016.csv',missing='',no_header=True)
    withdrawn_set = set(withdrawn.f0.take(withdrawn.count()))
    mt0 = mt0.filter_cols(hl.literal(withdrawn_set).contains(mt0['s']),keep=False)
    mt0 = mt0.key_cols_by('s')
    
    phen_tb0 = hl.import_table('gs://ukb31063/ukb31063.PHESANT_January_2019.both_sexes.tsv.bgz',
                           missing='',impute=True,types={'s': hl.tstr}, key='s')    
    
    for phen, phen_desc in phen_dict.items():

        mt = get_mt(mt0=mt0, phen_tb0=phen_tb0, phen=phen)
#        mt_both, mt_f, mt_m, seed = remove_n_individuals(mt=mt, n_remove_per_sex=n_remove_per_sex, 
#                                                         phen=phen,sexes = 'fm', seed=phen_dict[phen][1])
#        for mt_tmp, sex in [(mt_f,'female'), (mt_m,'male'), (mt_both,'both_sexes')]:            
#            gwas_path = wd+f'{phen}.gwas.{sex}.n_remove_{n_remove_per_sex}.seed_{seed}.tsv.bgz'
#            try:
#                subprocess.check_output([f'gsutil', 'ls', gwas_path ]) != None
#            except:
#                old_path = wd+f'{phen}.gwas.{sex}.n_remove_{n_remove_per_sex}.seed_{seed}.old.tsv.bgz'
#                try:
#                    subprocess.check_output([f'gsutil', 'ls', old_path]) != None
#                except:
#                    print(f'\n... Running {sex} GWAS on "{phen_dict[phen][0]}" (code: {phen}) ...\n')
#                    gwas(mt=mt_tmp, 
#                         x=mt_tmp.dosage, 
#                         y=mt_tmp['phen'], 
#                         path_to_save=gwas_path ,
#                         is_std_cov_list=True)                    
#                print(f'\n... "{phen_dict[phen][0]}" GWAS of {sex} already complete but needs more fields! ...\n')
#                get_freq(mt=mt_tmp, sex=sex, n_remove=n_remove_per_sex, seed=seed)
#                
#            print(f'\n... "{phen_dict[phen][0]}" GWAS of {sex} already complete! ...\n')
#            
#

#        for sex in ['both_sexes']:
#            prs(mt_all=mt, phen=phen,sex=sex,n_remove=n_remove_per_sex,
#                prune=prune,percentiles=percentiles,seed=phen_dict[phen][1])
            

        for sex in ['both_sexes','male','female']:
            prs_phen_reg(mt_all=mt, phen=phen, sex=sex, n_remove=n_remove_per_sex, 
                         prune=prune, percentiles=percentiles, seed=phen_dict[phen][1])


#            prs_corr(phen=phen,sex=sex,n_remove=n_remove_per_sex,prune=prune,percentiles=percentiles,seed=seed)

#else:
#    ht = hl.Table.parallelize(hl.struct(phen = ['50_irnt']*2,
#                                        val = [0]*2))
