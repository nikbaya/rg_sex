#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb  4 14:50:10 2019

Run blood pressure gwas.

@author: nbaya
"""

import hail as hl
import datetime


output_bucket = 'gs://nbaya/rg_sex/sex_strat/'

meds = ['"20003_1141194794"','"20003_1141146234"','"20003_1140866738"','"20003_1140879802"','"20003_1140860806"',
        '"20003_1140860696"','"20003_1140861958"','"6177_2"','"6153_2"']



phen_tb_all = hl.import_table('gs://phenotype_31063/ukb31063.phesant_phenotypes.both_sexes.tsv.bgz',
                                          missing='',impute=True,types={'"userId"': hl.tstr}).rename({ '"userId"': 's'})
withdrawn = hl.import_table('gs://nbaya/w31063_20181016.csv',missing='',no_header=True)
withdrawn_set = set(withdrawn.f0.take(withdrawn.count()))
phen_tb_all = phen_tb_all.filter(hl.literal(withdrawn_set).contains(phen_tb_all['s']),keep=False) 
phen_tb_all = phen_tb_all.key_by('s')


## Filter phenotype table

#tb2 = phen_tb_all
#for phen in meds[:-2]:
#        tb2 = tb2.filter(tb2[phen] == 0)
#        print(phen+': '+str(tb2.count()))
#        
#tb3 = tb2
#for phen in meds[::-1][:2]:
#    with_meds = tb3.filter(tb3[phen] == 1)
#    ids = with_meds['s'].take(with_meds.count())
#    if len(ids) > 0:
#            tb3 = tb3.filter(hl.literal(ids).contains(tb3['s']),keep=False)
#    print(phen+': '+str(tb3.count()))
#    
#tb3.write('gs://nbaya/rg_sex/sex_strat/no_bp_phens.ht')


## Run GWAS

tb = hl.read_table('gs://nbaya/rg_sex/sex_strat/no_bp_phens.ht')
gwas_phens = ['4079','4080']

variants = hl.read_matrix_table('gs://nbaya/split/ukb31063.hm3_variants.gwas_samples_repart.mt') #hm3 variants matrix table

for phen in gwas_phens:
    print('####################')
    print('Starting phen '+phen)
    print('####################')
    phen_starttime = datetime.datetime.now()
    
    phen_tb = tb.select(tb['"'+phen+'"']).rename({'"'+phen+'"': 'phen'})
    mt1 = variants.annotate_cols(phen_str = hl.str(phen_tb[variants.s]['phen']).replace('\"',''))
    mt1 = mt1.filter_cols(mt1.phen_str == '',keep=False)
    if phen_tb.phen.dtype == hl.dtype('bool'):
        mt1 = mt1.annotate_cols(phen = hl.bool(mt1.phen_str)).drop('phen_str')
    else:
        mt1 = mt1.annotate_cols(phen = hl.float64(mt1.phen_str)).drop('phen_str')
    
    for sex in ['female','male']:
        mt2 = mt1.filter_cols(mt1.isFemale == (sex == 'female'))
    
        mt3 = mt2.add_col_index()
        mt3 = mt3.rename({'dosage': 'x', 'phen': 'y'})
    
        n_samples = mt3.count_cols()
        print('\n>>> phen '+sex+' '+phen+': N samples = '+str(n_samples)+' <<<') 
        
        mt = mt3
        
        cov_list = [ mt['isFemale'], mt['age'], mt['age_squared'], mt['age_isFemale'],
                    mt['age_squared_isFemale'] ]+ [mt['PC{:}'.format(i)] for i in range(1, 21)]
        
        ht = hl.linear_regression_rows(
                    y=mt.y,
                    x=mt.x,
                    covariates=[1]+cov_list,
                    pass_through = ['rsid'])
        
        ht = ht.rename({'rsid':'SNP'}).key_by('SNP')
        ht = ht.select(Z = ht.beta/ht.standard_error)
        
        sumstats_template = hl.import_table('gs://nbaya/rg_sex/50_snps_alleles_N.tsv.gz',types={'N': hl.tint64})
        sumstats_template = sumstats_template.key_by('SNP')
        sumstats_template = sumstats_template.annotate(N = int(n_samples/2))

        sumstats = sumstats_template.annotate(Z = ht[sumstats_template.SNP]['Z'])
        
        sumstats_path = output_bucket+phen+'_'+sex+'.tsv.bgz' 
        
        sumstats.export(sumstats_path)
        
        print('\n####################')
        print('Files written to:')
        print(sumstats_path)
        print('####################')
        
    
    phen_endtime = datetime.datetime.now()
    phen_elapsed = phen_endtime -phen_starttime
    print('\n####################')
    print('Finished '+sex+' phen '+phen)
    print('iter time: '+str(round(phen_elapsed.seconds/60, 2))+' minutes')
    print('####################')  