#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 16 13:51:32 2019

Analysis of 659 heritable phens from iterative GWAS

@author: nbaya
"""

import pandas as pd
import scipy.stats as stats
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


df = pd.read_csv('/Users/nbaya/Documents/lab/ukbb-sexdiff/rg_sex/ukbb31063.rg_sex.hm3.heritable_phens.nchunks2.batch_1.tsv.gz',compression='gzip',sep='\t')

sns.kdeplot(df[abs(df.rg-1)<0.5].rg)
plt.title('rg of 659 heritable phenotypes')
plt.ylabel('density')
plt.xlabel('rg')
fig=plt.gcf()
fig.set_size_inches(6, 4)
fig.savefig('/Users/nbaya/Documents/lab/ukbb-sexdiff/rg_sex/kdeplot_rg_659heritablephens.png',dpi=600)

print('Mean: '+str(np.mean(df.rg)))
print('Std: '+str(np.std(df.rg)))
print('ttest p-val: '+str(stats.ttest_1samp(df.rg,1)[1]))

#Calculate percentage of phens with CIs containing 1
contains_1 = [0]*659
for i in range(0,df.shape[0]):
    contains_1[i] = (df.loc[i,'rg']-2*df.loc[i,'se']<1)&(df.loc[i,'rg']+2*df.loc[i,'se']>1)
    contains_1[i] = (abs(df.loc[i,'z'])>2)
    
mat[contains_1,4]+2*mat[contains_1,5]

plt.plot(df.ph2_h2_obs, df.ph1_h2_obs,'.')
plt.plot([0,0.5],[0,0.5],lw=2,alpha=0.5,color='red',ls='--')

df.columns.values

mat = df.as_matrix()
plt.hist([x for x in mat[abs(mat[:,6])<2,4].flatten()])

plt.plot(mat[:,4],mat[:,5],'.')
plt.plot([1,1],[mat[:,5].min(),mat[:,5].max()],lw=2,alpha=0.5,color='red',ls='--')
plt.ylabel('SE')
plt.xlabel('rg')
fig=plt.gcf()
fig.set_size_inches(6, 4)
fig.savefig('/Users/nbaya/Documents/lab/ukbb-sexdiff/rg_sex/rg_vs_SE_659heritablephens.png',dpi=600)


plt.plot(mat[:,4],mat[:,6],'.')
plt.plot([1,1],[mat[:,6].min(),mat[:,6].max()],lw=2,alpha=0.5,color='red',ls='--')
plt.ylabel('Z')
plt.xlabel('rg')
fig=plt.gcf()
fig.set_size_inches(6, 4)
fig.savefig('/Users/nbaya/Documents/lab/ukbb-sexdiff/rg_sex/rg_vs_Z_659heritablephens.png',dpi=600)



###############################################################################
df1 = pd.read_csv('/Users/nbaya/Downloads/ukbb31063.rg_sex.hm3.sex_strat.nchunks2.batch_1.50_30_32.tsv.gz',compression='gzip',sep='\t')
df2 = pd.read_csv('/Users/nbaya/Downloads/ukbb31063.rg_sex.hm3.sex_strat.nchunks2.batch_1.50_34_36.tsv.gz',compression='gzip',sep='\t')
df_sex_strat = df1.append(df2,ignore_index=True)
df_sex_strat[['p1','p2','rg']]

rg_new = np.mean(df_sex_strat.loc[2:3,'rg'])/np.mean(df_sex_strat.loc[0:1,'rg'])

###############################################################################
"""
Compare random splits with sex splits
"""

df_rand = pd.read_csv('/Users/nbaya/Documents/lab/ukbb-sexdiff/rg_sex/ukbb31063.rg_sex.hm3.heritable_phens.nchunks2.batch_1.tsv.gz',compression='gzip',sep='\t')
df_sex = pd.read_csv('/Users/nbaya/Documents/lab/ukbb-sexdiff/imputed-v3-results/ukbb31063.phesant.rg_sex.batch_1.tsv.gz',compression='gzip',sep='\t')
df_sex['phenotype'] = df_sex['phenotype'].str.strip('_irnt')
#df_sex_alt = pd.read_csv('/Users/nbaya/Documents/lab/ukbb-sexdiff/imputed-v3-results/ukbb31063.rg_sex.v1.csv',sep=',')

combined = df_rand.merge(df_sex, on=['phenotype','description'],suffixes=['_rand','_sex'])

df_rand.columns.values

stat = 'gcov_int'
save = True
fig,ax = plt.subplots(figsize=(10,8))
ax.plot(combined[stat+'_rand'],combined[stat+'_sex'],'.')
plt.xlabel(stat+' (random split)')
plt.ylabel(stat+' (sex split)')
plt.title(stat+' of random and sex split')
fig=plt.gcf()
if save: fig.savefig('/Users/nbaya/Documents/lab/ukbb-sexdiff/rg_sex/plots/'+stat+'_random_vs_sex_split.png',dpi=300)

fig,ax = plt.subplots(figsize=(10,8))
sns.kdeplot(combined[stat+'_rand'],combined[stat+'_sex'],ax=ax,shade=True,shade_lowest=False,gridsize=100)
ax.plot(combined[stat+'_rand'],combined[stat+'_sex'],'k.',alpha=0.1)
plt.xlabel(stat+' (random split)')
plt.ylabel(stat+' (sex split)')
plt.title('bivariate kde for '+stat+' of random and sex split')
fig=plt.gcf()
if save: fig.savefig('/Users/nbaya/Documents/lab/ukbb-sexdiff/rg_sex/plots/bivariate_kde_'+stat+'_random_vs_sex_split.png',dpi=300)


fig,ax = plt.subplots(figsize=(10,8))
sns.kdeplot(combined[stat+'_rand'],ax=ax,shade=True,shade_lowest=False)
sns.kdeplot(combined[stat+'_sex'],ax=ax,shade=True,shade_lowest=False)
#sns.kdeplot(df_rand[stat],ax=ax,shade=True,shade_lowest=False)
#sns.kdeplot(df_sex[stat],ax=ax,shade=True,shade_lowest=False)
plt.title('bivariate kde for '+stat+' of random and sex split')
plt.legend(['random split','sex split'])
plt.xlabel(stat)
plt.ylabel('density')
fig=plt.gcf()
if save: fig.savefig('/Users/nbaya/Documents/lab/ukbb-sexdiff/rg_sex/plots/kdes_'+stat+'_random_vs_sex_split.png',dpi=300)

print(stat)
print('\nrandom split\nmean: '+str(np.mean(combined[stat+'_rand'])))
print('std: '+str(np.std(combined[stat+'_rand']))+'\n')
print('sex split\nmean: '+str(np.mean(combined[stat+'_sex'])))
print('std: '+str(np.std(combined[stat+'_sex']))+'\n')
print('n: '+str(len(combined[stat+'_sex']))+'\n')

