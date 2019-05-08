# -*- coding: utf-8 -*-
"""
Created on Friday January 1 20:27:27 2016

@author: Mingjun Zhong, School of Informatics, University of Edinburgh
Email: mingjun.zhong@gmail.com

Requirements: 
1) the (free academic) MOSEK license: 
        https://www.mosek.com/resources/academic-license 

References:
 [1] Mingjun Zhong, Nigel Goddard and Charles Sutton. 
 Latent Bayesian melding for integrating individual and population models.
 In Advances in Neural Information Processing Systems 28, 2015.
 [2] Mingjun Zhong, Nigel Goddard and Charles Sutton. 
 Signal Aggregate Constraints in Additive Factorial HMMs, 
 with Application to Energy Disaggregation.
 In Advances in Neural Information Processing Systems 27, 2014.
 
Note: 
1)  The optimization problems described in the papers were transformed to
    a second order conic programming (SOCP) which suits the MOSEK solver; 
    I should write a technical report for this.
2)  Any questions please drop me an email.
"""

# -*- coding: utf-8 -*-

import pandas as pd
from pandas import HDFStore
import matplotlib.pyplot as plt
from fhmm_relaxed import FHMM_Relaxed
from latent_Bayesian_melding import LatentBayesianMelding
import os

############# some global variables ##########################
# Sampling time was 6-7 second for refit
sample_seconds = 6
datadir = '/home/mzhong/MingjunZhong/lbmNilm/LatentBayesianMelding/dataset_management/refit/'
########### Appliances to be disaggregated: not all of the appliance #########
appliance_list = ['kettle', 'dishwasher','washingmachine','fridge', 'microwave']
meter_list = ['kettle']
dataset = 'test'
results_dir = '../results/refit/'
# read data
for appliance_name in meter_list:
    #appliance_name = 'kettle'    
    print(appliance_name)
    for filename in os.listdir(datadir + appliance_name):
            if dataset == 'train' and dataset.upper() in filename.upper() and 'TEST' in filename.upper():
                test_filename = filename
            elif dataset == 'training' and dataset.upper() in filename.upper():
                test_filename = filename
            elif dataset == 'test' and dataset.upper() in filename.upper() and 'train' not in filename.upper():
                test_filename = filename
            elif dataset == 'validation' and dataset.upper() in filename.upper():
                test_filename = filename
    data_test = pd.read_csv(datadir + appliance_name + '/' + test_filename,
                                            # index_col=False,
                                            names=['aggregate', appliance_name],
                                            # usecols=[1, 2],
                                            # iterator=True,
                                            #skiprows=15 * 10 ** 6,                                        
                                            header=0
                                            )     
####### Select a house: this is the house 2 in UKDALE #########
# This is a demo for applying Latent Bayesian Melding to energy disaggregation.
# The HES data were used for training the model parameters, and the population
# models. This demo shows how to use the LBM to disaggregate the mains readings
# of a day in UKDALE.
# Note that since HES data were read every 2 minutes, so UKDALE data were 
# resampled to 2 minutes.
# Note that all the readings were tranferred to the unit: deciwatt-hour, 
# which is identical to the unit of HES data. 
################################################################

# Note that this demo is only for a chunk (=a day).
# This should be easy to amend for all the data you want to disaggregate.
#########################################################################

## This is the building information: dataset/building_number/date
#building_information = 'ukdale/building2/2013-06-08'

# Read the data from h5 file.
# meterdata is a DataFrame and its columns are: 1) appliance readings;
# 2) the mains readings; 3) the synthetic mains readings which are the sum
# of all the appliance readings.
#meterdata_ukdale = HDFStore('meterdata_ukdale.h5')
#meterdata = meterdata_ukdale['meterdata']

# Map the keywords of appliances between HES and UKDALE
#appliance_map = {'cooker':"('cooker', 1)",
#                 'kettle':"('kettle', 1)",
#                 'dishwasher':"('dish washer', 1)",
#                 'toaster':"('toaster', 1)",
#                 'washingmachine':"('washing machine', 1)",
#                 'fridgefreezer':"('fridge', 1)", 
#                 'microwave':"('microwave', 1)"}
                 
## the ground truth reading for those appliances to be disaggregated ########
#appliancedata = meterdata
#groundTruthApplianceReading = pd.DataFrame(index=meterdata.index)
#for meter in appliance_map:
#    groundTruthApplianceReading[meter] = meterdata[appliance_map[meter]]
#    appliancedata = appliancedata.drop(appliance_map[meter],axis=1)    
## the sum of other meter readings which will not be disaggregated
#groundTruthApplianceReading['othermeters'] = appliancedata.sum(axis=1)

## The mains readings to be disaggregated ###
#mains = meterdata['mains']

#### declare an instance for lbm ################################
#lbm = LatentBayesianMelding()
lbm = FHMM_Relaxed()

# to obtain the model parameters trained by using HES data
individual_model = lbm.import_model(appliance_list,'individual_model_refit.json')

# try smaller data set
nchunk = 20
chunk_size = 720
data_test = data_test[0:nchunk*chunk_size]

# split dataframe into chuncks
df_list = [data_test[i:i+chunk_size] for i in range(0,data_test.shape[0],chunk_size)]

for ichunk, chunk_df in enumerate(df_list):

    mains_data = chunk_df['aggregate']
    appliance_chunk = chunk_df[appliance_name]
    
    # use lbm to disaggregate mains readings into appliances
    results = lbm.disaggregate_chunk(mains_data)
    
    # the inferred appliance readings
    infApplianceReading=results['inferred appliance energy']
    
    mains_chunk = infApplianceReading['mains']
    infmains_chunk = infApplianceReading['inferred mains']
    infAppliance_chunk = infApplianceReading[appliance_name]
    if ichunk==0:
        infAppliance = infAppliance_chunk
        appliance_data = appliance_chunk
        ground_mains = mains_chunk
        infmains = infmains_chunk
    else:
        infAppliance = infAppliance.append(infAppliance_chunk)
        appliance_data = appliance_data.append(appliance_chunk)
        ground_mains = ground_mains.append(mains_chunk)
        infmains = infmains.append(infmains_chunk)
infAppliance = infAppliance.rename("inferred_"+appliance_name)
df_appliance = pd.concat([appliance_data,infAppliance],axis=1)
df_appliance.to_csv(results_dir+appliance_name+'_inferred.csv')
        
# compare inferred appliance readings and the ground truth
for meter in meter_list:
    plt.figure()
    ax = appliance_data.plot(legend=True)
    infAppliance.plot(ax=ax,title=meter,color='r',legend=True)
    ax.legend(['truth','inferred'])
    ax.set_xlabel('time')
    ax.set_ylabel('deciwatt-hour')
plt.figure()
ax = ground_mains.plot(title='mains')
infmains.plot(ax=ax,color='r')
ax.legend(['truth','inferred'])
ax.set_xlabel('time')
ax.set_ylabel('deciwatt-hour')