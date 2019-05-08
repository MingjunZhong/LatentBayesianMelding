#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu May  2 15:11:45 2019

@author: mzhong
"""
# simple example:
#X1 = [[0.5], [1.0], [-1.0], [0.42], [0.24]]
#X2 = [[2.4], [4.2], [0.5], [-0.24]]
#X = np.concatenate([X1, X2])
#lengths = [len(X1), len(X2)]
#model = hmm.GaussianHMM(n_components=3).fit(X)

import numpy as np
from hmmlearn import hmm
import pandas as pd
import os
import matplotlib.pyplot as plt
import json

appliance_list = ['kettle','microwave','fridge','washingmachine','dishwasher']
noStates = {'kettle':3,'microwave':3,'fridge':2,'washingmachine':3,'dishwasher':3}
dataset = 'validation'
chunksize = 10 ** 6
datadir = '/home/mzhong/MingjunZhong/lbmNilm/LatentBayesianMelding/dataset_management/refit/'
individual_model_refit = {}

for appliance_name in appliance_list:
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
    data_training = pd.read_csv(datadir + appliance_name + '/' + test_filename,
                                            # index_col=False,
                                            names=['aggregate', appliance_name],
                                            # usecols=[1, 2],
                                            # iterator=True,
                                            #skiprows=15 * 10 ** 6,                                        
                                            header=0
                                            )         
    lengths = len(data_training[appliance_name])   
    X_data = data_training[appliance_name].values.reshape((lengths,1))
    #plt.plot(X)
    #lengths = len(X_data)
    n_components = noStates[appliance_name]
    model = hmm.GaussianHMM(n_components=n_components).fit(X_data)
    
    # get the parameters
    startprob = model.startprob_.reshape((n_components,1)).tolist()
    transprob = model.transmat_.transpose().tolist()
    means = model.means_.tolist()
    numberOfStates = n_components
    adict = {'startprob':startprob,
             'transprob':transprob,
             'means':means,
             'numberOfStates':numberOfStates}
    individual_model_refit[appliance_name] = adict
    
with open('individual_model_refit.json', 'w') as fp:
    json.dump(individual_model_refit, fp)
    