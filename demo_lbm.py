# -*- coding: utf-8 -*-
"""
Created on Wed Mar  4 14:03:27 2015

@author: mzhong
"""

# -*- coding: utf-8 -*-
# demo of nilmtk v0.1

from nilmtk import DataSet, TimeFrame
from nilmtk.disaggregate import CombinatorialOptimisation
from nilmtk.disaggregate import FHMM
from nilmtk import HDFDataStore
from nilmtk.disaggregate.afhmm_lbm import LatentBayesianMelding
from collections import OrderedDict
import pandas as pd
from pandas import HDFStore, ExcelWriter, date_range
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
from nilmtk.accuracymatrics import compute_error, compute_error_basic
from nilmtk.checkmodel import check_constraints
from nilmtk.sortmeter import readdata,truncate_meter
import datetime
import json
from analyse_results import read_performance_error

############# some global variables ##########################
# Sampling time was 2 minutes
sample_seconds = 120

######## Select a house: this is going to use house 2 in ukdale ###############
house = 2
store_afhmm_cm = HDFStore('store_afhmm_inducedDen_building2.h5')
exelfile = 'afhmm_inducedDen_building2.xlsx'
excelfile_appliance = 'afhmm_inducedDen_building2_appliance.xlsx'
exelfileTime = 'time_afhmm_indecedDen_building2.xlsx'
###########################################################
dftime = pd.DataFrame()
for day in range(1):
    day = day + 1
    ############# Read the data for prediction ###################
    target_file = 'energy-data/ukdale/ukdale.h5'
    ukdale = DataSet(target_file)

    ########### House 2 ################
    ######### Select a time period #####################
    start = datetime.datetime(2013, 6, day, 0, 0)
    end = datetime.datetime(2013, 6, day, 23, 59)
    ######## Maping appliance names #####################
    appliance_map = {'cooker':"('cooker', 1)",
                     'kettle':"('kettle', 1)",
                     'dishwasher':"('dish washer', 1)",
                     'toaster':"('toaster', 1)",
                     'washingmachine':"('washing machine', 1)",
                     'fridgefreezer':"('fridge', 1)", 
                     'microwave':"('microwave', 1)"}
    ########### Appliances to be disaggregated ############
    meterlist = ['cooker', 'kettle', 'dishwasher','toaster',
                 'washingmachine','fridgefreezer', 'microwave']
    building = 'ukdale/building2/'
    ############ To get the data ##############################################
    ukdale.set_window(start=str(start), end=str(end))
    
    ########### Select the house ##############################################
    elec_reading = ukdale.buildings[2].elec    
    # Read the meter data and transform Watts to deci-Watt hours 
    meterdata = readdata(elec_reading,sample_seconds)*(((sample_seconds/60.0)/60.0)*10.0) 
    appliancedata = meterdata
    synthmains = meterdata.sum(axis=1)
    
    # Read the mains reading for prediction
    mains = elec_reading.mains().power_series(sample_period=sample_seconds).next()
    # transfer watts to deci-watt hours
    meterdata['mains'] = mains*(((sample_seconds/60.0)/60.0)*10.0) 
    meterdata['synthetic mains'] = synthmains
    
    ############### Maping appliance names #######################################
    groundTruthMeter = pd.DataFrame(index=meterdata.index)
    for meter in appliance_map:
        groundTruthMeter[meter] = meterdata[appliance_map[meter]]
        appliancedata = appliancedata.drop(appliance_map[meter],axis=1)
    groundTruthMeter['othermeters'] = appliancedata.sum(axis=1)
    
    #############################################################################
    
    #### disaggregating the mains into appliances ################################
    lbm = LatentBayesianMelding()
    
    # to obtain the model parameters
    individual_model = lbm.train(meterlist)
    
    output = HDFStore('output.h5')
    mains = elec_reading.mains()
    prediction = lbm.disaggregate(mains,output)
    output.close()
    
    # the time
    #dftime.ix[house,str(day)] = sum(prediction['time'])
    
    ############### The prediction results ############################
    infAppliance=prediction['inferred appliance energy']
    latentEnergy = prediction['inferred latent energy']
    
    ############### Compute the error measurements ####################
    method_name = [str(start.date())]
    (errorFrame, errorFrameSingle) = compute_error(prediction,groundTruthMeter,
                                               meterlist,sample_seconds,method_name,individual_model)
    keyname = building+str(start.date())+'/inferredAppliance'                                                
    store_afhmm_cm[keyname] = infAppliance
    keyname = building+str(start.date())+'/latentEnergy'                                                
    store_afhmm_cm[keyname] = latentEnergy
    keyname = building+str(start.date())+'/groundTruthAppliance'                                                
    store_afhmm_cm[keyname] = groundTruthMeter
    keyname = building+str(start.date())+'/ErrorSingleMeasure'
    store_afhmm_cm[keyname] = errorFrameSingle  
    keyname = building+str(start.date())+'/ErrorMultiMeasure'
    store_afhmm_cm[keyname] = errorFrame
    if day is 1:
        errorMeasure = errorFrameSingle
    else:
        for index in errorMeasure.index:
            errorMeasure.ix[index,method_name[0]] = \
                                        errorFrameSingle.ix[index,method_name[0]]
## write data frame to excel
writer = ExcelWriter(exelfile)
errorMeasure.to_excel(writer,'Sheet1')
meanErrorMeasure = pd.DataFrame(index=errorMeasure.index)
meanErrorMeasure['mean'] = errorMeasure.mean(axis=1)
meanErrorMeasure['std']=errorMeasure.std(axis=1)
meanErrorMeasure.to_excel(writer,'Sheet2')
writer.save()
        


