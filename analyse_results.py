# -*- coding: utf-8 -*-
"""
Created on Sat Apr  4 20:15:45 2015

@author: mzhong

This method is to read the multi-measure errors for each appliance and for
each day, which are stored in an HDFStore file file_hdfstore.
"""

import numpy as np
import pandas as pd
from pandas import HDFStore, ExcelWriter

def read_performance_error(store,meterlist,excelfile):
    # Load the hdf file
    #store = HDFStore(file_hdfstore)
    invalidChar = '[]:*?/\\'
    # Save the error measures in a dict
    error_measures = {}
    for store_key in store.keys():
        print('Appliance:{}'.format(store_key))
        if 'ErrorMultiMeasure' in store_key:
            multiMeasure = store[store_key]
            for appliance in meterlist:
                if appliance in error_measures.keys():
                    df_appliance = pd.DataFrame(multiMeasure[appliance])
                    df_appliance.columns = [store_key]
                    #error_measures[appliance] = pd.concat([error_measures[appliance],
                    #                  df_appliance],ignore_index=True,axis=1)
                    error_measures[appliance].ix[:,store_key] = df_appliance
                else:
                    df_error = pd.DataFrame(multiMeasure[appliance])
                    df_error.columns = [store_key]
                    error_measures[appliance] = df_error
                    
    writer = ExcelWriter(excelfile)
    for appliance in error_measures:
        try:
            error_measures[appliance].to_excel(writer,appliance)
        except Exception:
            app_replace = appliance
            for c in invalidChar:
                app_replace = app_replace.replace(c,'-')
            error_measures[appliance].to_excel(writer,app_replace)
        meanErrorMeasure = pd.DataFrame(index=error_measures[appliance].index)
        meanErrorMeasure['mean'] = error_measures[appliance].mean(axis=1)
        meanErrorMeasure['std'] = error_measures[appliance].std(axis=1)
        try:
            meanErrorMeasure.to_excel(writer,appliance+'mean')
        except Exception:
            app_replace = appliance
            for c in invalidChar:
                app_replace = app_replace.replace(c,'-')
            meanErrorMeasure.to_excel(writer,app_replace+'mean')
    writer.save()
    #return error_measures