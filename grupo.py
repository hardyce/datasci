# -*- coding: utf-8 -*-
"""
Created on Sat Jul 23 18:52:05 2016

@author: hardy_000
"""

import pandas as pd

import numpy as np

import statsmodels.formula.api as sm
path="C:\\Users\\hardy_000\\grimbo\\train.csv"
data=pd.read_csv(path)
#data=data[0:1000]
#data['shop']=data['Ruta_SAK'].astype('str')+'c'+data['Cliente_ID'].astype('str')+'c'+data['Producto_ID'].astype('str')
np.average(data['Demanda_uni_equil'])
data['Cliente_ID']=data['Cliente_ID'].astype('category')
data['Ruta_SAK']=data['Ruta_SAK'].astype('category')
data['Producto_ID']=data['Producto_ID'].astype('category')
#clients=data.groupby(['Cliente_ID','Ruta_SAK','Semana'])
#means=clients.mean()
#data.join(means, on=['Cliente_ID','Ruta_SAK','Semana'], rsuffix='_av')
#data.dtypes

result = sm.ols(formula="Demanda_uni_equil ~ Venta_uni_hoy + Producto_ID+Ruta_SAK+Cliente_ID", data=data).fit()
result.predict(data.ix[0,:])
data.dtypes