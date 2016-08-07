# -*- coding: utf-8 -*-
"""
Created on Sat Jul 23 18:52:05 2016

@author: hardy_000
"""

import pandas as pd

import numpy as np

import statsmodels.formula.api as sm

from patsy.contrasts import ContrastMatrix

def _name_levels(prefix, levels):
     return ["[%s%s]" % (prefix, level) for level in levels]


class Simple(object):
    def _simple_contrast(self, levels):
        nlevels = len(levels)
        contr = -1./nlevels * np.ones((nlevels, nlevels-1))
        contr[1:][np.diag_indices(nlevels-1)] = (nlevels-1.)/nlevels
        return contr
 

    def code_with_intercept(self, levels):
         contrast = np.column_stack((np.ones(len(levels)),
                                    self._simple_contrast(levels)))
         return ContrastMatrix(contrast, _name_levels("Simp.", levels))
    def code_without_intercept(self, levels):
          contrast = self._simple_contrast(levels)
          return ContrastMatrix(contrast, _name_levels("Simp.", levels[:-1]))


path="C:\\Users\\hardy_000\\grimbo\\train.csv"
data=pd.read_csv(path)
#data=data[0:1000]
#data['shop']=data['Ruta_SAK'].astype('str')+'c'+data['Cliente_ID'].astype('str')+'c'+data['Producto_ID'].astype('str')
#np.average(data['Demanda_uni_equil'])
data['Cliente_ID']=data['Cliente_ID'].astype('category')
data['Ruta_SAK']=data['Ruta_SAK'].astype('category')
data['Producto_ID']=data['Producto_ID'].astype('category')
#contrast = Simple().code_without_intercept(np.unique(data.Cliente_ID).shape)


clients=data.groupby(['Cliente_ID','Ruta_SAK','Producto_ID'])
means=clients.mean()
data.join(means, on=['Cliente_ID','Ruta_SAK','Semana'], rsuffix='_av')
#need to calc average and subtract
#data.dtypes
#result = sm.ols(formula="Demanda_uni_equil ~ C(Ruta_SAK,Simple)+Cliente_ID", data=data).fit()
result = sm.ols(formula="Demanda_uni_equil ~ Venta_uni_hoy + Producto_ID+Ruta_SAK+Cliente_ID", data=data).fit()
result.predict(data.ix[0,:])



