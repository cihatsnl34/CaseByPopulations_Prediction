# -*- coding: utf-8 -*-
"""
Created on Tue Nov 30 22:07:37 2021

@author: cihat
"""

#1.kutuphaneler
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
from sklearn.metrics import r2_score
veriler= pd.read_csv("worldwide covid data.csv")

ulkeler = veriler.iloc[:,0:1]
population = veriler.iloc[:,9:10]
cases =  veriler.iloc[:,1:2]

#Veri Ön İşleme Bitti
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
ulkeler = pd.DataFrame(le.fit_transform(ulkeler))

x = pd.concat([ulkeler, population], axis=1, ignore_index=False)
y = cases
X=x.values
Y=y.values

#RASSAL AĞAÇLAR(RANDOM FOREST)

from sklearn.ensemble import RandomForestRegressor
rf_reg=RandomForestRegressor(n_estimators=10,random_state=0)
rf_reg.fit(X,Y.ravel())

print('Random Forest R2 degeri')
print(r2_score(Y, rf_reg.predict(X)))

for i in ulkeler.values:
    print(veriler.iloc[i,0].values,":",i)
    
 
country=int(input("Ülke Kodunu Giriniz : "))
pop=int(input("Nüfus Sayısını Girin : "))
print(rf_reg.predict([[country,pop]]))
