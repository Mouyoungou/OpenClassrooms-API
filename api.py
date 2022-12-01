# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle
from flask import Flask, jsonify, request

app = Flask(__name__)

pred_model_banq2 = pickle.load( open( "pred_model_banq2.md", "rb" ) )
val_set = pickle.load( open( "val_set.p", "rb" ) )
#pred_frame=pd.read_csv('pred_frame.csv').drop('Unnamed: 0',axis=1)

@app.route('/')
def home():
    return 'Entrer une ID client dans la barre URL'

@app.route('/<int:ID>/')
def requet_ID(ID):
    
    if ID not in list(val_set['SK_ID_CURR']):
        result = 'Ce client n\'est pas dans la base de donnée'
    else:
        val_set_ID=val_set[val_set['SK_ID_CURR']==int(ID)]

        y_proba=pred_model_banq2.predict_proba(val_set_ID.drop(['SK_ID_CURR','TARGET'],axis=1))[:, 1]

        seuil=0.52 
       
        if y_proba >= seuil:
            y_Target=1
        elif y_proba < seuil:
            y_Target=0
   
   
        if y_Target == 0:
           result=('ce client est solvable avec un taux de risque de '+ str(np.around(y_proba*100,2))+'%')

        elif y_Target == 1:
            result=('ce client est non solvable avec un taux de risque de '+ str(np.around(y_proba*100,2))+'%')
  
    return result


if __name__ == '__main__':
  app.run()