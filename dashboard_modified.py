# -*- coding: utf-8 -*-
"""

@author: Stephen Faller
"""
import streamlit as st
import shap
import streamlit.components.v1 as components
import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from lightgbm import LGBMClassifier
import time


#Chargement des dataset
shap_values = pickle.load( open( "shap_values.p", "rb" ) )
set_shap = pickle.load( open( "set_shap.p", "rb" ) )
pred_frame_dash_s1000 = pickle.load( open( "pred_frame_dash_s1000.p", "rb" ) )
pred_model_banq2 = pickle.load( open( "pred_model_banq2.md", "rb" ) )
set_tru_data = pickle.load( open( "set_tru_data.p", "rb" ) )


#fenetre input
st.title('SCORING CREDIT FOR A CUSTOMER')
st.subheader("Customer bank score predictions")
id_input = st.text_input('Please enter the customer ID:', )
st.write('Customer ID example: 407405; 336258; 332126')

@st.cache
def requet_ID(ID):
    
    ID_client=int(ID)
    
    if ID_client not in list(pred_frame_dash_s1000['SK_ID_CURR']) :
        result = 'This customer is not in the database.'
    
    else:
        
        #Récupération des infos clients
        X_ID=pred_frame_dash_s1000[pred_frame_dash_s1000['SK_ID_CURR']==ID_client].copy()
        X=X_ID.drop(['SK_ID_CURR','Proba', 'TARGET'],axis=1)
        y_ID=X_ID['SK_ID_CURR']
        y_Target=X_ID['TARGET']
        y_proba=X_ID['Proba']
      
        if y_Target.item() == 0:
          result=('This client is creditworthy with a risk rate of '+ str(round(y_proba.iloc[0]*100,2))+'%')
        
        elif y_Target.item() == 1:
          result=('This client is not creditworthy with a risk rate of '+ str(round(y_proba.iloc[0]*100,2))+'%')
      
    return result

def plot_ft_global(ID,frame_shap):
    id_num=int(ID)    
    ### Récupération des feat. les plus importants pour le client
    set_shap_id=frame_shap[frame_shap["SK_ID_CURR"]==id_num].copy().T
    set_shap_id=set_shap_id.rename({frame_shap[frame_shap["SK_ID_CURR"]==id_num].index.item(): 'valeur'}, axis=1)
    set_shap_id=set_shap_id.drop(['SK_ID_CURR','Proba', 'TARGET','PREDICTION'],axis=0).sort_values(by='valeur')
  
    ft_ID_0=[]
    ft_ID_index_0=[]
    ft_ID_1=[]
    ft_ID_index_1=[]

    ft_ID_1.append(set_shap_id['valeur'][0])
    ft_ID_index_1.append(set_shap_id[set_shap_id['valeur']==ft_ID_1[0]].index.item())
    ft_ID_1.append(set_shap_id['valeur'][1])
    ft_ID_index_1.append(set_shap_id[set_shap_id['valeur']==ft_ID_1[1]].index.item())

    ft_ID_0.append(set_shap_id['valeur'][-1])
    ft_ID_index_0.append(set_shap_id[set_shap_id['valeur']==ft_ID_0[0]].index.item())
    ft_ID_0.append(set_shap_id['valeur'][-2])
    ft_ID_index_0.append(set_shap_id[set_shap_id['valeur']==ft_ID_0[1]].index.item())


    #Feat augmentant le risque

    #calcul des moyennes global pour les feat. les plus importants
    ft_moy_glob=[]
    ft_moy_0=[]
    ft_moy_1=[]
    ft_glob_index=[]

    for ft in ft_ID_index_1:
        ft_glob_index.append(ft)
        #moy global
        ft_moy_glob.append(frame_shap[ft].mean())
        #moy sur les client solvable
        ft_moy_0.append(frame_shap[frame_shap['TARGET']==0][ft].mean())
        #moy sur les client non solvable
        ft_moy_1.append(frame_shap[frame_shap['TARGET']==1][ft].mean())
        
    set_ft_glob=pd.DataFrame()
    set_ft_glob['feature']=ft_glob_index
    set_ft_glob['moy_global']=ft_moy_glob
    set_ft_glob['moy_solvable']=ft_moy_0
    set_ft_glob['moy_non_solvable']=ft_moy_1
    set_ft_glob['ID:'+str(id_num)]=ft_ID_1
    set_ft_glob=set_ft_glob.set_index(set_ft_glob['feature']).T.drop('feature')
    

    st.write('Comparison of client information for key indicators increasing risk against all clients in the database.')

    fig,ax=plt.subplots( figsize=(10,4))

    for ft in set_ft_glob.columns:
        st.subheader(ft)
        sns.barplot(y = set_ft_glob[ft].index,x=set_ft_glob[ft])
        plt.show()
        st.pyplot(fig)

    #feat diminuant le risque

    #calcul des moyennes globales pour les feat. les plus importants
    ft_moy_glob=[]
    ft_moy_0=[]
    ft_moy_1=[]
    ft_glob_index=[]

    for ft in ft_ID_index_0:
        ft_glob_index.append(ft)
        #moy global
        ft_moy_glob.append(frame_shap[ft].mean())
        #moy sur les client solvable
        ft_moy_0.append(frame_shap[frame_shap['TARGET']==0][ft].mean())
        #moy sur les client non solvable
        ft_moy_1.append(frame_shap[frame_shap['TARGET']==1][ft].mean())

    set_ft_glob_0=pd.DataFrame()
    set_ft_glob_0['feature']=ft_glob_index
    set_ft_glob_0['moy_global']=ft_moy_glob
    set_ft_glob_0['moy_solvable']=ft_moy_0
    set_ft_glob_0['moy_non_solvable']=ft_moy_1
    set_ft_glob_0['ID:'+str(id_num)]=ft_ID_0
    set_ft_glob_0=set_ft_glob_0.set_index(set_ft_glob_0['feature']).T.drop('feature')

    st.write('Comparison of client information for key risk decreasing indicators against all clients in the database.')
    fig,ax=plt.subplots(figsize=(10,4))

    for ft in set_ft_glob_0.columns:
        st.subheader(ft)
        sns.barplot(y = set_ft_glob_0[ft].index,x=set_ft_glob_0[ft])
        plt.show()
        st.pyplot(fig)

def hist_plot_global(ID,frame_shap,frame_cl):
  id_num=int(ID)
  ### Récupération des feat. les plus importants pour le client
  set_shap_id=frame_shap[frame_shap['SK_ID_CURR']==id_num].copy().T
  set_shap_id=set_shap_id.rename({frame_shap[frame_shap["SK_ID_CURR"]==id_num].index.item(): 'valeur'}, axis=1)
  set_shap_id=set_shap_id.drop(['SK_ID_CURR','Proba', 'TARGET','PREDICTION'],axis=0).sort_values(by='valeur')

  ft_ID_1=[]
  ft_ID_0=[]
  ft_ID_1.append(set_shap_id.index[0])
  ft_ID_1.append(set_shap_id.index[1])
  ft_ID_0.append(set_shap_id.index[-1])
  ft_ID_0.append(set_shap_id.index[-2])

  #Feat augmentant le risque
   
  st.write('Comparison of client information for key indicators increasing risk against all clients in the database.')
  for ft in ft_ID_1:
    #plt.style.use('seaborn-deep')  
    st.subheader(ft)
    fig,ax=plt.subplots( figsize=(10,4))
    
    x = frame_cl[frame_cl['TARGET']==0][ft]
    y = frame_cl[frame_cl['TARGET']==1][ft]
    z = frame_cl[ft]
    bins = np.linspace(0, 1, 15)

    risque_client=frame_cl[frame_cl['SK_ID_CURR']==id_num][ft].item()

    plt.hist([x, y,z], bins, label=['Solvable', 'Non solvable','Global'])
    plt.axvline(risque_client,linewidth=4, color='#d62728')

    plt.legend(loc='upper right')
    plt.ylabel('Nb de client')
    plt.xlabel('Valeur normalisée')
    plt.figtext(0.755,0.855,'-',fontsize = 60,fontweight = 'bold',color = '#d62728')
    plt.figtext(0.797,0.9,'Client '+str(id_num))
    plt.show()
    st.pyplot(fig)

  #feat diminuant le risque
  
  st.write('Comparison of client information for key indicators decreasing risk against all clients in the database.')
  for ft in ft_ID_0:
    st.subheader(ft)
    fig,ax=plt.subplots( figsize=(10,4))
    
    #plt.style.use('seaborn-deep')

    x = frame_cl[frame_cl['TARGET']==0][ft]
    y = frame_cl[frame_cl['TARGET']==1][ft]
    z = frame_cl[ft]
    bins = np.linspace(0, 1, 15)

    risque_client=frame_cl[frame_cl['SK_ID_CURR']==id_num][ft].item()

    plt.hist([x, y,z], bins, label=['creditworthy', 'Non creditworthy ','Global'])
    plt.axvline(risque_client,linewidth=4, color='#d62728')

    plt.legend(loc='upper right')
    plt.ylabel('Number of client')
    plt.xlabel('Normalized value')
    plt.figtext(0.755,0.855,'-',fontsize = 60,fontweight = 'bold',color = '#d62728')
    plt.figtext(0.797,0.9,'Client '+str(id_num))
    plt.show() 
    st.pyplot(fig)


def comparaison_client_voisin(ID,frame_shap,frame_hist):
  
  frame_shap['AGE']=round(frame_shap['DAYS_BIRTH'],1)
  ID_c=int(ID)

  #INFO CLIENT SHAP
  info_client=frame_shap[frame_shap['SK_ID_CURR']==ID_c]
  enfant_c=info_client['CNT_CHILDREN'].item()
  age_c=info_client['AGE'].item()
  genre_c=info_client['CODE_GENDER_M'].item()
  region_c=info_client['REGION_RATING_CLIENT'].item()
     
  #PROCHE VOISIN
  enfant_v=frame_shap[frame_shap['CNT_CHILDREN']==enfant_c]
  age_v=enfant_v[enfant_v['AGE']==age_c]
  genre_v=age_v[age_v['CODE_GENDER_M']==genre_c]
  region_v=genre_v[genre_v['REGION_RATING_CLIENT']==region_c]

  if len(region_v) < 15:
    set_client_voisin=region_v.sample(len(region_v),random_state=42)
  if len(region_v) >= 15:
    set_client_voisin=region_v.sample(15,random_state=42)

  fig,ax=plt.subplots( figsize=(10,4))
  plt.barh(range(len(set_client_voisin)),set_client_voisin['Proba'])
  risque_client=info_client['Proba'].item()
  plt.axvline(x=risque_client,linewidth=8, color='#d62728')
  plt.xlabel('% of risk')
  plt.ylabel('N° similar profiles')
  plt.figtext(0.755,0.855,'-',fontsize = 60,fontweight = 'bold',color = '#d62728')
  plt.figtext(0.797,0.9,'Client '+str(ID_c))
  st.pyplot(fig)

  moy_vois=set_client_voisin['Proba'].mean()
  diff_proba=round(abs(risque_client-moy_vois)*100,2)
  st.write('The client',str(ID_c),'has a difference of',str(diff_proba),'% of risk with customers of similar profiles.')
  
  hist_plot_global(ID_c,frame_hist,set_client_voisin)

  

def profil_client(ID,frame_True):
  
  frame_True['AGE']=round(abs(frame_True['DAYS_BIRTH']/365)).astype(int)
  ID_c=int(ID)  
  #INFO CLIENT True
  info_client_t=frame_True[frame_True['SK_ID_CURR']==ID_c]
  enfant_c_t=info_client_t['CNT_CHILDREN'].item()
  age_c_t=info_client_t['AGE'].item()
  genre_c_t=info_client_t['CODE_GENDER'].item()
  region_c_t=info_client_t['REGION_RATING_CLIENT'].item()
  arr_cl=[]
  arr_cl.append(ID_c)
  arr_cl.append(age_c_t)
  arr_cl.append(genre_c_t)
  arr_cl.append(enfant_c_t)
  arr_cl.append(region_c_t)
  frame_info_client=pd.DataFrame(arr_cl)
  frame_info_client=frame_info_client.T
  frame_info_client.columns=['SK_ID_CURR','AGE','CODE_GENDER','CNT_CHILDREN','REGION_RATING_CLIENT']
  frame_info_client.index=[str(ID_c)]
  return frame_info_client


def st_shap(plot, height=None):
    shap_html = f"<head>{shap.getjs()}</head><body>{plot.html()}</body>"
    components.html(shap_html, height=height)

def plot_shap(ID):
    ID_client=int(ID)
    
    X_ID=pred_frame_dash_s1000[pred_frame_dash_s1000['SK_ID_CURR']==ID_client].copy()
    X_ID=X_ID.reset_index(drop=True)
    X=X_ID.drop(['SK_ID_CURR','Proba', 'TARGET','PREDICTION'],axis=1)
    y_ID=X_ID['SK_ID_CURR']
    y_Target=X_ID['TARGET']
    y_proba=X_ID['Proba']
    y_pred=X_ID['PREDICTION']   
    explainer = shap.TreeExplainer(pred_model_banq2)
    shap_values = explainer.shap_values(X)
    st.subheader('Main indicators influencing the risk rate')
    st_shap(shap.force_plot(explainer.expected_value[0], shap_values[0][0,:], X.iloc[0,:]))


#######
#Appel de l'API : 
    API_url = "http://127.0.0.1:5000/" + str(id_input)

    with st.spinner('Please enter a customer ID in the URL bar...'):
        json_url = urlopen(API_url)

        API_data = json.loads(json_url.read())
        prediction = API_data['client risk in %']

if id_input == '':
    st.write('Please enter an ID')
    
else:
    r_ID=requet_ID(id_input)
    st.write(r_ID)
    
    if r_ID != 'This customer is not in the database.':
    
        option = st.sidebar.selectbox('Interpretation',['','Individual','Global','Similar profiles'])  
               
        if option == 'Global':
            st.sidebar.subheader('Customer profile '+ str(id_input))
            st.sidebar.write(profil_client(id_input,set_tru_data))
            st.write(hist_plot_global(id_input,set_shap,pred_frame_dash_s1000))
        
        elif option == 'Individual':
            st.sidebar.subheader('Customer profile '+ str(id_input))
            st.sidebar.write(profil_client(id_input,set_tru_data))
            st.write(plot_shap(id_input,set_shap,pred_frame_dash_s1000))
            
        elif option == 'Similar profiles':
            st.sidebar.subheader('Customer profile '+ str(id_input))
            st.sidebar.write(profil_client(id_input,set_tru_data))
            st.write(comparaison_client_voisin(id_input, pred_frame_dash_s1000,set_shap))
        
        else:
            st.sidebar.write('Please choose a comparison mode')