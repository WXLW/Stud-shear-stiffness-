#!/usr/bin/env python
# coding: utf-8
from pycaret.regression import load_model, predict_model
import streamlit as st
import pandas as pd
import numpy as np
import math

# Importing data using pandas
import pandas as pd
import matplotlib.pyplot as plt
from pycaret.regression import *
from timeit import default_timer as timer


data = pd.read_csv('Stud_shear_stiffness.csv')#, encoding= 'unicode_escape')


exp_reg101 = setup(data = data, target = 'ks', session_id=123,train_size = 0.8, 
                  normalize = True, normalize_method='maxabs', #transformation = True, transform_target = True, 
                  remove_outliers = True, outliers_threshold = 0.05,
                  silent =True)


X = get_config('X') 
y = get_config('y')
X_train = get_config('X_train') 
y_train = get_config('y_train') 
X_test = get_config('X_test') 
y_test = get_config('y_test') 
seed = get_config('seed') 

####################################
#######    0.Deep forest  ##########
####################################



estimator = 'DF'
ensemble =False
method =None
fold = 10
verbose = True
optimize = 'r2'
n_iter = 1000
#setting turbo parameters
cv = 3


#general dependencies
import numpy as np
from sklearn import metrics
from sklearn.model_selection import KFold
import random
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from scipy.stats import pearsonr
#setting numpy seed
np.random.seed(seed)

from deepforest import CascadeForestRegressor
from sklearn.metrics import mean_squared_error
#model = LinearRegression()
full_name = 'Deep forest'


model = CascadeForestRegressor(random_state=seed,max_depth=14, n_bins=229,n_estimators=3,n_trees=122)  
model.fit(X.values, y.values)
model11= load_model('Final ATDF Model 23April2022')


def predict(model,input_df):
    
    prep_pipe_transformer = model11.pop(0)

    Xtest = prep_pipe_transformer.transform(input_df)
    #predictions_df = model.predict(input_df.values)
    y_pred = model.predict(Xtest.values)
    predictions=y_pred[0][0]
    #predictions = predictions_df['Label'][0]
    return predictions


    



def run():
    st.write("""

    # StudATDF

    
    """)
    st.write("""
    An App For Predicting Shear Stiffness of Headed Studs in Steel-Concrete Composite Structures via Auto-Tuning Deep Forest   
    """)

    from PIL import Image
    image = Image.open('ATDF.png')

    st.image(image, width=1050)#,use_column_width=False)


    image1 = Image.open('G:\Jupyter\Lin.png')
    st.subheader('Modified Lin equation')
    st.image(image1, width=250)
    
    st.sidebar.header('User Input Features')
    
    ds= st.sidebar.slider('Stud shank diameter, ds (mm) ', 9.5, 31.8, 19.0)

    hs= st.sidebar.slider('Stud height, hs (mm) ', 20, 220, 110)

    dw= st.sidebar.slider('Weld collar diameter, dw (mm) ', 13.0, 44.5, 23.0)

    hw= st.sidebar.slider('Weld collar height, hw (mm) ', 1.7, 7.0, 2.0)

    tc= st.sidebar.slider('Concrete slab thickness , tc (mm) ', 30, 500, 200)

    Ec= st.sidebar.slider('Concrete elastic modulus, Ec(GPa)', 15.0, 50.0, 38.0)

    Es= st.sidebar.slider('Stud elastic modulus, Es(GPa)', 185.0, 220.0, 210.0)

    beta=st.sidebar.slider('Target reliability index, ' +chr(946), 1.5, 2.9, 2.9)


    input_dict = {'Ec': Ec,'Es': Es, 'ds': ds,
          'Aw' :dw*hw,'hs' :hs,'tc': tc}
                 
    input_df = pd.DataFrame([input_dict])
    input_df.reset_index(drop=True, inplace=True)    
    
    if st.button("Predict"):
       output = predict(model=model, input_df=input_df)
       output = round(output, 1)
       gama_M=1/(1.00*math.exp(-0.32*beta*0.312))
       d_output = round(output/gama_M, 1)
       output =  str(output) +'kN/mm'
       d_output =  str(d_output) +'kN/mm'
       
       st.subheader('Predictive Results by Auto-Tuning Deep Forest')
       st.success('The Nominal Shear Stiffness of Headed Studs is  :  {}'.format(output))
       
       st.success('The Design Shear Stiffness of Headed Studs is  :  {}'.format(d_output))       

       st.subheader('Predictive Results by Modified Lin equation')
       output_lin = round(0.4*ds*Es**(1/4)*Ec**(3/4), 1)
       gama_M_lin=1/(1.00*math.exp(-0.32*beta*0.42))
       d_output_lin = round(output_lin/gama_M_lin, 1)
       output_lin =  str(output_lin) +'kN/mm'
       d_output_lin =  str(d_output_lin) +'kN/mm'
              
       st.success('The Nominal Shear Stiffness of Headed Studs is  :  {}'.format(output_lin))
       
       st.success('The Design Shear Stiffness of Headed Studs is  :  {}'.format(d_output_lin)) 

       
    st.info('***Written by Dr. Xianlin Wang,  Department of bridge engineering,  Tongji University,  E-mail:xianlinwang96@gmail.com***')
       
    #output = predict(model=model, input_df=input_df)    
    
    #st.write(output)  assessment of 
        
if __name__ == '__main__':
    run()
