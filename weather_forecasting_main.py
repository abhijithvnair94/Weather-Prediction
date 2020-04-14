"""
Created on Wed Apr  1 12:01:10 2020

@author: Abhi

Weather Forecasting  - Daily Average temperature prediction
"""

import numpy as np
import pandas as pd
import utils_main as main


# In[1]:

weather_data = pd.read_csv("Weather_Data__.csv")

# In[2]:

weather_data.info()

# In[3]
# Data processing

X,y = main.data_processing(weather_data)

# In[4]:
# Regression result

final_model = main.model(X,y) #final_model --> trained model


# In[5]

''' Predicting the temperature
 Variables for prediction ----- 
 [Rain, Avg_Wind,Avg_Humdity, Avg_Pressure] '''

variables = main.Input() 

predicted_value = main.prediction_real(final_model,variables)

