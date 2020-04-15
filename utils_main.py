"""
Created on Wed Apr  1 18:48:13 2020

@author: Abhi

Functions for the prediction of the weather
"""
# Required Libraries

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
from sklearn.svm import SVR
plt.show()



def data_processing(weather_data):
    
    #Removing the data with NaN values
    weather_data.dropna(inplace =True)
    
#    corr_mat = weather_data.corr()
    #sns.heatmap(corr_mat,annot=True)
    
    
    #Extracting the required features
    final_data = weather_data[["Rain","AvgWind","AvgHumidity","AvgPressure"]]
    
    #Avergaing the temperature value
    final_data["temperature"] = (weather_data["MinTemp"]+weather_data["MaxTemp"])/2 #** do something here for this equation **

#    corr_mat = weather_data.corr()
    
    #Data for the regression
    #Splitting the data into dependent variable and independent variables
    
    X = final_data.iloc[:,:4].values
    y = final_data.iloc[:,4].values
    
    return X,y


    
def model(X,y):

    #Splitting the training and testing data
    X_train,X_test,y_train,y_test = train_test_split(X,y,train_size =0.75,random_state=1)
    
    #Fitting the training data
    
    ss = StandardScaler()
    ss.fit_transform(X_train)
    ss.fit(X_test)
    
    #Regression
    svr = SVR(kernel = "rbf", gamma = 0.001, C = 50, verbose=False)
    
    svr.fit(X_train,y_train)
    y_pred = svr.predict(X_test)  # with the splitted data
    r2 = r2_score(y_test,y_pred)

    print("\n\nThe R2 score is %f" % r2)
    # comparing with true value and predicted value
#    axis_1 = sns.distplot(y_test,hist=False,color ="r",label ="Actual Value")
#    sns.distplot(y_pred,color ="b",hist = False,label = "Predicted Value",ax =axis_1)
    
    return svr
    

def prediction_real(svr,test):
    
    test = np.array(test).reshape(1,-1) 
    pred = svr.predict(test)
    
    print("\nThe predicted Temperature value is %0.2f" %pred)

    return pred

def Input():
    
    print("Enter the required inputs for prediction ----> \n")
    rain_data = float(input("Rain_data        -  "))
    avg_wind = float(input("\nWind (kmph)      -  "))
    humidity = float(input("\nHumidity (%)     -  "))
    pressure = float(input("\nPressure (mb)    -  "))
    
    in_data = [rain_data,avg_wind,humidity,pressure]
    
    return in_data


