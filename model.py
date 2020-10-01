# -*- coding: utf-8 -*-
"""
Created on Thu Oct  1 16:29:53 2020

@author: 2nd pc
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.ensemble import RandomForestRegressor
import seaborn as sns
import pickle
 

def display_info(df_name , df):
    """
    Displays some basic information about dataframe
    """
    
    print("Data: {}".format(df_name))
    print("---------------------------------------------")
    print("Columns and Rows: {}".format(df.shape))
    print("Few coloumns of data frame")
    print(df.head())
    



def main():
    
    #load data from csv file
    data=pd.read_csv("car data.csv")
    
    #display raw data
    display_info("Raw Data",data)
    
    #drop name of car
    data.drop(["Car_Name"],axis=1,inplace=True)
    
    #Feature Engineering
    
    #derived the age of car by subtracting current year with year of car built
    data["CurrentYear"]=2020
    data["no_years"]=data["CurrentYear"]-data["Year"]
    
    #Drop the year and current year column
    data.drop(["Year","CurrentYear"],axis=1,inplace=True)
    
    #Create dummy variables
    final_data=pd.get_dummies(data,drop_first=True)
    
    #display cleaned dataframe
    display_info("Cleaned Data",final_data)
    
    
    #Independat and dependant variable
    X=final_data.iloc[:,1:]
    y=final_data.iloc[:,0]
    
    #Split the data into train and test set
    X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=42)
    
    #Train model randomforest
    rf=RandomForestRegressor()
    
    #Define hyperparameter grid
    n_estimators = [int(x) for x in np.linspace(100,1200, num=12)]
    max_features=["auto","sqrt"]
    max_depth=[int(x) for x in np.linspace(5,30, num=6)]
    min_samples_split=[2,5,10,15,100]
    min_samples_leaf=[1,2,5,10]
    
    #create search grid
    param_grid={
                "n_estimators":n_estimators,
                "max_features":max_features,
                "max_depth":max_depth,
                "min_samples_split":min_samples_split,
                "min_samples_leaf":min_samples_leaf}
    
    #Search through parameter grid
    rf_random=RandomizedSearchCV(
                                    rf,
                                    param_grid,
                                    cv=5,
                                    n_iter=10,
                                    scoring="neg_mean_squared_error",
                                    verbose=2,
                                    random_state=42,
                                    n_jobs=2)
    
    #train model
    rf_random.fit(X_train,y_train)
    
    #Prediction and Evaluation
    predict=rf_random.predict(X_test)
    
    #plot of difference btw test and predict
    sns.distplot(y_test-predict)
    plt.show()
    plt.scatter(y_test,predict)
    
    #pickle model
    file=open("random_forest_regressor_model.pkl","wb")
    
    pickle.dump(rf_random,file)


if __name__ == "__main__":
    main()


 