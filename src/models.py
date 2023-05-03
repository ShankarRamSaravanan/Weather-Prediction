from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.svm import SVR

import xgboost as xgb
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import statsmodels.api as sm
import datetime as dt
import util
import streamlit as st
from prophet import Prophet

def run_linear_regression(weather):
    train_size = 18366
    Y = weather['CO2']
    Y_train = Y[:train_size]
    Y_test = Y[train_size:]

    X = weather[['DATE']]
    X['DATE'] = X['DATE'].map(dt.datetime.toordinal)
    X_train = X[:train_size]
    X_test = X[train_size:]

    lm = LinearRegression()
    lm.fit(X_train, Y_train)
    y_pred = lm.predict(X_test)

    print("R^2 : ", r2_score(Y_test, y_pred))
    print("MAE :", mean_absolute_error(Y_test,y_pred))
    print("RMSE:",np.sqrt(mean_squared_error(Y_test, y_pred)))
    
def run_random_forest(weather):
    model_graphs = {}
    Y = weather['CO2']
    X = weather[['Day','Month', 'Year']]
    today = dt.date.today()
    co2_forecast = util.create_dates(str(today), 365)
    co2_forecast['Day'] = co2_forecast.index.day
    co2_forecast['Month'] = co2_forecast.index.month
    co2_forecast['Year'] = co2_forecast.index.year
    co2_forecast = co2_forecast.reset_index(drop=True)
    X_train, X_test, Y_train, Y_test = train_test_split(X,Y,test_size=0.20, random_state=40)
    forestModel = RandomForestRegressor(n_estimators=200, max_depth=70, random_state =77, n_jobs=-1)
    forestModel.fit(X_train, Y_train)
    y_pred = forestModel.predict(X_test)
    y_forecast = forestModel.predict(co2_forecast)

    print("R^2 : ", r2_score(Y_test, y_pred))
    print("MAE :", mean_absolute_error(Y_test,y_pred))
    print("RMSE:",np.sqrt(mean_squared_error(Y_test, y_pred)))

    co2_forecast["CO2"] = np.array(y_forecast)
    fig = plt.figure()
    plt.plot(co2_forecast["CO2"], color='red')
    plt.ylabel('Parts per million(ppm)')  
    plt.title('CO2 prediction')

    model_graphs["Predicted CO2 values for next 365 days"] = fig
    return model_graphs

def run_xgboost(weather):
    train_size = 18366
    Y = weather['TMAX']
    Y_train = Y[:train_size]
    Y_test = Y[train_size:]

    X = weather[['Day','Month','Year','SEASON', 'CO2','TSI', 'PRCP', 'SNOW']]
    X_train = X[:train_size]
    X_test = X[train_size:]

    xgb_model = xgb.XGBRegressor()
    xgb_model.fit(X_train, Y_train)
    y_xgb_pred = xgb_model.predict(X_test)

    print("R^2 : ", r2_score(Y_test, y_xgb_pred))
    print("MAE :", mean_absolute_error(Y_test,y_xgb_pred))
    print("RMSE:",np.sqrt(mean_squared_error(Y_test, y_xgb_pred)))

    plt.scatter(Y_test, y_xgb_pred, color='c')

def run_svm(weather):
    Y = weather['TMIN']
    X = weather[['Day','Month', 'Year','SEASON','CO2', 'TSI', 'PRCP', 'SNOW']]
    X_train, X_test, Y_train, Y_test = train_test_split(X,Y,test_size=0.20, random_state=40)

    regr = make_pipeline(StandardScaler(), SVR(C=1.0, epsilon=0.2))
    regr.fit(X_train, Y_train)
    y_pred = regr.predict(X_test)

    print("R^2 : ", r2_score(Y_test, y_pred))
    print("MAE :", mean_absolute_error(Y_test,y_pred))
    print("RMSE:",np.sqrt(mean_squared_error(Y_test, y_pred)))

    plt.scatter(Y_test, y_pred)

def run_sarimax(weather):
    new_stat_model = weather[['DATE','TAVG']]
    new_stat_model.set_index('DATE',inplace=True)

    new_data_temp = pd.DataFrame(new_stat_model.TAVG.resample('M').mean())

    model=sm.tsa.statespace.SARIMAX(new_data_temp.TAVG,order=(1, 1, 1),seasonal_order=(1,1,1,12))
    results=model.fit()

    new_data_temp['forecast']=results.predict(start=80,end=150,dynamic=True)
    new_data_temp[['TAVG','forecast']].plot(figsize=(15,6))
    
def create_data(x, len_seq):
    X = []  
    y = []  
    for i in range(len(x) - len_seq):
        X.append(x[i:i+len_seq])
        y.append(x[i+len_seq])  
    return np.array(X), np.array(y)

def lstm_model(units, X_train, X_test, Y_train, Y_test):
    model = Sequential()
    model.add(LSTM(units,return_sequences=True, input_shape=(X_train.shape[1],X_train.shape[2]),kernel_initializer='lecun_uniform'))
    model.add(Dropout(0.2))    
    model.add(LSTM(units, return_sequences=True))
    model.add(Dropout(0.2))    
    model.add(LSTM(units))
    model.add(Dropout(0.2))
    model.add(Dense(1))        
    model.compile(optimizer=Adam(learning_rate=0.001), loss='mean_squared_error')
    model.fit(X_train, Y_train, batch_size=120, epochs=100, validation_data=(X_test, Y_test), verbose=0)
    return model  

def run_forecast(model, X_test, test, lag, scaler, days):
    seven_days = []
    new0 = X_test[-1]        
    last = test.iloc[-1]
    new_predict = last[0]        
    new_array = np.insert(new0, 0, new_predict)        
    new_array = np.delete(new_array, -1)
    new_array_reshape = np.reshape(new_array, (-1,1,lag))       
    new_predict = model.predict(new_array_reshape)
    temp_predict = scaler.inverse_transform(new_predict) 
    seven_days.append(temp_predict[0][0].round(2))
    
    for i in range(1,days):
        new_array = np.insert(new_array, 0, new_predict)             
        new_array = np.delete(new_array, -1)
        new_array_reshape = np.reshape(new_array, (-1,1,lag))            
        new_predict = model.predict(new_array_reshape)
        temp_predict = scaler.inverse_transform(new_predict) 
        seven_days.append(temp_predict[0][0].round(2))

    return seven_days

def run_lstm(weather, lag, days):
    model_graphs = {}
    weather = weather[['DATE','TAVG']]
    weather = weather.set_index(weather.columns[0])
    today = dt.date.today()
    lstm_forecast = util.create_dates(str(today),days)

    average_temperatures = weather[[weather.columns[0]]].dropna()      
    train, test = util.train_test_split(average_temperatures, 0.8)        
    train_scaled_df, test_scaled_df, scaler = util.data_transformation(train,test) 

    train = util.get_lagged_values(train_scaled_df, lag+1) 
    test = util.get_lagged_values(test_scaled_df, lag+1)              

    X_train, Y_train, X_test, Y_test = util.create_train_test(train,test)        
    X_train = np.reshape(X_train, (X_train.shape[0],1,X_train.shape[1]))
    X_test = np.reshape(X_test, (X_test.shape[0],1,X_test.shape[1]))                
    model = lstm_model(30, X_train, X_test, Y_train, Y_test)             

    #Test the model and find the R2 score.
    y_pred = model.predict(X_test)
    r2 = r2_score(Y_test, y_pred)
    print('R2 score is: %f'%r2) 

    forecast = run_forecast(model, X_test, test, lag, scaler, days)       
    lstm_forecast["TAVG"] = np.array(forecast)       
    
    fig = plt.figure()
    plt.plot(lstm_forecast["TAVG"], color='red')
    plt.ylabel('Temperature(°C)')  
    plt.title('Average temperature forecast')

    model_graphs["Predicted weather for next 7 days"] = fig
    return model_graphs

def run_fb_prophet(weather, num_days):
    model_graphs = {}
    tomorrow = (dt.date.today() + dt.timedelta(days=1)).strftime('%Y-%m-%d')
    last_day = (dt.date.today() + dt.timedelta(days=int(num_days))).strftime('%Y-%m-%d')

    print(tomorrow)
    print(last_day)
    weather_new =weather[['DATE','TAVG']]
    weather_new['DATE'] = pd.to_datetime(weather_new['DATE'],errors='coerce') 
    weather_new.columns = ['ds','y']

    if weather is not None:
        obj = Prophet()
        obj.fit(weather_new)
        future = obj.make_future_dataframe(periods=700)
        weather_forecast = obj.predict(future)
        forecast = weather_forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']]
        forecast = forecast.set_index('ds')
        forecast = forecast.loc[tomorrow:last_day]

        fig = plt.figure()
        plt.plot(forecast["yhat"], color='red')
        plt.ylabel('Temperature(°C)')  
        plt.title('Average temperature forecast')
        title = "Predicted weather for next {0} days".format(num_days)
        model_graphs[title] = fig
        return model_graphs
        
    
        

