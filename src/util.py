import netCDF4 as nc
import pandas as pd
import numpy as np
from functools import reduce
import streamlit as st
from sklearn.preprocessing import MinMaxScaler

def create_dates(start,days):
    dates = pd.date_range(start=start, periods=days+1, freq='D', inclusive='right')
    date_df = pd.DataFrame(index=dates) 
    return date_df

def train_test_split(value, ratio):
    split_row = int(len(value)*ratio)
    train = value.iloc[:split_row]
    test = value.iloc[split_row:]
    return train, test

def load_and_clean_data(weather_dataset_buffer):
    weather_data = load_weather_dataset(weather_dataset_buffer)
    co2_data = load_co2_data()
    pd_tsr_data = load_solar_radiation_data()
    
    #Integrate all the dataframes
    weather = integrate_data_frames(pd_tsr_data, weather_data,co2_data)
    weather = add_tavg_column(weather)
    weather = add_season_column(weather)
    return weather

def load_weather_dataset(weather_dataset_buffer):
    weather_cols_needed = [2,3,4,5,6]    #Selecting required columns from the dataset
    weather_data = pd.read_csv(weather_dataset_buffer, usecols=weather_cols_needed)
    print("The current type of date column is:",weather_data['DATE'].dtype)
    weather_data['DATE'] = pd.to_datetime(weather_data['DATE'])
    print("The type of date column after conversion is",weather_data['DATE'].dtype)
    weather_data.head(10)
    return weather_data

@st.cache
def load_co2_data():
    co2_data = pd.read_csv("../datasets/co2_1958_current.csv")
    print("\n CO2 data before dropping\n", co2_data.head(10))
    co2_data.drop(co2_data.columns[[4,5]], axis=1, inplace= True) #drop unnecessary columns.
    print("\n CO2 data after dropping\n", co2_data.head(10))
    print(co2_data.shape)
    print("\n\n\nMissing values in wise of column:\n",co2_data.isna().sum(axis='rows'))
    print("\n\n\nTotal no of NA's in the Co2 dataset :",co2_data.isna().sum().sum())

    """<h3> Data Transformation on Atmospheric Co2 dataset</h3>"""

    print("CO2 dataset before merging columns to form single date column:\n", co2_data.head(10))
    co2_data = co2_data.dropna(axis = 'rows')    # Dropping null values in wise of row
    co2_data['Mn'] = co2_data['Mn'].apply(lambda x: '{0:0>2}'.format(x))     #Converting the month and date into a proper format
    co2_data['Dy'] = co2_data['Dy'].apply(lambda x: '{0:0>2}'.format(x))

    co2_data[[ 'Yr', 'Mn','Dy']] = co2_data[['Yr','Mn','Dy']].astype(str)   
    co2_data["DATE"] = co2_data["Yr"] + co2_data["Mn"] + co2_data["Dy"]     # Concatenate columns(year, month, day) into one single colums
    co2_data["DATE"] =  pd.to_datetime(co2_data["DATE"], format="%Y/%m/%d")
    co2_data.drop([ 'Yr', 'Mn','Dy'],axis=1,inplace=True)
    print("CO2 dataset after merging columns to form single date column:\n", co2_data.head(10))

    print('CO2 data before filling missing values\n', co2_data.head(10))
    co2_data.set_index('DATE', inplace=True)
    co2_data = co2_data.resample('D').ffill().reset_index()
    print('CO2 data after filling missing values\n', co2_data.head(10))

    co2_data.head(10)
    print("\n\n\nTotal no of NA's in the Co2 dataset :",co2_data.isna().sum().sum())
    return co2_data

@st.cache
def load_solar_radiation_data():
    print("Inside tsr_data")
    years = [str(x) for x in range(1940, 2022)]
    tsr_datasets = []
    for year in years:
        tsr_datasets.append(nc.Dataset("../datasets/Total_Solar_Irradiation/{0}.nc".format(year)))

    tsr_data_map = {}
    for tsr_data in tsr_datasets:
        times = tsr_data.variables["time"]
        tsi = tsr_data.variables["TSI"]
        dates = nc.num2date(times[:], times.units,calendar='standard',only_use_cftime_datetimes=False,only_use_python_datetimes=True)
        for idx in range(len(dates)):
            tsr_data_map[pd.to_datetime(dates[idx])] = np.ma.getdata(tsi[nc.date2index(dates[idx],times,select='nearest')])

    pd_tsr_data = pd.DataFrame(tsr_data_map.items(), columns=["DATE", "TSI"])
    pd_tsr_data['TSI'] = pd_tsr_data['TSI'].astype(int)   
    pd_tsr_data["DATE"] = pd.to_datetime(pd_tsr_data["DATE"])

    return pd_tsr_data

def integrate_data_frames(pd_tsr_data, weather_data, co2_data):
    print(weather_data)
    print(co2_data)
    #4. Combine all pandas dataframe into a single dataframe by using the date as the index.
    dataframes_to_be_combined = [pd_tsr_data, weather_data, co2_data]
    weather = reduce(lambda  left,right: pd.merge(left,right,on=['DATE'],how='outer'), dataframes_to_be_combined)
    print(weather.head(10))
    print("\nTotal no of NA's in the weather data is:",weather.isna().sum().sum())

    weather=weather.dropna(axis=0)
    print("\nWeather data after dropping NaNs")
    weather= weather.reset_index()
    weather.drop(['index'],axis=1,inplace=True)
    weather.head(10)
    return weather

def add_tavg_column(weather):
    print("Weather data before adding TAVG column\n", weather.head(10))
    weather['TAVG'] = (weather['TMIN'] + weather['TMAX']) / 2
    print("Weather data after adding TAVG column\n", weather.head(10))
    return weather

def add_season_column(weather):
    weather['SEASON'] = np.where((weather['DATE'].dt.month > 2) & (weather['DATE'].dt.month < 6), 1, 0)
    weather['SEASON'] = np.where((weather['DATE'].dt.month > 5) & (weather['DATE'].dt.month < 9), 2, weather['SEASON'])
    weather['SEASON'] = np.where((weather['DATE'].dt.month > 9) & (weather['DATE'].dt.month < 12), 3, weather['SEASON'])
    weather['SEASON'] = np.where((weather['DATE'].dt.month > 11) | (weather['DATE'].dt.month < 3), 4, weather['SEASON'])

    weather['Month'] = weather['DATE'].dt.month
    weather['Year'] = weather['DATE'].dt.year
    weather['Day'] = weather['DATE'].dt.day
    weather.head(20)
    return weather

def normalize_data(weather):
    """<h3>Normalizing CO2 and TSI values for use in modelling. </h3>"""

    weather_for_model =  weather.copy()
    weather_for_model['CO2'] = weather_for_model['CO2']/1000
    weather_for_model['TSI'] = weather_for_model['TSI']/10000
    print(weather_for_model.head(10))
    return weather_for_model

def get_lagged_values(dataframe, lag):
    dataframe_copy = dataframe.copy()
    for i in range(1,lag):
        dataframe_copy['lag'+str(i)] = dataframe.shift(i) 
    return dataframe_copy

def data_transformation(train,test):
    scaler = MinMaxScaler()
    train_scaled = scaler.fit_transform(train)
    test_tract1_scaled = scaler.fit_transform(test)          
    train_scaled_df = pd.DataFrame(train_scaled, index = train.index, columns=[train.columns[0]])
    test_scaled_df = pd.DataFrame(test_tract1_scaled, index = test.index, columns=[test.columns[0]])
    return train_scaled_df, test_scaled_df, scaler

def create_train_test(train,test):
    X_train = train.dropna().drop(train.columns[0], axis=1).values
    Y_train = train.dropna()[train.columns[0]].values
    X_test = test.dropna().drop(test.columns[0], axis=1).values
    Y_test = test.dropna()[test.columns[0]].values    
    return X_train, Y_train, X_test, Y_test