import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import LSTM
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from keras.layers import Dense




#Opens the file for analysis
data = pd.read_csv (r'C:\Users\meht1\OneDrive\Desktop\MATH3599\Index_Analysis\IndexFutures\IndexFutures\ContractFutures_1min_uic4039.csv')

# This function creates a dataframe that stores the previous 3 values of close and matches it with the current
# target value
def df_to_windowed_df(df, numOfLags, length):
  rdf = df.copy()
  print("Data Frame Table")
  print(rdf)
  
  #Sets column names
  rdf = rdf.rename(columns = {"Close" : "Target"})
  
  for x in range(1,numOfLags+1):
      rdf["Target-"+ str(x)] = np.nan
      for z in range(length):
           rdf["Target-"+ str(x)] = rdf["Target"].shift(x)
  
  for x in range(numOfLags):
     rdf =  rdf.drop(df.index[x])
     
  #HARD CODED REARANGEMENT OF THE COLUMNS MUST CHANGE IF NUMBER OF LAGS IS CHANGED   
  rdf = rdf[['Target-3', 'Target-2', 'Target-1', 'Target']]
  print("Targets Table")
  print(rdf)       
  return rdf


def windowed_df_to_date_X_y(windowed_dataframe):
  df_as_np = windowed_dataframe
  df_as_np = df_as_np.reset_index()
  df_as_np = df_as_np.to_numpy()

  dates = df_as_np[:, 0]

  middle_matrix = df_as_np[:, 1:-1]
  X = middle_matrix.reshape((len(dates), middle_matrix.shape[1], 1))

  Y = df_as_np[:, -1]
  
  return dates, X.astype(np.float32), Y.astype(np.float32)




def main(start,end):
    # Creates a copy of the data for modification
    df = data.copy()
    df = df.iloc[start:end,:]
    #Sets df to only have two columns for the timeseries Time and Close
    df = df[['Time', 'Close']]
    #Alters the Time column from a string to datetime and makes the date the index
    df['Time'] = pd.to_datetime(df["Time"], errors = 'ignore')
    df.index = df.pop('Time')
    df.index = df.index.tz_convert('UTC')
    length = int(len(df))
    
    #Plots a line graph of close price
    plt.plot(df.index, df['Close'], color = "black")
    plt.rcParams['figure.figsize'] = (15,8)
    plt.xlabel("Date-Time")
    plt.ylabel("Close Price ($)")
    plt.title("Plot of Close Price")
    plt.show()
    plt.close()
    
    #Creates a Windowed Dataframe of the dates
    rdf = df_to_windowed_df(df, 3, length)
    #Splits the data into Training, Testing and Validation
    dates, X, y = windowed_df_to_date_X_y(rdf)
    #print("Showcasing the size of dates, x and y", dates.shape, X.shape, y.shape)
    q_80 = int(len(dates) * .8)
    q_90 = int(len(dates) * .9)
    dates_train, X_train, y_train = dates[:q_80], X[:q_80], y[:q_80]
    dates_val, X_val, y_val = dates[q_80:q_90], X[q_80:q_90], y[q_80:q_90]
    dates_test, X_test, y_test = dates[q_90:], X[q_90:], y[q_90:]
    
   
    #Plots the Training, Validation and Test Split
    plt.plot(dates_train, y_train)
    plt.plot(dates_val, y_val)
    plt.plot(dates_test, y_test)
    plt.rcParams['figure.figsize'] = (15,8)
    plt.legend(['Train', 'Validation', 'Test'])
    plt.xlabel("Date/Time")
    plt.ylabel("Price ($)")
    plt.title("Plot of Observations for Training, Testing and Validation")
    plt.show()
    plt.close()
    
    
    #Creats the LSTM Model 
    model = Sequential()
    model.add(LSTM(100, input_shape=(X_train.shape[1], X_train.shape[2]), activation='relu'))
    model.add(Dense(1))
    model.compile(loss='mse',  optimizer=Adam(learning_rate=0.001), metrics=['mean_absolute_error'])
    # Notes: Epochs = number of iterations, 
    model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=350)
   
    
    loss_per_epoch = model.history.history['val_mean_absolute_error']
    plt.plot(range(len(loss_per_epoch)),loss_per_epoch)
    plt.xlabel("EPOCHS")
    plt.ylabel("Mean Absolute Error")
    plt.title("MAE for model fitting")
    plt.show()
    plt.close()
   
    #Plots for Observations vs Predictions
    train_predictions = model.predict(X_train).flatten()
    plt.plot(dates_train, y_train)
    plt.plot(dates_train, train_predictions, label = '--')
    plt.rcParams['figure.figsize'] = (15,8)
    plt.title("Training Observations vs Training Predictions")
    plt.xlabel("Date/Time")
    plt.ylabel("Price ($)")
    plt.legend(['Training Predictions', 'Training Observations'])
    plt.show()
    plt.close()
    
    val_predictions = model.predict(X_val).flatten()
    plt.title("Validation Observations vs Validation Predicitons")
    plt.plot(dates_val, y_val)
    plt.plot(dates_val, val_predictions, label = '--')
    plt.legend(['Validation Predictions', 'Validation Observations'])
    plt.show()
    plt.close()
    
    test_predictions = model.predict(X_test).flatten()
    plt.plot(dates_test, test_predictions)
    plt.title("Test Observations vs Test Predicitons")
    plt.plot(dates_test, y_test)
    plt.legend(['Testing Predictions', 'Testing Observations'])
    plt.show()
    plt.close()
    
    plt.plot(dates_train, y_train)
    plt.plot(dates_train, train_predictions, linestyle = '--')
    plt.plot(dates_val, val_predictions)
    plt.title("Observations vs Predicitons")
    plt.plot(dates_val, y_val)
    plt.plot(dates_test, test_predictions)
    plt.plot(dates_test, y_test)
    plt.legend(['Training Predictions', 
                'Training Observations',
                'Validation Predictions', 
                'Validation Observations',
                'Testing Predictions', 
                'Testing Observations'])
    plt.show()
    plt.close()
    
    OvP = pd.DataFrame()
    for x in range(len(dates_test)):
        OvP = OvP.append({"Time": dates_test[x], "Observations":y_train[x], "Predictions":test_predictions[x]}, ignore_index = True)
    print(OvP.head())
    print(OvP.tail())
    

main(0,10000)
