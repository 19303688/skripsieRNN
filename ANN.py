import sys

import tensorflow.keras
import pandas as pd
import numpy as np
import sklearn as sk
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import platform
from sklearn.model_selection import train_test_split
from scipy.ndimage.filters import gaussian_filter1d
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import matplotlib.pyplot as plt
import os

print(f"Python Platform: {platform.platform()}")
print(f"Tensor Flow Version: {tf.__version__}")
print(f"Keras Version: {tensorflow.keras.__version__}")
print()
print(f"Python {sys.version}")
print(f"Pandas {pd.__version__}")
print(f"Scikit-Learn {sk.__version__}")
gpu = len(tf.config.list_physical_devices('GPU'))>0
print("GPU is", "available" if gpu else "NOT AVAILABLE")

#Load data
df = pd.read_csv('data_cleaned_R3.csv')

df.loc[df["Coal Input"] < 30] = 0

timeBackWindow, timeForwardWindow = 5, 2

df_scaled = df

#TODO Convert the hard code 8 to a stats term?
df_scaled['Power Plant On'] = (df_scaled['Coal Input'] >= 30).astype(int)

dataframes = []
indices = []
for index, row in df_scaled.iterrows():
    if row['Power Plant On'] == 0:
        if indices:
            dataframes.append(df_scaled.loc[indices].copy())
            indices = []
    else:
        indices.append(index)

# Check if there are any remaining indices after the loop ends
if indices:
    dataframes.append(df_scaled.loc[indices].copy())

# Calculate the total number of rows
total_rows= sum(len(dataset) for dataset in dataframes)



def timewindow_df(df, timeBackWindow, timeForwardWindow):

    # Define column names for the first part
    cols_part1 = ['ECO SYS FW Supply SATN T', 'Economizer SYS FW Supply P', 'Live Steam TOT F',
                  'AMB Air T', 'Coal Input', 'BLR Primary Air F', 'BLR Secondary Air Total F',
                  'OVR FIR CORR Air NOZ 71-73 F', 'OVR FIR CORR Air NOZ 74-76 F',
                  'OVR FIR CORR Air NOZ 81-83 F', 'OVR FIR CORR Air NOZ 84-86 F']
    # Define column names for the second part
    cols_part2 = ['ATT 1 m_dot', 'ATT 2 m_dot', 'R/H ATT m_dot', 'HP Steam Average Temp', 'Hot R/H Average Temp']
    # Shift rows and concatenate for both parts
    df_temp = pd.DataFrame()

    for col in cols_part1 + cols_part2:
        # Shift columns forward for the first part
        if col in cols_part1:
            df_new = pd.concat([df[col].shift(-i) for i in range(timeBackWindow-1, -1, -1)], axis=1, 
                               keys=[f'{col}-{i}' for i in range(timeBackWindow)])
        # Shift columns backward for the second part
        else:
            shifted_cols = [df[col].shift(i) for i in range(timeForwardWindow)]
            df_new = pd.concat(shifted_cols, axis=1, 
                               keys=[f'{col}+{i}' for i in range(timeForwardWindow)])
        # Concatenate the resulting dataframe with df_temp, along the columns axis (axis=1)
        df_temp = pd.concat([df_temp, df_new], axis=1)

    return df_temp

#Time window the datasets
manipulated_dataframes = []

#---------------------------------------------------#
for df in dataframes:
    manipulated_df = timewindow_df(df,timeBackWindow=timeBackWindow,timeForwardWindow=timeForwardWindow)
    manipulated_dataframes.append(manipulated_df)
#---------------------------------------------------#

# Print the number of rows for each dataset in the manipulated dataframes
print(f"Number of manipulated datasets: {len(manipulated_dataframes)}")
#for i, dataset in enumerate(manipulated_dataframes):
    #print(f"Manipulated Dataset {i+1} length: {len(dataset)}")

# Calculate the total number of rows across all the manipulated datasets
total_rows = sum(len(dataset) for dataset in manipulated_dataframes)

# Remove rows with NaN values from each dataframe
manipulated_dataframes_cleaned = []
for dataset in manipulated_dataframes:
    cleaned_dataset = dataset.dropna()
    manipulated_dataframes_cleaned.append(cleaned_dataset)

total_rows = sum(len(dataset) for dataset in manipulated_dataframes_cleaned)

processed_df = pd.concat(manipulated_dataframes_cleaned, ignore_index=True)

#Remove rows that contain zeros
initial_row_count = len(processed_df)
processed_df = processed_df[(processed_df != 0).all(axis=1)]
rows_removed = initial_row_count - len(processed_df)
rows_remaining = len(processed_df)

temp_df = processed_df.copy()

scaler = MinMaxScaler()
scaled_df = scaler.fit_transform(processed_df)

# Convert the scaled data back to a DataFrame
processed_df = pd.DataFrame(scaled_df, columns=processed_df.columns)

x_cols = timeBackWindow * 11  # Adjust this 11 to be the number of columns in the x part of the dataset
x_data = processed_df.iloc[:, :x_cols].values
y_data = processed_df.iloc[:, x_cols:].values

#Test/Train Split
train_perc = 0.9

x_train = x_data[:int(train_perc*x_data.shape[0]),:]
x_test = x_data[int(train_perc*x_data.shape[0]):,:]

y_train = y_data[:int(train_perc*y_data.shape[0]),:]
y_test = y_data[int(train_perc*y_data.shape[0]):,:]


class ANN(keras.Model):
    def __init__(self):
        super(ANN, self).__init__()
        self.input_layer = keras.layers.Dense(units=64, input_shape=(x_train.shape[0],), activation='relu')
        self.hidden_layers = [keras.layers.Dense(units=224, activation='relu'),
                              keras.layers.Dense(units=160, activation='relu')]
        self.output_layer = keras.layers.Dense(units=10, activation='linear')

    def call(self, inputs):
        x = self.input_layer(inputs)
        for layer in self.hidden_layers:
            x = layer(x)
        output = self.output_layer(x)
        return output

model = ANN()

# loss_fn = tf.keras.losses.MeanSquaredError()
loss_fn = keras.losses.MeanAbsoluteError()
optimizer = tf.keras.optimizers.Adam(learning_rate=0.0005058857837580419)
model.compile(optimizer=optimizer, 
              loss=loss_fn,
              metrics=[tf.keras.metrics.MeanAbsolutePercentageError(), 
                       tf.keras.metrics.RootMeanSquaredError()])

# Build the model before training
model.build((None, 55))

print(model.summary())

# Define EarlyStopping callback
early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

hist = model.fit(x_train, y_train, epochs=500, batch_size=16, validation_split=0.2, callbacks=[early_stopping])
#hist = model.fit(x_train, y_train, epochs=50, batch_size=16, validation_split=0.2)

loss, MAPE, RMSE = model.evaluate(x_test, y_test)
forecast = model.predict(x_test)

#Add blank columns to forecast to match the shape
blank_array = np.empty((2236, 55))
forecast_shape = np.concatenate((blank_array, forecast), axis=1)
pred_original_scaled = scaler.inverse_transform(forecast_shape)

#headings
col_headings = ["ATT 1 m_dot+0","ATT 1 m_dot+1","ATT 2 m_dot+0","ATT 2 m_dot+1","R/H ATT m_dot+0","R/H ATT m_dot+1","HP Steam Average Temp+0","HP Steam Average Temp+1","Hot R/H Average Temp+0","Hot R/H Average Temp+1"]
for col in range(55,65):
    column_pred = pred_original_scaled[:, col]  # Assuming column 56 corresponds to index 55
    column_temp_last_2236 = temp_df.iloc[-2236:, col].values  # Extract last 2236 entries of column 56

    # Create the plot
    plt.figure(figsize=(12, 9))
    plt.plot(column_temp_last_2236, label='Raw')
    plt.plot(column_pred, label='Predicted')

    # Add labels and legend
    plt.xlabel('Data Point')
    plt.ylabel('Value')
    plt.legend()
    plt.title(f"Column {col_headings[col-55]}")
    # Show the plot
    plt.show()


    # Plot the training loss
plt.plot(hist.history['loss'])
plt.plot(hist.history['val_loss'], color="red")
plt.title('Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.show()

# Plot the training metrics
plt.plot(hist.history['mean_absolute_percentage_error'])
plt.plot(hist.history['val_mean_absolute_percentage_error'], color="red")
plt.title('Mean Absolute Percentage Error')
plt.xlabel('Epochs')
plt.ylabel('MAPE')
plt.show()

# Plot the MAPE without training MAPE
plt.plot(hist.history['val_mean_absolute_percentage_error'], color="red")
plt.title('Validation Absolute Percentage Error')
plt.xlabel('Epochs')
plt.ylabel('MAPE')
plt.show()

# Plot the training metrics
plt.plot(hist.history['root_mean_squared_error'])
plt.plot(hist.history['val_root_mean_squared_error'], color="red")
plt.title('Root Mean Squared Error')
plt.xlabel('Epochs')
plt.ylabel('RMSE')
plt.show()