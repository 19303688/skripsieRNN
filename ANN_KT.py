import sys

import tensorflow.keras
import keras_tuner as kt
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

#TODO Convert plot graph of why 30 was chosen.
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

for df in dataframes:
    manipulated_df = timewindow_df(df,timeBackWindow=timeBackWindow,timeForwardWindow=timeForwardWindow)
    manipulated_dataframes.append(manipulated_df)


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
    def __init__(self, hp):
        super(ANN, self).__init__()
        self.input_layer = keras.layers.Dense(units=64, 
                                              input_shape=(x_train.shape[0],),
                                               activation='relu')
        self.hidden_layers = []
        self.output_layer = keras.layers.Dense(units=10, activation='linear')

        # Define the hyperparameters for the hidden layers
        for i in range(hp.Int('num_layers', 1, 4)):
            self.hidden_layers.append(
                keras.layers.Dense(units=hp.Int('units_' + str(i), 32, 256, step=32),
                                    activation='relu')
            )

    def call(self, inputs):
        x = self.input_layer(inputs)
        for layer in self.hidden_layers:
            x = layer(x)
        output = self.output_layer(x)
        return output

# Define the hypermodel
class ANNHyperModel(kt.HyperModel):
    def build(self, hp):
        model = ANN(hp)
        # loss_fn = keras.losses.MeanSquaredError()
        # The choice of loss function can affect how the model optimizes its parameters during training. 
        # If the loss function is not appropriate for capturing volatility, such as mean squared error (MSE) 
        # that tends to penalize larger errors more than smaller errors, the model may prioritize minimizing the 
        # overall error rather than capturing the specific volatility patterns. In such cases, 
        # alternative loss functions like mean absolute error (MAE) or Huber loss, which are less sensitive to outliers, may be more suitable.
        loss_fn = keras.losses.MeanAbsoluteError()
        lr = hp.Float("lr", min_value=0.000001, max_value=0.001, sampling="log")
        optimizer = keras.optimizers.Adam(learning_rate=lr)
        model.compile(optimizer=optimizer, loss=loss_fn,
                    metrics=[keras.metrics.MeanAbsolutePercentageError(),
                                keras.metrics.RootMeanSquaredError()])
        return model

    def fit(self, hp, model, *args, **kwargs):
            return model.fit(
                *args,
                batch_size=hp.Choice("batch_size", [16, 32, 64, 128, 256]),
                **kwargs,
            )

tuner = kt.Hyperband(ANNHyperModel(),
                     objective='val_loss',
                     directory='keras_tuner',
                     project_name='ann_kt_mae')

stop_early = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5)
tuner.search(x_train, y_train, epochs=100, 
             validation_split=0.2, 
              callbacks=[stop_early])

# Get the best hyperparameters
best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]

print(f"""
The hyperparameter search is complete. 
The optimal number of layers = {best_hps.get('num_layers')}.
and the optimal learning rate for the optimizer is {best_hps.get('lr')}. 
Batch size = {best_hps.get('batch_size')}.
""")

num_layers = best_hps.get('num_layers')
for i in range(num_layers):
    print(f"Layer {i+1}: Units: {best_hps.get('units_' + str(i))}")


# class MyHyperModel(kt.HyperModel):
#     def build(self, hp):
    
#         model = keras.Sequential()
#         model.add(keras.layers.Dense(units=64, input_shape=(x_train.shape[0],), activation='relu'))
#         for layers in range(hp.Int('num_layers', min_value = 2, max_value = 4, step = 1)):
#             # Choose an optimal value between 32-512
#             hp_units = hp.Choice('units', values=[32, 64, 128, 256, 512])
#             model.add(keras.layers.Dense(units=hp_units, activation='relu'))

#         # Output layer
#         model.add(keras.layers.Dense(units=10, activation='linear'))

#         # Tune the learning rate for the optimizer
#         hp_learning_rate = hp.Float("lr", min_value=0.000001, max_value=0.001, sampling="log")
#         loss_fn = tf.keras.losses.MeanSquaredError()
#         model.compile(optimizer=keras.optimizers.Adam(learning_rate=hp_learning_rate),
#                     loss=loss_fn,
#                     metrics=[tf.keras.metrics.MeanAbsolutePercentageError(),
#                             tf.keras.metrics.RootMeanSquaredError()])

#         return model
    
#     def fit(self, hp, model, *args, **kwargs):
#         return model.fit(
#             *args,
#             batch_size=hp.Choice("batch_size", [16, 32, 64, 128, 256]),
#             **kwargs,
#         )

# tuner = kt.Hyperband(MyHyperModel(),
#                      objective='val_loss',
#                      directory='keras_tuner',
#                      project_name='ann_kt')

# stop_early = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5)
# tuner.search(x_train, y_train, epochs=100, 
#              validation_split=0.2,
#               callbacks=[stop_early])

# best_hps=tuner.get_best_hyperparameters()[0]


# hypermodel = MyHyperModel()
# best_hp = tuner.get_best_hyperparameters()[0]
# model = hypermodel.build(best_hp)
# hypermodel.fit(best_hp, model, x_all, y_all, epochs=1)