import sys

#import tensorflow.keras
import pandas as pd
import numpy as np
import sklearn as sk
import tensorflow as tf
from tensorflow import keras
#from tensorflow.keras import layers
import platform
from sklearn.model_selection import train_test_split
from scipy.ndimage.filters import gaussian_filter1d
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import matplotlib.pyplot as plt
import kerastuner as kt
import os

print(f"Python Platform: {platform.platform()}")
print(f"Tensor Flow Version: {tf.__version__}")
print(f"Keras Version: {keras.__version__}")
print()
print(f"Python {sys.version}")
print(f"Pandas {pd.__version__}")
print(f"Scikit-Learn {sk.__version__}")
gpu = len(tf.config.list_physical_devices('GPU'))>0
print("GPU is", "available" if gpu else "NOT AVAILABLE")
#Load data
df = pd.read_csv('data_cleaned_R3.csv')

df.loc[df["Coal Input"] < 30] = 0

df = df.dropna()

#TODO Convert to a stats term?
df['Power Plant On'] = (df['Coal Input'] >= 30).astype(int)

inputs = df.iloc[:, :11].copy()

# Copy the next 5 rows into outputs
outputs = df.iloc[:, 11:16].copy()

off_index = df.iloc[:, 16].copy()


def create_3d_array(dataframe, time_window):
    # Create a new 3-dimensional array with the same dimensions as the dataframe
    array_3d = np.zeros((dataframe.shape[0], dataframe.shape[1], time_window))
    
    # Iterate over each column in the dataframe
    for col_idx, col in enumerate(dataframe.columns):
        # Get the values of the current column
        current_col = dataframe[col].values
        
        # Iterate over each time step
        for t in range(time_window):
            # Create a shifted version of the column
            shifted_col = np.roll(current_col, t) if t < len(current_col) else np.nan
            
            # Assign the shifted column values to the corresponding positions in the 3D array
            array_3d[:, col_idx, t] = shifted_col
    
    return array_3d

outputs_3d = create_3d_array(outputs, 2)
inputs_3d = create_3d_array(inputs, 5)
print(inputs_3d.shape)
#print("inputs_3d.shape")
print(outputs_3d.shape)
#print("outputs_3d.shape")

# Filter out when the plant is off.
mask  = off_index[:] != 0
#print(mask)
on_inputs = inputs_3d[mask]
on_outputs = outputs_3d[mask]
#print("on_inputs.shape")
#print(on_inputs.shape)
#print("on_outputs.shape")
#print(on_outputs.shape)

#add zeros to the output array before the scaling
padded_array = np.pad(on_outputs, ((0,0),(0,6),(0,3)), 'constant')
# print("padded_array.shape")
# print(padded_array.shape)

#Scale the data

scaler = MinMaxScaler()
reshaped_inputs = on_inputs.reshape(-1, on_inputs.shape[-1])

scaler.fit(reshaped_inputs)
scaled_inputs = scaler.transform(reshaped_inputs)
scaled_inputs = scaled_inputs.reshape(on_inputs.shape)
# print("scaled_inputs.shape")
# print(scaled_inputs.shape)

reshaped_padded_array = padded_array.reshape(-1, padded_array.shape[-1])
scaled_outputs = scaler.transform(reshaped_padded_array)
scaled_outputs = scaled_outputs.reshape(padded_array.shape)
# print("scaled_outputs.shape")
# print(scaled_outputs.shape)

scaled_outputs = scaled_outputs[:on_outputs.shape[0], :on_outputs.shape[1], :on_outputs.shape[2]]


# Remove the added dimensions
scaled_outputs = scaled_outputs[:on_outputs.shape[0], :on_outputs.shape[1], :on_outputs.shape[2]]


# print("scaled_outputs.shape")
# print(scaled_outputs.shape)


#Split data into training and testing
X_train, X_test, Y_train, Y_test = train_test_split(scaled_inputs, scaled_outputs, test_size=0.1, shuffle=False)

class RNNModel(keras.Model):
    def __init__(self,hp):
        super(RNNModel, self).__init__()
        self.eta = hp.Float('eta', min_value=1e-4, max_value=1e-2, sampling='LOG', default=1e-3)
        self.decay = hp.Float('decay_rate', min_value=1e-4, max_value=1e-2, sampling='LOG', default=1e-3)
        # self.eta = 1e-3
        # self.decay = 1e-3
        encoder_inputs = keras.layers.Input(shape=(None, X_train.shape[2]))
        encoder_outputs = []
        for i in range(hp.Int('num_layers', 1, 3)):
            if i == 0:
                gru_layer = keras.layers.GRU(units=hp.Int('units_' + str(i), 32, 256, step=32), return_sequences=True,
                                            return_state=True, activation='relu',
                                            kernel_initializer=keras.initializers.glorot_normal(seed=2),
                                            activity_regularizer=keras.regularizers.l2(self.eta))
                outputs, state_h = gru_layer(encoder_inputs)
            else:
                gru_layer = keras.layers.GRU(units=hp.Int('units_' + str(i), 32, 256, step=32), return_sequences=True,
                                            return_state=True, activation='relu',
                                            kernel_initializer=keras.initializers.glorot_normal(seed=2),
                                            activity_regularizer=keras.regularizers.l2(self.eta))
                outputs, state_h = gru_layer(encoder_outputs[-1])
            encoder_outputs.append(outputs)
        encoder_states = [state_h]
        decoder_inputs = keras.layers.Input(shape=(None, Y_train.shape[2]))
        decoder_outputs = []
        for i in range(hp.Int('num_layers', 1, 3)):
            if i == 0:
                gru_layer = keras.layers.GRU(units=hp.Int('units_' + str(i), 32, 256, step=32), return_sequences=True,
                                            return_state=True, activation='relu',
                                            kernel_initializer=keras.initializers.glorot_normal(seed=2),
                                            activity_regularizer=keras.regularizers.l2(self.eta))
                outputs, _ = gru_layer(decoder_inputs, initial_state=encoder_states)
            else:
                gru_layer = keras.layers.GRU(units=hp.Int('units_' + str(i), 32, 256, step=32), return_sequences=True,
                                            return_state=True, activation='relu',
                                            kernel_initializer=keras.initializers.glorot_normal(seed=2),
                                            activity_regularizer=keras.regularizers.l2(self.eta))
                outputs, _ = gru_layer(decoder_outputs[-1], initial_state=encoder_states)

            decoder_outputs.append(outputs)
        decoder_dense = keras.layers.TimeDistributed(keras.layers.Dense(Y_train.shape[2],
                                                                        kernel_initializer=keras.initializers.glorot_normal(
                                                                            seed=2),
                                                                        activity_regularizer=keras.regularizers.l2(
                                                                            self.eta)))
        decoder_outputs = decoder_dense(outputs)
        self.model = keras.Model(inputs=[encoder_inputs, decoder_inputs], outputs=decoder_outputs)

    def call(self, inputs):
        return self.model(inputs)
    

class RNNHyperModel(keras.Model):
    def build(self,hp):
        model = RNNModel(hp)
        loss_fn = keras.losses.MeanSquaredError()
        lr = hp.Float('learning_rate', min_value=1e-4, max_value=1e-2, sampling='LOG', default=1e-3)
        eta = hp.Float('eta', min_value=1e-4, max_value=1e-2, sampling='LOG', default=1e-3)
        decay_rate = hp.Float('decay_rate', min_value=1e-4, max_value=1e-2, sampling='LOG', default=1e-3)
        
        optimizer = keras.optimizers.Adam(learning_rate=lr)
        model.compile(optimizer=optimizer, loss=loss_fn, metrics=['mae', tf.keras.metrics.MeanAbsolutePercentageError(name="mape"), tf.keras.metrics.RootMeanSquaredError(name="rmse")])
        return model

    def fit(self,hp,model,*args,**kwargs):
        return model.fit(*args,
                         batch_size=hp.Choice('batch_size', values=[16, 32, 64, 128, 256, 512]),
                         **kwargs)

tuner = kt.Hyperband(RNNHyperModel(),
                        objective='val_loss',
                        directory='RNN_Keras_Tuner',
                        project_name='RNN Tuner')

stop_early = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5)
tuner.search(X_train, Y_train, epochs=100, 
             validation_split=0.2, 
              callbacks=[stop_early])

best_hps=tuner.get_best_hyperparameters(num_trials=1)[0]
print(f"""
The hyperparameter search is complete. 
The optimal number of layers = {best_hps.get('num_layers')}.
The optimal learning rate for the optimizer is {best_hps.get('learning_rate')}. 
Batch size = {best_hps.get('batch_size')}.
""")

num_layers = best_hps.get('num_layers')
for i in range(num_layers):
    print(f"Layer {i+1}: Units: {best_hps.get('units_' + str(i))}")