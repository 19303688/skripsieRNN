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
import kerastuner as kt


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
# print("X_train.shape")
# print(X_train.shape)
# print("X_test.shape")
# print(X_test.shape)
# print("Y_train.shape")
# print(Y_train.shape)    
# print("Y_test.shape")

# Convert the input numpy arrays to float32
class RNNModel:
    def __init__(self):
        self.n_hidden_units_per_layer = None
        self.learning_rate = None
        self.eta = None
        self.loss = None
        self.metrics = None
        self.epochs = None
        self.batch_size = None
        self.decay_rate = None
        self.model = None
        self.history = None
        self.n_encoder_layers = None
        self.n_decoder_layers = None

    def set_hyperparameters(
                self, n_hidden_units_per_layer, learning_rate, eta, loss, metrics, epochs, batch_size,
                decay_rate, n_encoder_layers, n_decoder_layers):
                self.n_hidden_units_per_layer = n_hidden_units_per_layer
                self.learning_rate = learning_rate
                self.eta = eta
                self.loss = loss
                self.metrics = metrics
                self.epochs = epochs
                self.batch_size = batch_size
                self.decay_rate = decay_rate
                self.n_encoder_layers = n_encoder_layers
                self.n_decoder_layers = n_decoder_layers         

    def build_model(self, X_shape, Y_shape):
        encoder_inputs = keras.layers.Input(shape=(None, X_shape[2]))

        encoder_outputs = []
        for i in range(self.n_encoder_layers):
            if i == 0:
                gru_layer = keras.layers.GRU(self.n_hidden_units_per_layer, return_sequences=True,
                                            return_state=True, activation='relu',
                                            kernel_initializer=keras.initializers.glorot_normal(seed=2),
                                            activity_regularizer=keras.regularizers.l2(self.eta))
                outputs, state_h = gru_layer(encoder_inputs)
            else:
                gru_layer = keras.layers.GRU(self.n_hidden_units_per_layer, return_sequences=True,
                                            return_state=True, activation='relu',
                                            kernel_initializer=keras.initializers.glorot_normal(seed=2),
                                            activity_regularizer=keras.regularizers.l2(self.eta))
                outputs, state_h = gru_layer(encoder_outputs[-1])
            encoder_outputs.append(outputs)

        encoder_states = [state_h]

        decoder_inputs = keras.layers.Input(shape=(None, Y_shape[2]))
        decoder_outputs = []

        for i in range(self.n_decoder_layers):
            if i == 0:
                gru_layer = keras.layers.GRU(self.n_hidden_units_per_layer, return_sequences=True,
                                            return_state=True, activation='relu',
                                            kernel_initializer=keras.initializers.glorot_normal(seed=2),
                                            activity_regularizer=keras.regularizers.l2(self.eta))
                outputs, _ = gru_layer(decoder_inputs, initial_state=encoder_states)
            else:
                gru_layer = keras.layers.GRU(self.n_hidden_units_per_layer, return_sequences=True,
                                            return_state=True, activation='relu',
                                            kernel_initializer=keras.initializers.glorot_normal(seed=2),
                                            activity_regularizer=keras.regularizers.l2(self.eta))
                outputs, _ = gru_layer(decoder_outputs[-1], initial_state=encoder_states)

            decoder_outputs.append(outputs)

        decoder_dense = keras.layers.TimeDistributed(keras.layers.Dense(Y_shape[2],
                                                                        kernel_initializer=keras.initializers.glorot_normal(
                                                                            seed=2),
                                                                        activity_regularizer=keras.regularizers.l2(
                                                                            self.eta)))
        decoder_outputs = decoder_dense(outputs)

        self.model = keras.Model(inputs=[encoder_inputs, decoder_inputs], outputs=decoder_outputs)

    def compile_model(self):
        optimizer = keras.optimizers.Adam(lr=self.learning_rate, beta_1=0.9, beta_2=0.999,
                                          epsilon=1E-8, decay=self.decay_rate)
        self.model.compile(loss=self.loss, optimizer=optimizer, metrics=[self.metrics])

    def train(self, X, Y):
        self.history = self.model.fit([X, Y], Y, epochs=self.epochs, batch_size=self.batch_size,
                                        validation_split=0.2,
                                        shuffle=False)

    def get_model(self):
        return self.model
    
    def print_model_summary(self):
        self.model.summary()

    def get_history(self):
        return self.history
    
model = RNNModel()

model.set_hyperparameters(n_hidden_units_per_layer=64,  # Example with 2 layers
                          learning_rate=0.001,
                          eta=0.01,
                          loss='mse',
                          epochs=10,
                          batch_size=32,
                          decay_rate=0.0,
                          n_encoder_layers=2,
                          n_decoder_layers=5,
                          metrics=['mae', tf.keras.metrics.MeanAbsolutePercentageError(name="mape"), tf.keras.metrics.RootMeanSquaredError(name="rmse")])

model.build_model(X_train.shape, Y_train.shape)
model.compile_model()

print("Model Summary is: ")
model.print_model_summary()

model.train(X_train, Y_train)
trained_model = model.get_model()
history = model.get_history()

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'], color="red")
plt.title('Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.show()

plt.plot(history.history['mape'])
plt.plot(history.history['val_mape'], color="red")
plt.title('Mape')
plt.xlabel('Epochs')
plt.ylabel('MAPE %')
plt.show()

plt.plot(history.history['rmse'])
plt.plot(history.history['val_rmse'], color="red")
plt.title('RMSE')
plt.xlabel('Epochs')
plt.ylabel('RMSE %')
plt.show()