import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import json
import keras
import matplotlib.pyplot as plt

#Read data:
rawdata_inputs = pd.read_csv("Input_Asset_2019_5_30.csv", decimal=",")
inputs = np.asmatrix(np.float32(rawdata_inputs.iloc[0:,1:]))
rawdata_outputs = pd.read_csv("Output_Asset_2019_5_30.csv", decimal=",")
outputs = np.asmatrix(np.float32(rawdata_outputs.iloc[0:,17:]))
outputs = np.delete(outputs,15,axis=1)

rawdata_inputs_Test = pd.read_csv("InputTest_Asset_2019_6_7.csv", decimal=",")
inputs_Test = np.asmatrix(np.float32(rawdata_inputs_Test.iloc[0:,1:]))
rawdata_outputs_Test = pd.read_csv("OutputTest_Asset_2019_6_7.csv", decimal=",")
outputs_Test = np.asmatrix(np.float32(rawdata_outputs_Test.iloc[0:,17:]))
outputs_Test = np.delete(outputs_Test,15,axis=1)


#Stack outputs to inputs:
stacked_inputs = np.hstack((inputs, outputs))
stacked_inputsTest = np.hstack((inputs_Test, outputs_Test))

#Data prep
def split_data(inputs, outputs, f_train=0.7, f_dev=0.2, f_test=0.1):
    train_size = int(len(inputs) * f_train)
    test_size = int(len(inputs) * f_test)
    dev_size = len(inputs) - train_size - test_size
    inputs_train, inputs_dev, inputs_test = inputs[0:train_size,:], inputs[train_size:(train_size+dev_size),:], inputs[(train_size+dev_size):,:]
    outputs_train, outputs_dev, outputs_test = outputs[0:train_size,:], outputs[train_size:(train_size+dev_size),:], outputs[(train_size+dev_size):,:]
    return inputs_train, inputs_dev, inputs_test, outputs_train, outputs_dev, outputs_test

def arrange_data(inputs, outputs, lagwindow = 1, leadwindow = 1):
    X, Y, X2 = [], [], []
    for t in range(len(inputs) - lagwindow - leadwindow - 1):
        drange_in = inputs[t:(t+lagwindow)]
        drange_out = outputs[(t+lagwindow):(t+lagwindow+leadwindow)]
        drange_out2 = np.concatenate((np.zeros((1,drange_out.shape[1])),drange_out[:-1,:]), axis=0)
        X.append(drange_in)
        Y.append(drange_out)
        X2.append(drange_out2)
    X = np.array(X)
    Y = np.array(Y)
    X2 = np.array(X2)
    return X, Y, X2

#Define machine learning model architecture
def model_setup(X, Y, devX, devY, n_hidden_units_per_layer = 10, 
                learning_rate = 0.001,
                eta = 1E-8, loss = 'mean_squared_error', metrics = 'mae', 
                epochs = 50, batch_size = 64, decay_rate = 1E-6, n_encoder_depth = 1,
                n_decoder_depth = 1, n_int_layer_depth = 1):
    
    encoder_inputs = keras.layers.Input(shape=(None, X.shape[2]))
    encoder = keras.layers.LSTM(n_hidden_units_per_layer, return_state=True, 
                                activation='relu' , 
                                kernel_initializer=keras.initializers.glorot_normal(seed=2), 
                                activity_regularizer=keras.regularizers.l2(eta))
    encoder_outputs, state_h, state_c = encoder(encoder_inputs)
    encoder_states = [state_h, state_c]

    decoder_inputs = keras.layers.Input(shape=(None, Y.shape[2]))
    decoder_lstm = keras.layers.LSTM(n_hidden_units_per_layer, return_sequences=True,
                                    return_state=True, activation='relu', 
                                    kernel_initializer=keras.initializers.glorot_normal(seed=2), 
                                    activity_regularizer=keras.regularizers.l2(eta))
    decoder_outputs, _, _ = decoder_lstm(decoder_inputs, initial_state=encoder_states)
    decoder_dense = keras.layers.TimeDistributed(keras.layers.Dense(Y.shape[2], 
                                        kernel_initializer=keras.initializers.glorot_normal(seed=2), 
                                        activity_regularizer=keras.regularizers.l2(eta)))
    decoder_outputs = decoder_dense(decoder_outputs)

    model = keras.Model(inputs=[encoder_inputs, decoder_inputs], outputs=decoder_outputs)

    optimizer = keras.optimizers.Adam(lr = learning_rate, beta_1=0.9, beta_2=0.999, 
                                      epsilon=1E-8, decay = decay_rate)
    model.compile(loss = loss, optimizer=optimizer, metrics=[metrics])
    history = model.fit([X,Y], Y, epochs=epochs, batch_size=batch_size, verbose=2, 
                        shuffle=False)
    
    return model, history

#Split data sets:
inputs_train, inputs_dev, inputs_test, outputs_train, outputs_dev, outputs_test = split_data(stacked_inputs, outputs, f_train=0.7, f_dev=0.2, f_test=0.1)

#Normalize data:
scaler_inputs = MinMaxScaler()
scaler_outputs = MinMaxScaler()

inputs_train_scaled = scaler_inputs.fit_transform(inputs_train)
outputs_train_scaled = np.asmatrix(scaler_outputs.fit_transform(outputs_train))
inputs_dev_scaled = scaler_inputs.transform(inputs_dev)
outputs_dev_scaled = np.asmatrix(scaler_outputs.transform(outputs_dev))
inputs_test_scaled = scaler_inputs.transform(inputs_test)
outputs_test_scaled = np.asmatrix(scaler_outputs.transform(outputs_test))

TestIn = scaler_inputs.transform(stacked_inputsTest)
TestOut = np.asmatrix(scaler_outputs.transform(outputs_Test))

print(inputs_train_scaled.shape)
print(inputs_dev_scaled.shape)
print(TestIn.shape)

#Set lag window size:
lagwindow = 8
leadwindow = 5

#Rearrange datasets into LSTM format:
X, Y, X2 = arrange_data(inputs_train_scaled, outputs_train_scaled, lagwindow=lagwindow, leadwindow=leadwindow)
Xdev, Ydev, Xdev2 = arrange_data(inputs_dev_scaled, outputs_dev_scaled, lagwindow=lagwindow, leadwindow=leadwindow)
Xtest, Ytest, Xtest2 = arrange_data(inputs_test_scaled, outputs_test_scaled, lagwindow=lagwindow, leadwindow=leadwindow)

FinTestX, FinTestY, FinTestX2 = arrange_data(TestIn, TestOut, lagwindow=lagwindow, leadwindow=leadwindow)

print(X.shape, Y.shape, X2.shape)

#Specify number of hidden units in LSTM:
n_a = 512
#Specify number of the outputs:
n_fc = Y.shape[1]

#Train model:
model, history = model_setup(X, Y, Xdev, Ydev, n_hidden_units_per_layer=n_a,
                            epochs=1, n_encoder_depth=2, n_decoder_depth=2,
                            learning_rate = 0.001, eta=1E-8, batch_size=32)

#Plot training error history:
plt.plot(history.history['loss'], label='train')
plt.legend()
plt.show()

#Model performance:
yhat_train = model.predict([X,Y])
yhat_dev = model.predict([Xdev,Ydev])
yhat_test = model.predict([Xtest,Ytest])

yhatTrain_rescale = yhat_train * scaler_outputs.data_range_ + scaler_outputs.data_min_
Y_rescale = Y * scaler_outputs.data_range_ + scaler_outputs.data_min_

yhatDev_rescale = yhat_dev * scaler_outputs.data_range_ + scaler_outputs.data_min_
Ydev_rescale = Ydev * scaler_outputs.data_range_ + scaler_outputs.data_min_

yhatTest_rescale = yhat_test * scaler_outputs.data_range_ + scaler_outputs.data_min_
Ytest_rescale = Ytest * scaler_outputs.data_range_ + scaler_outputs.data_min_

for i in range(Y.shape[1]):
    print("Time step index:", i)
    mape_train = np.mean(np.abs(yhatTrain_rescale[:,i,:] - Y_rescale[:,i,:]), axis=1)
    mape_train = 100 * mape_train / np.mean(Y_rescale[:,i,:], axis=1)
    rmse_train = np.sqrt(np.mean((yhatTrain_rescale[:,i,:] - Y_rescale[:,i,:])**2, axis=1))
    print("MAPE train error: %.3f \nRMSE train error: %.3f" % (np.mean(mape_train), np.mean(rmse_train)))
        
    mape_dev = np.mean(np.abs(yhatDev_rescale[:,i,:] - Ydev_rescale[:,i,:]), axis = 1)
    mape_dev = 100 * mape_dev / np.mean(Ydev_rescale[:,i,:], axis=1)
    rmse_dev = np.sqrt(np.mean((yhatDev_rescale[:,i,:] - Ydev_rescale[:,i,:])**2, axis = 1))
    print("MAPE dev error: %.3f \nRMSE dev error: %.3f" % (np.mean(mape_dev), np.mean(rmse_dev)))
        
    mape_test = np.mean(np.abs(yhatTest_rescale[:,i,:] - Ytest_rescale[:,i,:]), axis = 1)
    mape_test = 100 * mape_test / np.mean(Ytest_rescale[:,i,:], axis = 1)
    rmse_test = np.sqrt(np.mean((yhatTest_rescale[:,i,:] - Ytest_rescale[:,i,:])**2, axis = 1))
    print("MAPE test error: %.3f \nRMSE test error: %.3f" % (np.mean(mape_test), np.mean(rmse_test)))

#Plot single temperature prediction with actual values:
plt.plot(yhatTest_rescale[:,2,3], c='r')
plt.plot(Ytest_rescale[:,2,3], c='b')
plt.show()

yhatFinTest = model.predict([FinTestX, FinTestX2])
yhatFinTest_rescale = yhatFinTest * scaler_outputs.data_range_ + scaler_outputs.data_min_
FinTestY_rescale = FinTestY * scaler_outputs.data_range_ + scaler_outputs.data_min_

print("Here is the final test values:")

for i in range(Y.shape[1]):
    print("Time step index:", i)
    mape_test = np.mean(np.abs(yhatFinTest_rescale[:,i,:] - FinTestY_rescale[:,i,:]), axis = 1)
    mape_test = 100 * mape_test / np.mean(FinTestY_rescale[:,i,:], axis = 1)
    rmse_test = np.sqrt(np.mean((FinTestY_rescale[:,i,:] - yhatFinTest_rescale[:,i,:])**2, axis = 1))
    print("MAPE test error: %.3f \nRMSE test error: %.3f" % (np.mean(mape_test), np.mean(rmse_test)))

#Plot single temperature prediction with actual values:
plt.plot(yhatFinTest_rescale[:,2,3], c='r')
plt.plot(FinTestY_rescale[:,2,3], c='b')
plt.show()

plt.plot(Ytest_rescale[0,:,10], c='b')
plt.plot(yhatTest_rescale[0,:,10], color='r',marker='o')
plt.show()

plt.plot(Ytest_rescale[5,:,10], c='b')
plt.plot(yhatTest_rescale[5,:,10], color='r',marker='o')
plt.show()

#Print model architecture:
print(model.summary())
