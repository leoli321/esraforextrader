import pandas as pd
import matplotlib.pyplot as plt
import numpy as np 
import math
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.layers import LSTM
from keras.layers import Dropout
import preprocessing 



def lstm_neural_network(historical_data, look_back=240):
	# FOR REPRODUCIBILITY
	np.random.seed(7)

	# IMPORTING DATASET 
	# dataset = pd.read_csv('apple_share_price.csv', usecols=[1,2,3,4])
	dataset = historical_data.drop(['Date', 'Volume'], axis=1)
	dataset = dataset.reindex(index = dataset.index[::-1])

	# CREATING OWN INDEX FOR FLEXIBILITY
	obs = np.arange(1, len(dataset) + 1, 1)

	# TAKING DIFFERENT INDICATORS FOR PREDICTION
	# OHLC_avg = dataset.mean(axis = 1)
	HLC_avg = dataset[['High', 'Low', 'Close']].mean(axis = 1)
	close_val = dataset[['Close']]
	OHLC_avg = dataset[['Close']]

	# PLOTTING ALL INDICATORS IN ONE PLOT
	plt.plot(obs, OHLC_avg, 'r', label = 'OHLC avg')
	plt.plot(obs, HLC_avg, 'b', label = 'HLC avg')
	plt.plot(obs, close_val, 'g', label = 'Closing price')
	plt.legend(loc = 'upper right')
	# plt.show()

	# PREPARATION OF TIME SERIES DATASE
	OHLC_avg = np.reshape(OHLC_avg.values, (len(OHLC_avg),1)) # 1664
	scaler = MinMaxScaler(feature_range=(0, 1))
	OHLC_avg = scaler.fit_transform(OHLC_avg)

	# TRAIN-TEST SPLIT
	train_OHLC = int(len(OHLC_avg) * 0.75)
	test_OHLC = len(OHLC_avg) - train_OHLC
	train_OHLC, test_OHLC = OHLC_avg[0:train_OHLC,:], OHLC_avg[train_OHLC:len(OHLC_avg),:]

	# TIME-SERIES DATASET (FOR TIME T, VALUES FOR TIME T+1)
	trainX, trainY = preprocessing.new_dataset(train_OHLC, look_back)
	testX, testY = preprocessing.new_dataset(test_OHLC, look_back)

	# RESHAPING TRAIN AND TEST DATA
	trainX = np.reshape(trainX, (trainX.shape[0], 1, trainX.shape[1]))
	testX = np.reshape(testX, (testX.shape[0], 1, testX.shape[1]))

	# # LSTM MODEL
	# model = Sequential()
	# model.add(LSTM(32, input_shape=(1, look_back), return_sequences = True))
	# model.add(Dropout(0.1))
	# model.add(LSTM(16))
	# model.add(Dropout(0.1))
	# model.add(Dense(1))
	# model.add(Activation('linear'))

	# # MODEL COMPILING AND TRAINING
	# model.compile(loss='mean_squared_error', optimizer='adagrad') # Try SGD, adam, adagrad and compare!!!
	# model.fit(trainX, trainY, epochs=1000, batch_size=1, verbose=2)

	# LSTM MODEL 2
	model = Sequential()
	model.add(LSTM(25, input_shape=(1, look_back)))
	model.add(Dropout(0.1))
	model.add(Dense(1))
	model.compile(loss='mse', optimizer='adam')
	model.fit(trainX, trainY, epochs=1000, batch_size=240, verbose=2)


	# PREDICTION
	trainPredict = model.predict(trainX)
	testPredict = model.predict(testX)

	# DE-NORMALIZING FOR PLOTTING
	trainPredict = scaler.inverse_transform(trainPredict)
	trainY = scaler.inverse_transform([trainY])
	testPredict = scaler.inverse_transform(testPredict)
	testY = scaler.inverse_transform([testY])

	# TRAINING RMSE
	trainScore = math.sqrt(mean_squared_error(trainY[0], trainPredict[:,0]))
	print('Train RMSE: %.2f' % (trainScore))

	# TEST RMSE
	testScore = math.sqrt(mean_squared_error(testY[0], testPredict[:,0]))
	print('Test RMSE: %.2f' % (testScore))

	# CREATING SIMILAR DATASET TO PLOT TRAINING PREDICTIONS
	trainPredictPlot = np.empty_like(OHLC_avg)
	trainPredictPlot[:, :] = np.nan
	trainPredictPlot[look_back:len(trainPredict)+look_back, :] = trainPredict

	# CREATING SIMILAR DATASSET TO PLOT TEST PREDICTIONS
	testPredictPlot = np.empty_like(OHLC_avg)
	testPredictPlot[:, :] = np.nan
	testPredictPlot[len(trainPredict)+(look_back*2)+1:len(OHLC_avg)-1, :] = testPredict

	# DE-NORMALIZING MAIN DATASET 
	OHLC_avg = scaler.inverse_transform(OHLC_avg)

	# PLOT OF MAIN OHLC VALUES, TRAIN PREDICTIONS AND TEST PREDICTIONS
	plt.plot(OHLC_avg, 'g', label = 'original dataset')
	plt.plot(trainPredictPlot, 'r', label = 'training set')
	plt.plot(testPredictPlot, 'b', label = 'predicted stock price/test set')
	plt.legend(loc = 'upper right')
	plt.xlabel('Time in Days')
	plt.ylabel('OHLC Value of Apple Stocks')
	# plt.show()

	# PREDICT FUTURE VALUES
	last_val = testPredict[-240:]
	print(last_val.shape)
	# last_val_scaled = last_val/np.linalg.norm(last_val)
	last_val_scaled = scaler.fit_transform(last_val)
	next_val = model.predict(np.reshape(last_val_scaled, (1,1,240)))
	last_val_dnormalized = scaler.inverse_transform(last_val)
	next_val_dnormalized = scaler.inverse_transform(next_val)
	print("Last Candle Value: {}".format(np.asscalar(last_val_dnormalized[-1])))
	print("Next Candle value: {}".format(np.asscalar(next_val_dnormalized[-1])))

	plt.plot(np.asscalar(last_val_dnormalized[-1]), 'y', label='next candle prediction') 
	# plt.show()

	return np.asscalar(last_val_dnormalized[-1]), np.asscalar(next_val_dnormalized[-1])



