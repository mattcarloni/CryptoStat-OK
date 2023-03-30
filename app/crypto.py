from flask import (
    Blueprint,
    render_template,
    request,
)
import io
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import date, timedelta
today = date.today()
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
import tensorflow as tf
import warnings
from numpy import array
warnings.filterwarnings('ignore')
from keras.callbacks import ModelCheckpoint  

crypto = Blueprint('crypto', __name__)

@crypto.route("/analyse")
def analyse():
    return render_template("analyse.html")

@crypto.route('/test_get_plot', methods=['POST','GET'])
def get_plot():
    # Taking input form the html form and setting it up for the yfinance call
    if request.method == 'POST':
        global a, c
        a = request.form.get('crypto')
        c = "-usd"
        ac = a.strip('"') + c
        # Setting today as ending date
        d1 = today.strftime("%Y-%m-%d")
        end_date = d1
        # Setting the start day as 730 days ago to have a good amount of data
        d2 = date.today() - timedelta(days=730)
        d2 = d2.strftime("%Y-%m-%d")
        start_date = d2
        # Downloading the data from yahoo finance
        data1 = yf.download(str(ac), 
                            start=start_date, 
                            end=end_date, 
                            progress=False)
        data1["Date"] = data1.index
        # Setting the index as Date, defining the columns names and resetting the index  
        data1 = data1[["Date", "Open", "High", "Low", "Close", "Adj Close", "Volume"]]
        data1.reset_index(drop=True, inplace=True)
        # Creating a dataframe for handling and accessing the data
        data1 = pd.DataFrame(data1)
        # Taking the column "Open" that's the opening price for the selected crypto
        data = data1.Open
        # Data being normalized using log conversion that resulted the best option for crypto prediction
        data = data.apply(np.log)
        data = data.apply(np.sqrt)
        # Taking the last 20 days as input to predict the next day value
        # so the model is able to understand the trend of the past 20 days
        # and give a better prediction    
        n_steps = 20
        # Split data into training and testing
        X, y = split_sequence(data, n_steps)
        # This is the output number, that in this case is 1 because we'll only have Open value as output    
        n_features = 1
        # Reshaping the data to meet LSTM criteria
        X = X.reshape((X.shape[0], X.shape[1], n_features))
        # ModelCheckpoint method, saving our model on criteria of validation loss, so lower the validation loss
        # better is the model.        
        checkpoint = ModelCheckpoint("best_model.hdf5", monitor='val_loss', verbose=1,
            save_best_only=True, mode='auto', period=1)
        #Adding LSTM layer, dense layer and output layer to the model code    
        model = Sequential()
        tf.keras.backend.clear_session()
        model=Sequential()
        model.add((LSTM(32,return_sequences=True, input_shape=(n_steps, n_features),)))
        model.add((LSTM(16)))
        model.add((Dense(5))) #Hidden layers
        model.add((Dense(1))) #Output layer
        # Compile the model with RSME error (Root Mean Squared Error) one of the two main performance 
        # indicators for a regression model. It measures the average difference between values predicted 
        # by a model and the actual values. It provides an estimation of how well the model is able to
        # predict the target value (accuracy).

        # Aadam as optimizer. When training a deep learning model, you must adapt every epoch's weight and 
        # minimize the loss function. An optimizer is an algorithm or function that adapts the neural network's 
        # attributes, like learning rate and weights. 
        # Hence, it assists in improving the accuracy and reduces the total loss.
        model.compile(loss='mean_squared_error',optimizer='adam')
        # Training of model for 60 epochs
        model.fit(X, y, 
            validation_data=(X, y),
            epochs=50, verbose=2, batch_size = 12,callbacks=[checkpoint])
        # Loading the best model from 60 epochs
        from keras.models import load_model
        model=load_model('best_model.hdf5')
        # Defining data again for future prediction
        n_steps = 20
        v = data[len(data)-21:len(data)]
        i = 0
        # Model prediction on whole dataset
        predicted = model.predict(X)
        # From above prediction we are trimming the data to show first plot which incudes model 
        # prediction and actual data
        fig,ax = plt.subplots(figsize=(10,5))
        ax.plot(list(predicted[600:]), linestyle='dashed', color='red')
        ax.plot(list(y[600:]), color='blue')
        ax.set_xlabel("Time (Days)")
        ax.set_ylabel("Value")
        ax.grid()
        ax.legend(('Predicted Trend','Actual Trend'))
        plt.savefig('app/static/media/predict.png')
        # Here we are creating code for our future 15 days prediction. Ao for that we are taking 
        # recent 21 values and then starting prediction of next value 
        # Then again appending this next value in current data and using this value and past 19 we are 
        # predicting next vaule and so on 
        i=0
        global lst_output
        lst_output=[]
        x_input = np.array(v)
        temp_input=list(x_input)
        while(i<15):
            
            if(len(temp_input)>3):
                x_input=array(temp_input[1:])
                x_input=pd.DataFrame(x_input)
                x_input=np.array(x_input)
                x_input = x_input.reshape((1, n_steps, n_features))
                x_input=np.asarray(x_input).astype(np.float32)
                yhat = model.predict(x_input, verbose=1)
                temp_input.append(yhat[0][0])
                temp_input=temp_input[1:]
                lst_output.append(yhat[0][0])
                i=i+1
            else:
                x_input = x_input.reshape((1, n_steps, n_features))
                yhat = model.predict(x_input, verbose=0)
            
                temp_input.append(yhat[0][0])
                lst_output.append(yhat[0][0])
                i=i+1
            
        global b
        # Plotting next predixtion graph
        data=data[500:]
        b = list(data)+list(lst_output)
        img = io.BytesIO()
        fig,ax = plt.subplots(figsize=(10,5))
        ax.plot(list(b),linestyle='dashed', color='red')
        ax.plot(list(data),color='blue')
        ax.set_xlabel("Time (Days)")
        ax.set_ylabel("Value")
        ax.grid()
        ax.legend(('Predicted Trend','Actual Trend'))
        plt.savefig('app/static/media/my_plot.png')
        img.seek(0)
        # Creating dates for future 15 days
        gv = pd.date_range(str(d1), periods=15)
        lst_output = pd.DataFrame(lst_output)
        # Our data is in log format so performing antilog to convert this data again into normal
        pred_log = lst_output.apply(np.square)
        pred = pred_log.apply(np.exp)
        dv = pred[0]
        # Creating dataframe to display output on website and saving output in csv
        gv = pd.DataFrame(gv)
        gv['Forecast'] = dv
        gv = gv.rename(columns={gv.columns[0]: 'Days', 'Forecast': ''})
        gv.set_index('Days', inplace=True)
        gv.to_csv('app/forecast.csv')
            
    return render_template('analyse.html', get_plot = True, plot_url = 'static/media/my_plot.png', plot_url1 = 'static/media/predict.png', data = data,tables=[gv.to_html(classes='data')], titles=gv.columns.values)

from flask import Flask,render_template,send_file
# Download the data 
@crypto.route('/download_file')
def download_file():
    path = "forecast.csv"
    return send_file(path,as_attachment=True)

from numpy import array
def split_sequence(sequence, n_steps):
	X, y = list(), list()
	for i in range(len(sequence)):
		# Find the end of this pattern
		end_ix = i + n_steps
		# Check if we are beyond the sequence
		if end_ix > len(sequence)-1:
			break
		# Gather input and output parts of the pattern
		seq_x, seq_y = sequence[i:end_ix], sequence[end_ix]
		X.append(seq_x)
		y.append(seq_y)
	return array(X), array(y)
