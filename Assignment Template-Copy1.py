#!/usr/bin/env python
# coding: utf-8

# # Building an Understanding of How Nodes in Neural Networks Work
# ## Gary Lawson - 03 May 2019
# ***

# ### 1.0 Abstract
# ***

# ### 2.0 Introduction
# ***

# ### 3.0 Literature Review
# ***
# For ARIMA Model:
# https://towardsdatascience.com/forecasting-exchange-rates-using-arima-in-python-f032f313fc56
# https://www.machinelearningplus.com/time-series/arima-model-time-series-forecasting-python/
# https://machinelearningmastery.com/arima-for-time-series-forecasting-with-python/
# https://medium.com/@josemarcialportilla/using-python-and-auto-arima-to-forecast-seasonal-time-series-90877adff03c
# 
# For coding CGP:
# https://medium.com/cindicator/genetic-algorithms-and-hyperparameters-weekend-of-a-data-scientist-8f069669015e
# 
# Another for coding CGP:
# http://aqibsaeed.github.io/2017-08-11-genetic-algorithm-for-optimizing-rnn/
# 
# Information from Rob J Hyndman on time series work
# https://otexts.com/fpp2/
# 
# Stock prediction using LSTM in Keras
# https://www.kaggle.com/amarpreetsingh/stock-prediction-lstm-using-keras/notebook

# ### 4.0 Methods
# ***

# #### 4.1 Exploratory Data Analysis
# ***

# In[1]:


# Import packages

###########################################################################################
import sys

# Import base and prepocessing packages
import numpy as np # for creating and working with arrays
import pandas as pd # for creating and working with dataframes
import sklearn as sk
from sklearn.preprocessing import LabelBinarizer # for One Hot Encoding
import time # for recording times on model runs
import pickle  # used for dumping and loading binary files

# To import data
import zipfile # to work with zip folders
from collections import defaultdict

# Import NN packages
import tensorflow as tf
import tensorflow.keras
from keras.models import Sequential
from keras.layers import Embedding, Flatten, Dense, SimpleRNN, LSTM, GRU, Bidirectional, Conv1D, MaxPooling1D, GlobalMaxPooling1D
from keras import backend as K
from keras.utils import plot_model # For visualizing Keras model
from keras.preprocessing.text import Tokenizer # For tokenizing the text
from keras.preprocessing.sequence import pad_sequences
from keras.utils.np_utils import to_categorical # For one-hot encoding

# Import plotting and data visualization packages
import matplotlib.pyplot as plt  # static plotting
import matplotlib as mpl
get_ipython().run_line_magic('matplotlib', 'inline')
from collections import OrderedDict  # for creating table output
import IPython # For plotting Keras Model visualizations
import seaborn as sns # For heatmap visualization
from bokeh.io import push_notebook, show, output_notebook
from bokeh.models import ColumnDataSource, ColorBar, DatetimeTickFormatter, HoverTool
from bokeh.palettes import Spectral6
from bokeh.transform import linear_cmap
from bokeh.plotting import figure
output_notebook()

# Import seeding packages for reproducible results
from numpy.random import seed
from tensorflow import set_random_seed
import random as rn

###########################################################################################

# Print versions of primary packages
print('Package Versions')
print('***********************************************************************************')
print(f"Tensor Flow Version: {tf.__version__}")
print(f"Keras Version: {tensorflow.keras.__version__}")
print()
print(f"Python {sys.version}")
print(f"Pandas {pd.__version__}")
print(f"Scikit-Learn {sk.__version__}")

# Print specs of this machine
import multiprocessing
import platform
import psutil
from tensorflow.python.client import device_lib

def sizeof_fmt(num, suffix='B'):
    for unit in ['','Ki','Mi','Gi','Ti','Pi','Ei','Zi']:
        if abs(num) < 1024.0:
            return "%3.1f%s%s" % (num, unit, suffix)
        num /= 1024.0
    return "%.1f%s%s" % (num, 'Yi', suffix)

p = platform.processor()
mem = psutil.virtual_memory()

print('\nEnvironment Specifications')
print('***********************************************************************************')
print(f'Your CPU is: {p}')
print(f'Your CPU has {multiprocessing.cpu_count()} cores.')
print(f'Total memory: {sizeof_fmt(mem.total)}')
print(mem)

local_device_protos = device_lib.list_local_devices()
gpuList = [x for x in local_device_protos if x.device_type == 'GPU']

print("Installed GPUs:")
for x in gpuList:
  print("{} - {}".format(x.name,x.physical_device_desc))


# #### Load Data
# 
# Create functions to identify available data and import it into notebook for use in modeling.  Data is stored in .zip files to conserve space, so the functions are created to unzip this data automatically.  EUR/USD is just one of the FOREX pairs available for evaluation, but the following functions can be used to identify and load other exchange pairs.

# In[2]:


# Define a function to look at files available in zip folder
#import zipfile 

def zip_file_names(zip_loc):
    with zipfile.ZipFile(zip_loc, "r") as z: # unzips file
       for filename in z.namelist():  # loops through filenames in unziped folder
          print(filename)  # prints filename
            
# Reference:
# https://stackoverflow.com/questions/40824807/reading-zipped-json-files


# In[3]:


# Set the file locations for data and other resources

EUR_USD_DATA = "C:/Users/DELL/Documents/Data Warehouse/FOREX DATA/EUR_USD.zip"


# In[4]:


# Look at what files exist in a zip directory
zip_file_names(EUR_USD_DATA)


# In[5]:


# Define function to open up all xlsx files in a zip folder.  Note that df_name and array_name must be passed
# as sting values.  The funtion will create the df and array.

#import zipfile

def zip_load_file(zip_loc, df_name, array_name): 
    #print(df_name.info())
    df_name = pd.DataFrame() # Create the dataframe
    with zipfile.ZipFile(zip_loc, "r") as z: # unzips file
        for filename in z.namelist():  # loops through filenames in unziped folder
            if filename.endswith('.xlsx'): # Looks only for specific file type.  Can search for csv file if available
                with z.open(filename) as f:  # Opens file, closes after this "with" command
                    data = pd.read_excel(f, names=['datetime', 'bid_open','bid_high',
                                                   'bid_low','bid_close','volume']) # Can use read_csv if csv file.
                    df_name = pd.concat([df_name,data]) # Concats this sheets data with previous
                    #print(df_name.info())
                print("Successfully loaded:'{}'".format(filename)) # Print filename to confirm loaded
                print("'{}' Observations Added".format(len(data))) # Print length to see what was added
    df_name = df_name.set_index('datetime') # set the index to the datetime field
    array_name = df_name.values # Convert the dataframe to an array
    print("\n***Final Dataframe Information***\n") # Print final dataframe info
    print(df_name.info())
    print("\n***Final Dataframe Head***\n", df_name.head()) # Print head of final dataframe
    print("\nData Array Shape:", array_name.shape) # Print the shape of the array
    return df_name, array_name # Return the df and the array

# Reference:
# https://stackoverflow.com/questions/40824807/reading-zipped-json-files


# In[6]:


# Unzip and load data as dataframe

EUR_USD_min_df, EUR_USD_min_array = zip_load_file(EUR_USD_DATA, "EUR_USD_min_df", "EUR_USD_min_array")


# In[7]:


# Define function to open up all xlsx files in a zip folder.  Note that df_name and array_name must be passed
# as sting values.  The funtion will create the df and array.

#import zipfile

def zip_load_file(zip_loc): 
    #print(df_name.info())
    df_name = pd.DataFrame() # Create the dataframe
    with zipfile.ZipFile(zip_loc, "r") as z: # unzips file
        for filename in z.namelist():  # loops through filenames in unziped folder
            if filename.endswith('.xlsx'): # Looks only for specific file type.  Can search for csv file if available
                with z.open(filename) as f:  # Opens file, closes after this "with" command
                    data = pd.read_excel(f, names=['datetime', 'bid_open','bid_high',
                                                   'bid_low','bid_close','volume']) # Can use read_csv if csv file.
                    df_name = pd.concat([df_name,data]) # Concats this sheets data with previous
                    #print(df_name.info())
                print("Successfully loaded:'{}'".format(filename)) # Print filename to confirm loaded
                print("'{}' Observations Added".format(len(data))) # Print length to see what was added
    df_name = df_name.set_index('datetime') # set the index to the datetime field
    array_name = df_name.reset_index().values # Convert the dataframe to an array
    
    print("\n***Final Dataframe Information***\n") # Print final dataframe info
    print(df_name.info())
    print("\n***Final Dataframe Head***\n", df_name.head()) # Print head of final dataframe
    print("\nData Array Shape:", array_name.shape) # Print the shape of the array
    return df_name, array_name # Return the df and the array

# Reference:
# https://stackoverflow.com/questions/40824807/reading-zipped-json-files


# In[8]:


# Unzip and load data as dataframe

EUR_USD_min_df, EUR_USD_min_array = zip_load_file(EUR_USD_DATA)


# #### Create Additional Time Series Datasets
# 
# Available data is on a 1-minute scale and includes Bid Open, Bid High, Bid Low, Bid Close, and Volume.  To evaluate an acceptable timeframe for predicting future exchange rates, functions will be created to convert the dataset from 1-minute to 1-hour and 1-day prices using the median value.

# In[9]:


# Create a function to take minute data and resample as hourly and daily data.  Use the median value for the 
# timeframe.  Returns dataframe and array for both hourly and daily datasets.

def create_hour_and_day_datasets(df):
    hour_df_name = df.resample('H').median() # Resample as hourly data using median
    day_df_name = df.resample('D').median() # Resample as daily data using median
    hour_array_name = hour_df_name.reset_index().values
    day_array_name = day_df_name.reset_index().values
    return hour_df_name, day_df_name, hour_array_name, day_array_name # Return two dataframes


# In[10]:


# Create the hourly and daily datasets

EUR_USD_hour_df, EUR_USD_day_df, EUR_USD_hour_array, EUR_USD_day_array = create_hour_and_day_datasets(EUR_USD_min_df)


# In[11]:


# Print hourly dataset info
print('Array shape:', EUR_USD_hour_array.shape)
print('Array head:', EUR_USD_hour_array[0])
EUR_USD_hour_df.head()


# In[12]:


# Print daily dataset info
print('Array shape:', EUR_USD_day_array.shape)
print('Array head:', EUR_USD_day_array[0])
EUR_USD_day_df.head()


# #### Create Train and Test Datasets
# 
# Model development will be completed using a training dataset, however a test dataset will be created and withheld until final testing to allow for veriftying the model's ability to generalize to new data that has not yet been seen.

# In[13]:


def train_val_test_gen(array_name, train_ratio, val_ratio, test_ratio):
    train_size = int(len(array_name) * train_ratio) # Determine training size based on dataset length * train_ratio
    val_size = int(len(array_name) * val_ratio)
    test_size = int(len(array_name) * test_ratio)
    train, val, test = array_name[0:train_size], array_name[train_size:train_size + val_size], array_name[train_size + val_size:len(array_name)] # Set train and test based on index
    print('Observations: %d' % (len(array_name))) # Print observation length
    print('Training Observation Shape: %d' % (len(train))) # Print train length
    print('Validation Observations: %d' % (len(val))) # Print train length
    print('Testing Observations: %d' % (len(test))) # Print test length
    return train, val, test

# https://machinelearningmastery.com/backtest-machine-learning-models-time-series-forecasting/


# In[14]:


# Create the minute dataset

train_EUR_USD_min, val_EUR_USD_min, test_EUR_USD_min = train_val_test_gen(EUR_USD_min_array, 0.8, 0.1, 0.1)


# In[15]:


# Print out info about the dataset to make sure it was created correctly.

print('Train Shape: ', train_EUR_USD_min.shape)
print('Val Shape: ', val_EUR_USD_min.shape)
print('Test Shape: ', test_EUR_USD_min.shape)
print('Example Observation: ', train_EUR_USD_min[0])


# In[16]:


# Create the minute dataset

train_EUR_USD_hour, val_EUR_USD_hour, test_EUR_USD_hour = train_val_test_gen(EUR_USD_hour_array, 0.8, 0.1, 0.1)


# In[17]:


# Print out info about the dataset to make sure it was created correctly.

print('Train Shape: ', train_EUR_USD_hour.shape)
print('Val Shape: ', val_EUR_USD_hour.shape)
print('Test Shape: ', test_EUR_USD_hour.shape)
print('Example Observation: ', train_EUR_USD_hour[0])


# In[18]:


# Create the minute dataset

train_EUR_USD_day, val_EUR_USD_day, test_EUR_USD_day = train_val_test_gen(EUR_USD_day_array, 0.8, 0.1, 0.1)


# In[19]:


# Print out info about the dataset to make sure it was created correctly.

print('Train Shape: ', train_EUR_USD_day.shape)
print('Val Shape: ', val_EUR_USD_day.shape)
print('Test Shape: ', test_EUR_USD_day.shape)
print('Example Observation: ', train_EUR_USD_day[0])


# In[20]:


# Create a function to plot the train, val, and test data to observe the splits.

def plot_train_val_test_dataset(duration, pair, train, val, test):
    duration = duration
    pair = pair
    x = train[:,0] # Set the time series variable to "x1"
    y1 = train[:,4] # Set the first price series variable as train.  [:,4] returns the Bid Close price
    x2 = val[:,0] # Set the time series variable to "x2"
    y2 = val[:,4] # Set the second price series variable as validation.  [:,4] returns the Bid Close price
    x3 = test[:,0] # Set the time series variable to "x2"
    y3 = test[:,4] # Set the third price series variable as test.  [:,4] returns the Bid Close price

    TOOLS = "hover,pan,wheel_zoom,box_zoom,reset,save" #Define the tools for the plot

    p = figure(tools=TOOLS, title='%s Data For %s Pair' %(duration,pair), plot_width=900, plot_height=300)
    r = p.line(x, y1, color='green', line_width=0.5,legend='Train')
    r1 = p.line(x2, y2, color='blue', line_width=0.5,legend='Validation')
    r2 = p.line(x3, y3, color='red', line_width=0.5,legend='Test')

    p.xaxis.axis_label = 'Time(%s)' %time
    p.xaxis.formatter = DatetimeTickFormatter(
        seconds="%d %B %Y",
        minutes="%d %B %Y",
        hours="%d %b %Y",
        days="%d %b %Y",
        months="%d %b %Y",
        years="%d %b %Y"
        ) # Plot the x-axis as day/month/year
    p.yaxis.axis_label = 'Exchange Rate'
    p.legend.location = 'top_right'

    hover = HoverTool(
        tooltips = [
            ("Date", "@x{%Y-%m-%d %H:%M:%S}"),
            ("Bid Close", "@y"),
        ],
        formatters={
            'x': 'datetime',
        },
        mode='vline'
    )
    p.add_tools(hover)

    show(p)
    
# https://stackoverflow.com/questions/55732163/modify-the-format-of-display-on-hover-in-bokeh
# https://stackoverflow.com/questions/51496142/formatting-pandas-datetime-in-bokeh-hovertool


# In[21]:


# Plote the minute dataset, including train, val, and test

plot_train_val_test_dataset('Minute', 'EUR_USD', train_EUR_USD_min, val_EUR_USD_min, test_EUR_USD_min)


# In[22]:


# Plote the minute dataset, including train, val, and test

plot_train_val_test_dataset('Hourly', 'EUR_USD', train_EUR_USD_hour, val_EUR_USD_hour, test_EUR_USD_hour)


# In[23]:


# Plote the minute dataset, including train, val, and test

plot_train_val_test_dataset('Daily', 'EUR_USD', train_EUR_USD_day, val_EUR_USD_day, test_EUR_USD_day)


# #### Create Training and Validation Batches
# 
# Functions will be created to allow for the raw time series data to be converted into batches of several training values and one or more validation values using a generator.  The method selected to accomplish this is a sliding window, though an expanding window could be evaluated in the future.

# In[24]:


# Create a function to generate batches of data that includes past exchange prices (training data) as well 
# as future prices (validation data).

'''
lookback - How many timesteps back the input data should go
delay - How many timesteps in the future the target should be
min_index and max_index - Inidices in the data array that delimit which time steps to draw from.  This is useful 
    for keeping a segment of the data for validation and another for testing.
shuffle - Whether to shuffle the samples or draw them in chronological order.  Always set as False in this experiment.
batch_size - The number of samples per batch
step - The period, in timesteps, at which you sample data.  
'''

def generator(data, lookback, delay, shuffle, batch_size, step):
    #if max_index is None:
    max_index = len(data) - delay - 1
    i = lookback
    while 1:
        if shuffle:
            rows = np.random.randint(lookback, max_index, size=batch_size)
        else:
            if i + batch_size >= max_index:
                i = lookback
            rows = np.arange(i, min(i + batch_size, max_index))
            i += len(rows)
        
        samples = np.zeros((len(rows),
                           lookback // step,
                           data.shape[-1]))
        
        targets = np.zeros((len(rows),))
        for j, row in enumerate(rows):
            indices = range(rows[j] - lookback, rows[j], step)
            samples[j] = data[indices]
            targets[j] = data[rows[j] + delay][3] # [3] defines the Bid Close price
        yield samples, targets


# In[420]:


# Create lists to hold generators for use in model loops
'format - variable type _ pair _ data window _ lookback _ forecast delay _ step'
# This list will hold the string name of the generator for reference
datasets = []
# These lists will hold the variables used to create each generator
duration_data_list = []
pair_data_list = []
lookback_data_list = []
delay_data_list = []
step_data_list = []
# These lists will hold the generator objects
train_gen_list = []
val_gen_list = []
test_gen_list = []
# These lists will hold the steps for the val and test generators
val_steps_list = []
test_steps_list = []


# In[421]:


# Due to the training time required for minute data, this dataset was not included.
'''
# Create generators for the following dataset.  I tried to do this through a function, but kept getting random nan values
# in the generator output.

duration = 'Minute'
pair = 'EUR_USD'
lookback = 60*24 # Look back 1 day
delay = 60 # Predict 1 hour into future
step = 1 # Look at data every minute
batch_size = 128

# Set the input data for generators
train = train_EUR_USD_min
test = val_EUR_USD_min
val = test_EUR_USD_min

# Create the train data generator
train_gen = generator(train[:,1:6],
                      lookback = lookback,
                      delay = delay,
                      shuffle = True,
                      step = step,
                      batch_size = batch_size)

# Create the validation data generator    
test_gen = generator(val[:,1:6],
                    lookback = lookback,
                    delay = delay,
                    shuffle = False,
                    step = step,
                    batch_size = batch_size)


# Create the test data generator
val_gen = generator(test[:,1:6],
                     lookback = lookback,
                     delay = delay,
                     shuffle = False,
                     step = step,
                     batch_size = batch_size)

# Define the number of validation steps
val_steps = len(val) - lookback
print('Validation steps:', val_steps)


# Define the number of test steps
test_steps = len(test) - lookback
print('\nTest steps:', test_steps)

# Print info about these generators
print('\n*** The following Generators have been created ***')
print('Duration:', duration)
print('Pair:', pair)
print('Input step: %r %r' %(step, duration))
print('Forecast delay: %r %r' %(delay, duration))
print('Generators: train, val, test')

# Rename variable for this dataset
# format - variable type _ pair _ data window _ lookback _ forecast delay _ step
train_gen_EUR_USD_min_day_hour_min = train_gen
val_gen_EUR_USD_min_day_hour_min = val_gen
test_gen_EUR_USD_min_day_hour_min = test_gen
val_steps_EUR_USD_min_day_hour_min = val_steps
test_steps_EUR_USD_min_day_hour_min = test_steps

# Append dataset extension to list for reference later
datasets.append('EUR_USD_min_day_hour_min')

# Append generators to a list for use in model loops
train_gen_list.append(train_gen_EUR_USD_min_day_hour_min)
val_gen_list.append(val_gen_EUR_USD_min_day_hour_min)
test_gen_list.append(test_gen_EUR_USD_min_day_hour_min)
val_steps_list.append(val_steps_EUR_USD_min_day_hour_min)
test_steps_list.append(test_steps_EUR_USD_min_day_hour_min)

# Append generator specs to allow for understanding of parameters of each dataset in future
duration_data_list.append(duration)
pair_data_list.append(pair)
lookback_data_list.append(lookback)
delay_data_list.append(delay)
step_data_list.append(step)

# Chollet Chap 6, pg 211
'''


# In[422]:


# Due to the training time required for minute data, this dataset was not included.
'''
print('Generated Sample Array Shape:', next(train_gen_EUR_USD_min_day_hour_min)[0].shape)
print('Generated Target Array Shape', next(train_gen_EUR_USD_min_day_hour_min)[1].shape)
print('\nGenerated Sample and Target Array Exampled\n\n', next(train_gen_EUR_USD_min_day_hour_min))
'''


# In[423]:


# Due to the training time required for minute data, this dataset was not included.
'''
# Create generators for the following dataset.  I tried to do this through a function, but kept getting random nan values
# in the generator output.

duration = 'Minute'
pair = 'EUR_USD'
lookback = 60*24 # Look back 1 day
delay = 60*24 # Predict 1 day into future
step = 1 # Look at data every minute
batch_size = 128

# Set the input data for generators
train = train_EUR_USD_min
test = val_EUR_USD_min
val = test_EUR_USD_min

# Create the train data generator
train_gen = generator(train[:,1:6],
                      lookback = lookback,
                      delay = delay,
                      shuffle = True,
                      step = step,
                      batch_size = batch_size)

# Create the validation data generator 
test_gen = generator(val[:,1:6],
                    lookback = lookback,
                    delay = delay,
                    shuffle = False,
                    step = step,
                    batch_size = batch_size)

# Create the test data generator
val_gen = generator(test[:,1:6],
                     lookback = lookback,
                     delay = delay,
                     shuffle = False,
                     step = step,
                     batch_size = batch_size)

# Define the number of validation steps
val_steps = len(val) - lookback
print('Validation steps:', val_steps)

# Define the number of test steps
test_steps = len(test) - lookback
print('\nTest steps:', test_steps)

# Print info about these generators
print('\n*** The following Generators have been created ***')
print('Duration:', duration)
print('Pair:', pair)
print('Input step: %r %r' %(step, duration))
print('Forecast delay: %r %r' %(delay, duration))
print('Generators: train, val, test')

# Rename variable for this dataset
# format - variable type _ pair _ data window _ lookback _ forecast delay _ step
train_gen_EUR_USD_min_day_day_min = train_gen
val_gen_EUR_USD_min_day_day_min = val_gen
test_gen_EUR_USD_min_day_day_min = test_gen
val_steps_EUR_USD_min_day_day_min = val_steps
test_steps_EUR_USD_min_day_day_min = test_steps

# Append dataset extension to list for reference later
datasets.append('EUR_USD_min_day_day_min')

# Append generators to a list for use in model loops
train_gen_list.append(train_gen_EUR_USD_min_day_day_min)
val_gen_list.append(val_gen_EUR_USD_min_day_day_min)
test_gen_list.append(test_gen_EUR_USD_min_day_day_min)
val_steps_list.append(val_steps_EUR_USD_min_day_day_min)
test_steps_list.append(test_steps_EUR_USD_min_day_day_min)

# Append generator specs to allow for understanding of parameters of each dataset in future
duration_data_list.append(duration)
pair_data_list.append(pair)
lookback_data_list.append(lookback)
delay_data_list.append(delay)
step_data_list.append(step)

# Chollet Chap 6, pg 211
'''


# In[424]:


# Due to the training time required for minute data, this dataset was not included.
'''
print('Generated Sample Array Shape:', next(train_gen_EUR_USD_min_day_day_min)[0].shape)
print('Generated Target Array Shape', next(train_gen_EUR_USD_min_day_day_min)[1].shape)
print('\nGenerated Sample and Target Array Exampled\n\n', next(train_gen_EUR_USD_min_day_day_min))
'''


# In[425]:


# Create generators for the following dataset.  I tried to do this through a function, but kept getting random nan values
# in the generator output.

duration = 'Minute'
pair = 'EUR_USD'
lookback = 60*24 # Look back 1 day
delay = 60 # Predict 1 hour into future
step = 60 # Look at data every hour
batch_size = 128

# Set the input data for generators
train = train_EUR_USD_min
test = val_EUR_USD_min
val = test_EUR_USD_min

# Create the train data generator
train_gen = generator(train[:,1:6],
                      lookback = lookback,
                      delay = delay,
                      shuffle = True,
                      step = step,
                      batch_size = batch_size)

# Create the validation data generator 
test_gen = generator(val[:,1:6],
                    lookback = lookback,
                    delay = delay,
                    shuffle = False,
                    step = step,
                    batch_size = batch_size)

# Create the test data generator
val_gen = generator(test[:,1:6],
                     lookback = lookback,
                     delay = delay,
                     shuffle = False,
                     step = step,
                     batch_size = batch_size)

# Define the number of validation steps
val_steps = len(val) - lookback
print('Validation steps:', val_steps)

# Define the number of test steps
test_steps = len(test) - lookback
print('\nTest steps:', test_steps)

# Print info about these generators
print('\n*** The following Generators have been created ***')
print('Duration:', duration)
print('Pair:', pair)
print('Input step: %r %r' %(step, duration))
print('Forecast delay: %r %r' %(delay, duration))
print('Generators: train, val, test')

# Rename variable for this dataset
# format - variable type _ pair _ data window _ lookback _ forecast delay _ step
train_gen_EUR_USD_min_day_hour_hour = train_gen
val_gen_EUR_USD_min_day_hour_hour = val_gen
test_gen_EUR_USD_min_day_hour_hour = test_gen
val_steps_EUR_USD_min_day_hour_hour = val_steps
test_steps_EUR_USD_min_day_hour_hour = test_steps

# Append dataset extension to list for reference later
datasets.append('EUR_USD_min_day_hour_hour')

# Append generators to a list for use in model loops
train_gen_list.append(train_gen_EUR_USD_min_day_hour_hour)
val_gen_list.append(val_gen_EUR_USD_min_day_hour_hour)
test_gen_list.append(test_gen_EUR_USD_min_day_hour_hour)
val_steps_list.append(val_steps_EUR_USD_min_day_hour_hour)
test_steps_list.append(test_steps_EUR_USD_min_day_hour_hour)

# Append generator specs to allow for understanding of parameters of each dataset in future
duration_data_list.append(duration)
pair_data_list.append(pair)
lookback_data_list.append(lookback)
delay_data_list.append(delay)
step_data_list.append(step)

# Chollet Chap 6, pg 211


# In[426]:


print('Generated Sample Array Shape:', next(train_gen_EUR_USD_min_day_hour_hour)[0].shape)
print('Generated Target Array Shape', next(train_gen_EUR_USD_min_day_hour_hour)[1].shape)
print('\nGenerated Sample and Target Array Exampled\n\n', next(train_gen_EUR_USD_min_day_hour_hour))


# In[427]:


# Create generators for the following dataset.  I tried to do this through a function, but kept getting random nan values
# in the generator output.

duration = 'Minute'
pair = 'EUR_USD'
lookback = 60*24 # Look back 1 day
delay = 60*24 # Predict 1 day into future
step = 60 # Look at data every hour
batch_size = 128

# Set the input data for generators
train = train_EUR_USD_min
test = val_EUR_USD_min
val = test_EUR_USD_min

# Create the train data generator
train_gen = generator(train[:,1:6],
                      lookback = lookback,
                      delay = delay,
                      shuffle = True,
                      step = step,
                      batch_size = batch_size)

# Create the validation data generator 
test_gen = generator(val[:,1:6],
                    lookback = lookback,
                    delay = delay,
                    shuffle = False,
                    step = step,
                    batch_size = batch_size)

# Create the test data generator
val_gen = generator(test[:,1:6],
                     lookback = lookback,
                     delay = delay,
                     shuffle = False,
                     step = step,
                     batch_size = batch_size)

# Define the number of validation steps
val_steps = len(val) - lookback
print('Validation steps:', val_steps)

# Define the number of test steps
test_steps = len(test) - lookback
print('\nTest steps:', test_steps)

# Print info about these generators
print('\n*** The following Generators have been created ***')
print('Duration:', duration)
print('Pair:', pair)
print('Input step: %r %r' %(step, duration))
print('Forecast delay: %r %r' %(delay, duration))
print('Generators: train, val, test')

# Rename variable for this dataset
# format - variable type _ pair _ data window _ lookback _ forecast delay _ step
train_gen_EUR_USD_min_day_day_hour = train_gen
val_gen_EUR_USD_min_day_day_hour = val_gen
test_gen_EUR_USD_min_day_day_hour = test_gen
val_steps_EUR_USD_min_day_day_hour = val_steps
test_steps_EUR_USD_min_day_day_hour = test_steps

# Append dataset extension to list for reference later
datasets.append('EUR_USD_min_day_day_hour')

# Append generators to a list for use in model loops
train_gen_list.append(train_gen_EUR_USD_min_day_day_hour)
val_gen_list.append(val_gen_EUR_USD_min_day_day_hour)
test_gen_list.append(test_gen_EUR_USD_min_day_day_hour)
val_steps_list.append(val_steps_EUR_USD_min_day_day_hour)
test_steps_list.append(test_steps_EUR_USD_min_day_day_hour)

# Append generator specs to allow for understanding of parameters of each dataset in future
duration_data_list.append(duration)
pair_data_list.append(pair)
lookback_data_list.append(lookback)
delay_data_list.append(delay)
step_data_list.append(step)

# Chollet Chap 6, pg 211


# In[428]:


print('Generated Sample Array Shape:', next(train_gen_EUR_USD_min_day_day_hour)[0].shape)
print('Generated Target Array Shape', next(train_gen_EUR_USD_min_day_day_hour)[1].shape)
print('\nGenerated Sample and Target Array Exampled\n\n', next(train_gen_EUR_USD_min_day_day_hour))


# In[429]:


# Due to the training time required for minute data, this dataset was not included.
'''
# Create generators for the following dataset.  I tried to do this through a function, but kept getting random nan values
# in the generator output.

duration = 'Minute'
pair = 'EUR_USD'
lookback = 60*24*7 # Look back 1 week
delay = 60 # Predict 1 hour into future
step = 1 # Look at data every minute
batch_size = 128

# Set the input data for generators
train = train_EUR_USD_min
test = val_EUR_USD_min
val = test_EUR_USD_min

# Create the train data generator
train_gen = generator(train[:,1:6],
                      lookback = lookback,
                      delay = delay,
                      shuffle = True,
                      step = step,
                      batch_size = batch_size)

# Create the validation data generator 
test_gen = generator(val[:,1:6],
                    lookback = lookback,
                    delay = delay,
                    shuffle = False,
                    step = step,
                    batch_size = batch_size)

# Create the test data generator
val_gen = generator(test[:,1:6],
                     lookback = lookback,
                     delay = delay,
                     shuffle = False,
                     step = step,
                     batch_size = batch_size)

# Define the number of validation steps
val_steps = len(val) - lookback
print('Validation steps:', val_steps)

# Define the number of test steps
test_steps = len(test) - lookback
print('\nTest steps:', test_steps)

# Print info about these generators
print('\n*** The following Generators have been created ***')
print('Duration:', duration)
print('Pair:', pair)
print('Input step: %r %r' %(step, duration))
print('Forecast delay: %r %r' %(delay, duration))
print('Generators: train, val, test')

# Rename variable for this dataset
# format - variable type _ pair _ data window _ lookback _ forecast delay _ step
train_gen_EUR_USD_min_week_hour_min = train_gen
val_gen_EUR_USD_min_week_hour_min = val_gen
test_gen_EUR_USD_min_week_hour_min = test_gen
val_steps_EUR_USD_min_week_hour_min = val_steps
test_steps_EUR_USD_min_week_hour_min = test_steps

# Append dataset extension to list for reference later
datasets.append('EUR_USD_min_week_hour_min')

# Append generators to a list for use in model loops
train_gen_list.append(train_gen_EUR_USD_min_week_hour_min)
val_gen_list.append(val_gen_EUR_USD_min_week_hour_min)
test_gen_list.append(test_gen_EUR_USD_min_week_hour_min)
val_steps_list.append(val_steps_EUR_USD_min_week_hour_min)
test_steps_list.append(test_steps_EUR_USD_min_week_hour_min)

# Append generator specs to allow for understanding of parameters of each dataset in future
duration_data_list.append(duration)
pair_data_list.append(pair)
lookback_data_list.append(lookback)
delay_data_list.append(delay)
step_data_list.append(step)

# Chollet Chap 6, pg 211
'''


# In[430]:


# Due to the training time required for minute data, this dataset was not included.
'''
print('Generated Sample Array Shape:', next(train_gen_EUR_USD_min_week_hour_min)[0].shape)
print('Generated Target Array Shape', next(train_gen_EUR_USD_min_week_hour_min)[1].shape)
print('\nGenerated Sample and Target Array Exampled\n\n', next(train_gen_EUR_USD_min_week_hour_min))
'''


# In[431]:


# Due to the training time required for minute data, this dataset was not included.
'''
# Create generators for the following dataset.  I tried to do this through a function, but kept getting random nan values
# in the generator output.

duration = 'Minute'
pair = 'EUR_USD'
lookback = 60*24*7 # Look back 1 week
delay = 60*24 # Predict 1 day into future
step = 1 # Look at data every minute
batch_size = 128

# Set the input data for generators
train = train_EUR_USD_min
test = val_EUR_USD_min
val = test_EUR_USD_min

# Create the train data generator
train_gen = generator(train[:,1:6],
                      lookback = lookback,
                      delay = delay,
                      shuffle = True,
                      step = step,
                      batch_size = batch_size)

# Create the validation data generator 
test_gen = generator(val[:,1:6],
                    lookback = lookback,
                    delay = delay,
                    shuffle = False,
                    step = step,
                    batch_size = batch_size)

# Create the test data generator
val_gen = generator(test[:,1:6],
                     lookback = lookback,
                     delay = delay,
                     shuffle = False,
                     step = step,
                     batch_size = batch_size)

# Define the number of validation steps
val_steps = len(val) - lookback
print('Validation steps:', val_steps)

# Define the number of test steps
test_steps = len(test) - lookback
print('\nTest steps:', test_steps)

# Print info about these generators
print('\n*** The following Generators have been created ***')
print('Duration:', duration)
print('Pair:', pair)
print('Input step: %r %r' %(step, duration))
print('Forecast delay: %r %r' %(delay, duration))
print('Generators: train, val, test')

# Rename variable for this dataset
# format - variable type _ pair _ data window _ lookback _ forecast delay _ step
train_gen_EUR_USD_min_week_day_min = train_gen
val_gen_EUR_USD_min_week_day_min = val_gen
test_gen_EUR_USD_min_week_day_min = test_gen
val_steps_EUR_USD_min_week_day_min = val_steps
test_steps_EUR_USD_min_week_day_min = test_steps

# Append dataset extension to list for reference later
datasets.append('EUR_USD_min_week_day_min')

# Append generators to a list for use in model loops
train_gen_list.append(train_gen_EUR_USD_min_week_day_min)
val_gen_list.append(val_gen_EUR_USD_min_week_day_min)
test_gen_list.append(test_gen_EUR_USD_min_week_day_min)
val_steps_list.append(val_steps_EUR_USD_min_week_day_min)
test_steps_list.append(test_steps_EUR_USD_min_week_day_min)

# Append generator specs to allow for understanding of parameters of each dataset in future
duration_data_list.append(duration)
pair_data_list.append(pair)
lookback_data_list.append(lookback)
delay_data_list.append(delay)
step_data_list.append(step)

# Chollet Chap 6, pg 211
'''


# In[432]:


# Due to the training time required for minute data, this dataset was not included.
'''
print('Generated Sample Array Shape:', next(train_gen_EUR_USD_min_week_day_min)[0].shape)
print('Generated Target Array Shape', next(train_gen_EUR_USD_min_week_day_min)[1].shape)
print('\nGenerated Sample and Target Array Exampled\n\n', next(train_gen_EUR_USD_min_week_day_min))
'''


# In[433]:


# Create generators for the following dataset.  I tried to do this through a function, but kept getting random nan values
# in the generator output.

duration = 'Minute'
pair = 'EUR_USD'
lookback = 60*24*7 # Look back 1 week
delay = 60 # Predict 1 hour into future
step = 60 # Look at data every hour
batch_size = 128

# Set the input data for generators
train = train_EUR_USD_min
test = val_EUR_USD_min
val = test_EUR_USD_min

# Create the train data generator
train_gen = generator(train[:,1:6],
                      lookback = lookback,
                      delay = delay,
                      shuffle = True,
                      step = step,
                      batch_size = batch_size)

# Create the validation data generator 
test_gen = generator(val[:,1:6],
                    lookback = lookback,
                    delay = delay,
                    shuffle = False,
                    step = step,
                    batch_size = batch_size)

# Create the test data generator
val_gen = generator(test[:,1:6],
                     lookback = lookback,
                     delay = delay,
                     shuffle = False,
                     step = step,
                     batch_size = batch_size)

# Define the number of validation steps
val_steps = len(val) - lookback
print('Validation steps:', val_steps)

# Define the number of test steps
test_steps = len(test) - lookback
print('\nTest steps:', test_steps)

# Print info about these generators
print('\n*** The following Generators have been created ***')
print('Duration:', duration)
print('Pair:', pair)
print('Input step: %r %r' %(step, duration))
print('Forecast delay: %r %r' %(delay, duration))
print('Generators: train, val, test')

# Rename variable for this dataset
# format - variable type _ pair _ data window _ lookback _ forecast delay _ step
train_gen_EUR_USD_min_week_hour_hour = train_gen
val_gen_EUR_USD_min_week_hour_hour = val_gen
test_gen_EUR_USD_min_week_hour_hour = test_gen
val_steps_EUR_USD_min_week_hour_hour = val_steps
test_steps_EUR_USD_min_week_hour_hour = test_steps

# Append dataset extension to list for reference later
datasets.append('EUR_USD_min_week_hour_hour')

# Append generators to a list for use in model loops
train_gen_list.append(train_gen_EUR_USD_min_week_hour_hour)
val_gen_list.append(val_gen_EUR_USD_min_week_hour_hour)
test_gen_list.append(test_gen_EUR_USD_min_week_hour_hour)
val_steps_list.append(val_steps_EUR_USD_min_week_hour_hour)
test_steps_list.append(test_steps_EUR_USD_min_week_hour_hour)

# Append generator specs to allow for understanding of parameters of each dataset in future
duration_data_list.append(duration)
pair_data_list.append(pair)
lookback_data_list.append(lookback)
delay_data_list.append(delay)
step_data_list.append(step)

# Chollet Chap 6, pg 211


# In[434]:


print('Generated Sample Array Shape:', next(train_gen_EUR_USD_min_week_hour_hour)[0].shape)
print('Generated Target Array Shape', next(train_gen_EUR_USD_min_week_hour_hour)[1].shape)
print('\nGenerated Sample and Target Array Exampled\n\n', next(train_gen_EUR_USD_min_week_hour_hour))


# In[435]:


# Create generators for the following dataset.  I tried to do this through a function, but kept getting random nan values
# in the generator output.

duration = 'Minute'
pair = 'EUR_USD'
lookback = 60*24*7 # Look back 1 week
delay = 60*24 # Predict 1 day into future
step = 60 # Look at data every hour
batch_size = 128

# Set the input data for generators
train = train_EUR_USD_min
test = val_EUR_USD_min
val = test_EUR_USD_min

# Create the train data generator
train_gen = generator(train[:,1:6],
                      lookback = lookback,
                      delay = delay,
                      shuffle = True,
                      step = step,
                      batch_size = batch_size)

# Create the validation data generator 
test_gen = generator(val[:,1:6],
                    lookback = lookback,
                    delay = delay,
                    shuffle = False,
                    step = step,
                    batch_size = batch_size)

# Create the test data generator
val_gen = generator(test[:,1:6],
                     lookback = lookback,
                     delay = delay,
                     shuffle = False,
                     step = step,
                     batch_size = batch_size)

# Define the number of validation steps
val_steps = len(val) - lookback
print('Validation steps:', val_steps)

# Define the number of test steps
test_steps = len(test) - lookback
print('\nTest steps:', test_steps)

# Print info about these generators
print('\n*** The following Generators have been created ***')
print('Duration:', duration)
print('Pair:', pair)
print('Input step: %r %r' %(step, duration))
print('Forecast delay: %r %r' %(delay, duration))
print('Generators: train, val, test')

# Rename variable for this dataset
# format - variable type _ pair _ data window _ lookback _ forecast delay _ step
train_gen_EUR_USD_min_week_day_hour = train_gen
val_gen_EUR_USD_min_week_day_hour = val_gen
test_gen_EUR_USD_min_week_day_hour = test_gen
val_steps_EUR_USD_min_week_day_hour = val_steps
test_steps_EUR_USD_min_week_day_hour = test_steps

# Append dataset extension to list for reference later
datasets.append('EUR_USD_min_week_day_hour')

# Append generators to a list for use in model loops
train_gen_list.append(train_gen_EUR_USD_min_week_day_hour)
val_gen_list.append(val_gen_EUR_USD_min_week_day_hour)
test_gen_list.append(test_gen_EUR_USD_min_week_day_hour)
val_steps_list.append(val_steps_EUR_USD_min_week_day_hour)
test_steps_list.append(test_steps_EUR_USD_min_week_day_hour)

# Append generator specs to allow for understanding of parameters of each dataset in future
duration_data_list.append(duration)
pair_data_list.append(pair)
lookback_data_list.append(lookback)
delay_data_list.append(delay)
step_data_list.append(step)

# Chollet Chap 6, pg 211


# In[436]:


print('Generated Sample Array Shape:', next(train_gen_EUR_USD_min_week_day_hour)[0].shape)
print('Generated Target Array Shape', next(train_gen_EUR_USD_min_week_day_hour)[1].shape)
print('\nGenerated Sample and Target Array Exampled\n\n', next(train_gen_EUR_USD_min_week_day_hour))


# In[437]:


print('Length of list: ', len(datasets))
datasets


# In[438]:


print('Length of list: ', len(train_gen_list))
train_gen_list


# In[439]:


print('Length of list: ', len(val_gen_list))
val_gen_list


# In[440]:


print('Length of list: ', len(test_gen_list))
test_gen_list


# In[441]:


print('Length of list: ', len(val_steps_list))
val_steps_list


# In[442]:


print('Length of list: ', len(test_steps_list))
test_steps_list


# In[443]:


print('Length of list: ', len(duration_data_list))
duration_data_list


# In[444]:


print('Length of list: ', len(pair_data_list))
pair_data_list


# In[445]:


print('Length of list: ', len(lookback_data_list))
lookback_data_list


# In[446]:


print('Length of list: ', len(delay_data_list))
delay_data_list


# In[447]:


print('Length of list: ', len(step_data_list))
step_data_list


# #### Plot the Data
# 
# Functions will be created to plot the training, validation, and test data in the experiement.  Bokeh will be used to produce interactive plots that allow for zooming in or selecting specific timeframes to explore.

# In[53]:


# Create a function to plot price per time series.

def plot_data(time, pair, x1_variable, y1_variable, x2_variable, y2_variable):
    x1 = x1_variable # Set the time series variable to "x"
    y1 = y1_variable # Set the first price series variable 
    x2 = x2_variable
    y2 = y2_variable # Plot a second price series variable, such as a predicted price
    
    #from bokeh.models import HoverTool

    TOOLS = "hover,pan,wheel_zoom,box_zoom,reset,save" #Define the tools for the plot
        
    if np.all(y2 == None): # Use "if" to test if y2 exists.  If not, only plot y1
        p = figure(tools=TOOLS, title='%s Data For %s Pair' %(time,pair), plot_width=900, plot_height=300)
        r = p.line(x1, y1, color='green', line_width=0.5,legend='Actual')
    else: # If y2 exists, plot y1 and y2
        p = figure(tools=TOOLS, title='%s Data For %s Pair' %(time,pair), plot_width=900, plot_height=300)
        r = p.line(x1, y1, color='green', line_width=0.5,legend='Actual')
        r1 = p.line(x2, y2, color='blue', line_width=0.5,legend='Predicted')
    
    p.xaxis.axis_label = 'Time(%s)' %time
    p.xaxis.formatter = DatetimeTickFormatter(
        seconds="%d %B %Y",
        minutes="%d %B %Y",
        hours="%d %b %Y",
        days="%d %b %Y",
        months="%d %b %Y",
        years="%d %b %Y"
        ) # Plot the x-axis as day/month/year
    p.yaxis.axis_label = 'Exchange Rate'
    p.legend.location = 'top_right'
    
    '''hover = p.select(dict(type=HoverTool)) # formats the dat
    hover.tooltips = [("Date", "@x{%Y-%m-%d %H:%M:%S}"), ("Bid Close", "@y")]
    hover.formatters = { "x": "datetime"}
    hover.mode = 'vline'
    '''
    
    hover = HoverTool(
                    tooltips = [
                        ("Date", "@x{%Y-%m-%d %H:%M:%S}"),
                        ("Bid Close", "@y"),
                    ],
                    formatters={
                        'x': 'datetime',
                    },    
    )
    p.add_tools(hover)
    
    show(p)


# In[54]:


plot_data('Minute', 'EUR_USD', EUR_USD_min_df.index, EUR_USD_min_df['bid_low'], None, None)


# In[55]:


plot_data('Hourly', 'EUR_USD', train_EUR_USD_min[:,0], train_EUR_USD_min[:,1],val_EUR_USD_min[:,0], val_EUR_USD_min[:,1])


# #### Experiment Baseline - Baseline Estimates
# 
# Add Text

# In[56]:


# Create a function to calculate the naive baseline.  This forecast assumes that the next observation will be equal
# to the previous observation.

def evaluate_naive_method():
    batch_maes = [] # Create list to hold maes for each batch
    target_list = []
    pred_list = []
    for step in range(val_steps): # Each step is a batch
        samples, targets = next(val_gen) # generates samples and targets per generator constraints
        preds = samples[:, -1, 3] # Looks at all sub-batches in sample [:,,], selects the last observation in that 
                                    #batch [,-1,], and returns only the Bid Close [,,3] as a prediction.
        mae = np.mean(np.abs(preds - targets)) # Computes the mean absolute error between the prediction and the target
        batch_maes.append(mae) # Appends the mae for this batch to the list
        target_list.append(targets)
        pred_list.append(preds)
    print('MAE: ', np.mean(batch_maes)) # Prints the mean mae for all batches
    #return target_list, pred_list


# In[57]:


for step, gen, dataset_name in zip(val_steps_list, val_gen_list, datasets):
        
    val_steps = step
    val_gen = gen
    dataset = dataset_name
    
    print('\nDataset: ', dataset)

    start_time = time.clock() # start timer to evaluate how long it takes to train this base model

    evaluate_naive_method()

    end_time = time.clock() # end timer
    runtime = end_time - start_time  # seconds of wall-clock time 
    print("Processing time (seconds): %f" % runtime)  # print process time to train model


# ### Functions to Facilitate Modeling

# In[58]:


# Define a function to plot Model Accuracy and Model Loss

def plot_figs(model_id, model_type, mod_nodes):
    # Create plot figure
    plt.figure(figsize=(8,3))
    
    # Calculate accuracy and loss variables
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    
    epochs = range(1, len(loss)+1)
    
    # Plot training and validation loss
    plt.plot(epochs, loss, 'bo', label='Training loss')
    plt.plot(epochs, val_loss, 'm', label='Validation loss')
    plt.legend()
    plt.title('Model Loss for Model No. %r,\n Model Type %r, \nNo. of Nodes: %r'          %(model_id, model_type, mod_nodes)) 
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    
    plt.show()


# ### Experiment #1 - Basic Fully Connected Dense Model

# In[566]:


# Specify attributes about this model
exp_no = 1 # Experiment Number
model_type = 'Dense NN'
#duration = 'Hourly'
#pair = 'EUR_USD'
#lookback = 1440 # Lookback 1440 observations [60 min], which is one day.
#step = 60 # Step of 60 observations [60 min]
#delay = 60 # Predict 60 observations [60 min] into future
batch_size = 128
mod_layers = 1
mod_nodes = [16, 32, 64]
hidden_activation = ['sigmoid','relu','tanh']
out_activation = 'softmax'
optimizer = 'rmsprop'
mod_epochs = 5

'''train_gen = train_gen_list[2] 
val_gen = val_gen_list[2]  
val_steps = val_steps_list[2]
datasets = datasets[2]
duration = duration_list[2]
pair = pair_list[2]
lookback = lookback_list[2]
step = step_list[2]
delay = delay_list[2]'''


# In[567]:


# Create lists to hold the results of each iteration.
model_id = 1 # initial model ID
model_id_list = []
model_type_list = []
duration_list = []
pair_list = []
lookback_list = []
step_list = []
delay_list = []
n_layer_list = []
node_list = []
hidden_activation_list = []
out_activation_list = []
optimizer_list = []
train_loss_list = []
valid_loss_list = []
max_epoch_list = []
epoch_list = []
batch_size_list = []
processing_time = []


# In[138]:


# Create a loop to iterate through model variations
for nodes in mod_nodes:
    for hid_act in hidden_activation:
                
        # Print the basic parameters of the model being iterated
        print('---------------------------------------------------------------')    
        print("Model No.: %r" %model_id)
        print("Model Type: %r" %model_type)
        print("Dataset: %r" %dataset)
        print("Duration: %r" %duration)
        print("Lookback: %r" %lookback)
        print("Step: %r" %step)
        print("Delay: %r" %delay)
        print("Number of Layers: %r" %mod_layers)
        print("Number of Nodes: %r" %nodes)
        print("Hidden Layer Activiation: %r" %hid_act)
        print("Hidden Layer Optimizer: %r" %optimizer)

        start_time = time.clock() # start timer to evaluate how long it takes to train this base model

        # Set seeds for reproducible results
        seed(1)
        set_random_seed(2)
        rn.seed(3)


        # Define the model structure that is to be evaluated.
        model = Sequential()
        model.add(Flatten(input_shape=(lookback // step, train[:,1:6].shape[-1])))
        model.add(Dense(nodes, activation=hid_act))
        model.add(Dense(1))
        #print(model.summary())
        model.compile(optimizer='rmsprop',
                      loss='mae')

        history = model.fit_generator(train_gen,
                                      steps_per_epoch=500,
                                      epochs=mod_epochs,
                                      validation_data=val_gen,
                                      validation_steps=val_steps)

        #model.save_weights('pre_trained_glove_model.h5')

        # Stop the timer and append time to list
        end_time = time.clock() # end timer
        runtime = end_time - start_time  # seconds of wall-clock time 
        print("\nProcessing time (seconds): %f" % runtime)  # print process time to train model        
        processing_time.append(runtime) # append process time results to list for comparison table

        # Identify and print maximum validation loss and epoch where that accuracy was realized
        epoch_max = np.argmin(history.history['val_loss'])
        val_loss_max = min(history.history['val_loss'])
        print('Maximum Validation Loss: %r, Epochs Required to Achieve this Loss: %r' %(val_loss_max, epoch_max))

        # Print validation accuracy
        train_loss_max = min(history.history['loss'])
        print("Maximum Train Loss: %r" %train_loss_max) 

        # Plot the accuracy and loss of each model iteration
        plot_figs(model_id, model_type, nodes)

        # Append lists with values for this model ID
        model_id_list.append(model_id)
        model_type_list.append(model_type)
        duration_list.append(duration)
        pair_list.append(pair)
        lookback_list.append(lookback)
        step_list.append(step)
        delay_list.append(delay)
        n_layer_list.append(mod_layers)
        node_list.append(nodes)
        hidden_activation_list.append(hid_act)
        out_activation_list.append(out_activation)
        optimizer_list.append(optimizer)
        train_loss_list.append(train_loss_max)
        valid_loss_list.append(val_loss_max)
        max_epoch_list.append(epoch_max)
        epoch_list.append(mod_epochs)
        batch_size_list.append(batch_size)

        model_id += 1


# In[ ]:


# Create a loop to iterate through model variations
for nodes in mod_nodes:
    for hid_act in hidden_activation:
        for train_gen, val_gen, val_steps, dataset, duration, pair, lookback, step, delay in zip(train_gen_list, 
                                                                                                 val_gen_list,  
                                                                                                 val_steps_list, 
                                                                                                 datasets,
                                                                                                 duration_data_list,
                                                                                                 pair_data_list,
                                                                                                 lookback_data_list,
                                                                                                 step_data_list,
                                                                                                 delay_data_list
                                                                                                ):
        
            # Print the basic parameters of the model being iterated
            print('---------------------------------------------------------------')    
            print("Model No.: %r" %model_id)
            print("Model Type: %r" %model_type)
            print("Dataset: %r" %dataset)
            print("Duration: %r" %duration)
            print("Lookback: %r" %lookback)
            print("Step: %r" %step)
            print("Delay: %r" %delay)
            print("Number of Layers: %r" %mod_layers)
            print("Number of Nodes: %r" %nodes)
            print("Hidden Layer Activiation: %r" %hid_act)
            print("Hidden Layer Optimizer: %r" %optimizer)

            start_time = time.clock() # start timer to evaluate how long it takes to train this base model

            # Set seeds for reproducible results
            seed(1)
            set_random_seed(2)
            rn.seed(3)


            # Define the model structure that is to be evaluated.
            model = Sequential()
            model.add(Flatten(input_shape=(lookback // step, train[:,1:6].shape[-1])))
            model.add(Dense(nodes, activation=hid_act))
            model.add(Dense(1))
            #print(model.summary())
            model.compile(optimizer='rmsprop',
                          loss='mae')

            history = model.fit_generator(train_gen,
                                          steps_per_epoch=500,
                                          epochs=mod_epochs,
                                          validation_data=val_gen,
                                          validation_steps=val_steps)

            #model.save_weights('pre_trained_glove_model.h5')

            # Stop the timer and append time to list
            end_time = time.clock() # end timer
            runtime = end_time - start_time  # seconds of wall-clock time 
            print("\nProcessing time (seconds): %f" % runtime)  # print process time to train model        
            processing_time.append(runtime) # append process time results to list for comparison table

            # Identify and print maximum validation loss and epoch where that accuracy was realized
            epoch_max = np.argmin(history.history['val_loss'])
            val_loss_max = min(history.history['val_loss'])
            print('Maximum Validation Loss: %r, Epochs Required to Achieve this Loss: %r' %(val_loss_max, epoch_max))

            # Print validation accuracy
            train_loss_max = min(history.history['loss'])
            print("Maximum Train Loss: %r" %train_loss_max) 

            # Plot the accuracy and loss of each model iteration
            plot_figs(model_id, model_type, nodes)

            # Append lists with values for this model ID
            model_id_list.append(model_id)
            model_type_list.append(model_type)
            duration_list.append(duration)
            pair_list.append(pair)
            lookback_list.append(lookback)
            step_list.append(step)
            delay_list.append(delay)
            n_layer_list.append(mod_layers)
            node_list.append(nodes)
            hidden_activation_list.append(hid_act)
            out_activation_list.append(out_activation)
            optimizer_list.append(optimizer)
            train_loss_list.append(train_loss_max)
            valid_loss_list.append(val_loss_max)
            max_epoch_list.append(epoch_max)
            epoch_list.append(mod_epochs)
            batch_size_list.append(batch_size)

            model_id += 1


# In[189]:


# round numbers in lists for readability
processing_time_formatted = [ '%.3f' % elem for elem in processing_time]
train_loss_list_formatted = [ '%.4f' % elem for elem in train_loss_list]
valid_loss_list_formatted = [ '%.4f' % elem for elem in valid_loss_list]


# In[194]:


# Create a dictionary to store the results of the trained models from code above.
data = OrderedDict([('Model ID', model_id_list),
                    ('Model Type', model_type_list),
                    ('Duration', duration_list),
                    ('Pair', pair_list),
                    ('Lookback', lookback_list),
                    ('Step', step_list),
                    ('Delay', delay_list),
                    ('No. of Layers', n_layer_list),
                    ('Nodes', node_list),
                    ('Hidden Layer Act.', hidden_activation_list),
                    ('Output Layer Activation', out_activation_list),
                    ('Optimizer', optimizer_list),
                    ('Process Time', processing_time_formatted),
                    ('Train Acc.', train_loss_list_formatted),
                    ('Val Acc.', valid_loss_list_formatted),
                    ('No. of Epochs to Max Acc.',epoch_max),
                    ('Batch Size', batch_size_list)])


# In[196]:


# Pickle the data for use in the future without having to run models again.  Write to binary file
file_name = ('Exp_%s_Results.pickle' %exp_no)
with open(file_name, 'wb') as f:
    # Pickle the 'data' dictionary using the highest protocol available.
    pickle.dump(data, f, pickle.HIGHEST_PROTOCOL)

print('\n Run complete. Data objects sent to binary file %r' %file_name)


# In[197]:


# Extract objects from the dictionary object data
file_name = 'Exp_1_Results.pickle'
pickle_in = open(file_name,"rb")
data = pickle.load(pickle_in)

print('\n Loading of data complete. The binary file %r has been loaded' %file_name)


# In[202]:


# Create a table to display results of this experiment
results = pd.DataFrame(data)
print('\nTable 1: Benchmark Experiment: 1 Fully Connected Dense Neural Network\n')
results 


# ### Experiment #2 - Explore a Basic RNN

# In[280]:


# Specify attributes about this model
exp_no = 2 # Experiment Number
model_type = 'RNN'
#duration = 'Hourly'
#pair = 'EUR_USD'
#lookback = 1440 # Lookback 1440 observations [60 min], which is one day.
#step = 60 # Step of 60 observations [60 min]
#delay = 60 # Predict 60 observations [60 min] into future
batch_size = 128
mod_layers = 1
mod_nodes = [16, 32, 64]
hidden_activation = 'N/A'
out_activation = 'N/A'
optimizer = ['sgd','adam','adagrad','rmsprop']
mod_epochs = 5

#train_gen = train_gen_list[2] 
#val_gen = val_gen_list[2]  
#val_steps = val_steps_list[2]
#dataset = datasets[2]
#duration = duration_list[2]
#pair = pair_list[2]
#lookback = lookback_list[2]
#step = step_list[2]
#delay = delay_list[2]


# In[281]:


# Create lists to hold the results of each iteration.
model_id = 1 # initial model ID
model_id_list = []
model_type_list = []
#duration_list = []
#pair_list = []
#lookback_list = []
#step_list = []
#delay_list = []
n_layer_list = []
node_list = []
hidden_activation_list = []
out_activation_list = []
optimizer_list = []
train_loss_list = []
valid_loss_list = []
max_epoch_list = []
epoch_list = []
batch_size_list = []
processing_time = []


# In[282]:


# Create a loop to iterate through model variations
for nodes in mod_nodes:
    for opt in optimizer:
        for train_gen, val_gen, val_steps, dataset, duration, pair, lookback, step, delay in zip(train_gen_list, 
                                                                                                 val_gen_list,  
                                                                                                 val_steps_list, 
                                                                                                 datasets,
                                                                                                 duration_data_list,
                                                                                                 pair_data_list,
                                                                                                 lookback_data_list,
                                                                                                 step_data_list,
                                                                                                 delay_data_list
                                                                                                ):
                
            # Print the basic parameters of the model being iterated
            print('---------------------------------------------------------------')    
            print("Model No.: %r" %model_id)
            print("Model Type: %r" %model_type)
            print("Dataset: %r" %dataset)
            print("Duration: %r" %duration)
            print("Lookback: %r" %lookback)
            print("Step: %r" %step)
            print("Delay: %r" %delay)
            print("Number of Layers: %r" %mod_layers)
            print("Number of Nodes: %r" %nodes)
            print("Hidden Layer Optimizer: %r" %opt)

            start_time = time.clock() # start timer to evaluate how long it takes to train this base model

            # Set seeds for reproducible results
            seed(1)
            set_random_seed(2)
            rn.seed(3)


            # Define the model structure that is to be evaluated.
            model = Sequential()
            model.add(SimpleRNN(nodes, input_shape=(None, train[:,1:6].shape[-1])))
            model.add(Dense(1))
            #print(model.summary())
            model.compile(optimizer=opt,
                          loss='mae')

            history = model.fit_generator(train_gen,
                                          steps_per_epoch=500,
                                          epochs=mod_epochs,
                                          validation_data=val_gen,
                                          validation_steps=val_steps)

            file_name = ('Exp_%s_Mod_%s_Results' %(exp_no, model_id))
            model.save_weights(file_name)

            # Stop the timer and append time to list
            end_time = time.clock() # end timer
            runtime = end_time - start_time  # seconds of wall-clock time 
            print("\nProcessing time (seconds): %f" % runtime)  # print process time to train model        
            processing_time.append(runtime) # append process time results to list for comparison table

            # Identify and print maximum validation loss and epoch where that accuracy was realized
            epoch_max = np.argmin(history.history['val_loss'])
            val_loss_max = min(history.history['val_loss'])
            print('Maximum Validation Loss: %r, Epochs Required to Achieve this Loss: %r' %(val_loss_max, epoch_max))

            # Print validation accuracy
            train_loss_max = min(history.history['loss'])
            print("Maximum Train Loss: %r" %train_loss_max) 

            # Plot the accuracy and loss of each model iteration
            plot_figs(model_id, model_type, nodes)

            # Append lists with values for this model ID
            model_id_list.append(model_id)
            model_type_list.append(model_type)
            duration_list.append(duration)
            pair_list.append(pair)
            lookback_list.append(lookback)
            step_list.append(step)
            delay_list.append(delay)
            n_layer_list.append(mod_layers)
            node_list.append(nodes)
            hidden_activation_list.append(hidden_activation)
            out_activation_list.append(out_activation)
            optimizer_list.append(opt)
            train_loss_list.append(train_loss_max)
            valid_loss_list.append(val_loss_max)
            max_epoch_list.append(epoch_max)
            epoch_list.append(mod_epochs)
            batch_size_list.append(batch_size)

            model_id += 1


# In[283]:


# round numbers in lists for readability
processing_time_formatted = [ '%.3f' % elem for elem in processing_time]
train_loss_list_formatted = [ '%.4f' % elem for elem in train_loss_list]
valid_loss_list_formatted = [ '%.4f' % elem for elem in valid_loss_list]


# In[314]:


# Create a dictionary to store the results of the trained models from code above.
data = OrderedDict([('Model ID', model_id_list),
                    ('Model Type', model_type_list),
                    ('Duration', duration_list),
                    ('Pair', pair_list),
                    ('Lookback', lookback_list),
                    ('Step', step_list),
                    ('Delay', delay_list),
                    ('No. of Layers', n_layer_list),
                    ('Nodes', node_list),
                    ('Hidden Layer Act.', hidden_activation_list),
                    ('Output Layer Activation', out_activation_list),
                    ('Optimizer', optimizer_list),
                    ('Process Time', processing_time_formatted),
                    ('Train Acc.', train_loss_list_formatted),
                    ('Val Acc.', valid_loss_list_formatted),
                    ('No. of Epochs to Max Acc.',epoch_max),
                    ('Batch Size', batch_size_list)])


# In[335]:


# Pickle the data for use in the future without having to run models again.  Write to binary file
file_name = ('Exp_%s_Results.pickle' %exp_no)
with open(file_name, 'wb') as f:
    # Pickle the 'data' dictionary using the highest protocol available.
    pickle.dump(data, f, pickle.HIGHEST_PROTOCOL)

print('\n Run complete. Data objects sent to binary file %r' %file_name)


# In[336]:


# Extract objects from the dictionary object data
file_name = 'Exp_2_Results.pickle'
pickle_in = open(file_name,"rb")
data = pickle.load(pickle_in)

print('\n Loading of data complete. The binary file %r has been loaded' %file_name)


# In[338]:


data.update({'Hidden Layer Act.': ['N/A'] * 48})


# In[339]:


data.update({'Output Layer Activation': ['N/A'] * 48})


# In[340]:


## Create a table to display results of this experiment
results = pd.DataFrame(data)
print('\nTable 2: Benchmark Experiment: Basic Recurrent Neural Network\n')
results 


# In[328]:


## Create a table to display results of this experiment
results = pd.DataFrame(data1)
print('\nTable 2: Benchmark Experiment: Basic Recurrent Neural Network\n')
results 


# In[ ]:





# ### Experiment #3 - Compare Basic RNN to LSTM and GRU Models

# In[55]:


# Specify attributes about this model
exp_no = 3 # Experiment Number
model_type = ['LSTM', 'GRU']
layer_type = ['LSTM', 'GRU']
#duration = 'Hourly'
#pair = 'EUR_USD'
#lookback = 1440 # Lookback 1440 observations [60 min], which is one day.
#step = 60 # Step of 60 observations [60 min]
#delay = 60 # Predict 60 observations [60 min] into future
batch_size = 128
mod_layers = 1
mod_nodes = 16
hidden_activation = 'N/A'
out_activation = 'N/A'
optimizer = ['adam','adagrad']
mod_epochs = 5

#train_gen = train_gen_list[2] 
#val_gen = val_gen_list[2]  
#val_steps = val_steps_list[2]
#dataset = datasets[2]
#duration = duration_list[2]
#pair = pair_list[2]
#lookback = lookback_list[2]
#step = step_list[2]
#delay = delay_list[2]


# In[56]:


# Create lists to hold the results of each iteration.
model_id = 1 # initial model ID
model_id_list = []
model_type_list = []
duration_list = []
pair_list = []
lookback_list = []
step_list = []
delay_list = []
n_layer_list = []
node_list = []
hidden_activation_list = []
out_activation_list = []
optimizer_list = []
train_loss_list = []
valid_loss_list = []
max_epoch_list = []
epoch_list = []
batch_size_list = []
processing_time = []


# In[57]:


# Create a loop to iterate through model variations

for opt in optimizer:
    for layer, model_type in zip(layer_type, model_type):
        for train_gen, val_gen, val_steps, dataset, duration, pair, lookback, step, delay in zip(train_gen_list, 
                                                                                                 val_gen_list,  
                                                                                                 val_steps_list, 
                                                                                                 datasets,
                                                                                                 duration_data_list,
                                                                                                 pair_data_list,
                                                                                                 lookback_data_list,
                                                                                                 step_data_list,
                                                                                                 delay_data_list
                                                                                                ):

            # Print the basic parameters of the model being iterated
            print('---------------------------------------------------------------')    
            print("Model No.: %r" %model_id)
            print("Model Type: %r" %model_type)
            print("Dataset: %r" %dataset)
            print("Duration: %r" %duration)
            print("Lookback: %r" %lookback)
            print("Step: %r" %step)
            print("Delay: %r" %delay)
            print("Number of Layers: %r" %mod_layers)
            print("Number of Nodes: %r" %mod_nodes)
            print("Hidden Layer Optimizer: %r" %opt)

            start_time = time.clock() # start timer to evaluate how long it takes to train this base model

            # Set seeds for reproducible results
            seed(1)
            set_random_seed(2)
            rn.seed(3)


            # Define the model structure that is to be evaluated.
            model = Sequential()
            if layer == 'LSTM':
                model.add(LSTM(mod_nodes, input_shape=(None, train[:,1:6].shape[-1])))
            if layer == 'GRU':
                model.add(GRU(mod_nodes, input_shape=(None, train[:,1:6].shape[-1])))
            model.add(Dense(1))
            #print(model.summary())
            model.compile(optimizer=opt,
                          loss='mae')

            history = model.fit_generator(train_gen,
                                          steps_per_epoch=500,
                                          epochs=mod_epochs,
                                          validation_data=val_gen,
                                          validation_steps=val_steps)

            file_name = ('Exp_%s_Mod_%s_Results' %(exp_no, model_id))
            model.save_weights(file_name)

            # Stop the timer and append time to list
            end_time = time.clock() # end timer
            runtime = end_time - start_time  # seconds of wall-clock time 
            print("\nProcessing time (seconds): %f" % runtime)  # print process time to train model        
            processing_time.append(runtime) # append process time results to list for comparison table

            # Identify and print maximum validation loss and epoch where that accuracy was realized
            epoch_max = np.argmin(history.history['val_loss'])
            val_loss_max = min(history.history['val_loss'])
            print('Maximum Validation Loss: %r, Epochs Required to Achieve this Loss: %r' %(val_loss_max, epoch_max))

            # Print validation accuracy
            train_loss_max = min(history.history['loss'])
            print("Maximum Train Loss: %r" %train_loss_max) 

            # Plot the accuracy and loss of each model iteration
            plot_figs(model_id, model_type, mod_nodes)

            # Append lists with values for this model ID
            model_id_list.append(model_id)
            model_type_list.append(model_type)
            duration_list.append(duration)
            pair_list.append(pair)
            lookback_list.append(lookback)
            step_list.append(step)
            delay_list.append(delay)
            n_layer_list.append(mod_layers)
            node_list.append(mod_nodes)
            hidden_activation_list.append(hidden_activation)
            out_activation_list.append(out_activation)
            optimizer_list.append(opt)
            train_loss_list.append(train_loss_max)
            valid_loss_list.append(val_loss_max)
            max_epoch_list.append(epoch_max)
            epoch_list.append(mod_epochs)
            batch_size_list.append(batch_size)

            model_id += 1


# In[98]:


# round numbers in lists for readability
processing_time_formatted = [ '%.3f' % elem for elem in processing_time]
train_loss_list_formatted = [ '%.4f' % elem for elem in train_loss_list]
valid_loss_list_formatted = [ '%.4f' % elem for elem in valid_loss_list]


# In[123]:


# Create a dictionary to store the results of the trained models from code above.
data = OrderedDict([('Model ID', model_id_list),
                    ('Model Type', model_type_list),
                    ('Duration', duration_list),
                    ('Pair', pair_list),
                    ('Lookback', lookback_list),
                    ('Step', step_list),
                    ('Delay', delay_list),
                    ('No. of Layers', n_layer_list),
                    ('Nodes', node_list),
                    ('Hidden Layer Act.', hidden_activation_list),
                    ('Output Layer Activation', out_activation_list),
                    ('Optimizer', optimizer_list),
                    ('Process Time', processing_time_formatted),
                    ('Train Acc.', train_loss_list_formatted),
                    ('Val Acc.', valid_loss_list_formatted),
                    ('No. of Epochs to Max Acc.',epoch_max),
                    ('Batch Size', batch_size_list)])


# In[124]:


# Pickle the data for use in the future without having to run models again.  Write to binary file
file_name = ('Exp_%s_Results.pickle' %exp_no)
with open(file_name, 'wb') as f:
    # Pickle the 'data' dictionary using the highest protocol available.
    pickle.dump(data, f, pickle.HIGHEST_PROTOCOL)

print('\n Run complete. Data objects sent to binary file %r' %file_name)


# In[125]:


# Extract objects from the dictionary object data
file_name = 'Exp_3_Results.pickle'
pickle_in = open(file_name,"rb")
data = pickle.load(pickle_in)

print('\n Loading of data complete. The binary file %r has been loaded' %file_name)


# In[126]:


## Create a table to display results of this experiment
results = pd.DataFrame(data)
print('\nTable 3: Benchmark Experiment: 1-Layer RNN Comparison of LSTM and GRU Model\n')
results 


# ### Experiment #4 - Evaluate Impacts of Dropout

# In[124]:


# Specify attributes about this model
exp_no = 4 # Experiment Number
model_type = 'RNN'
layer_type = 'RNN'
#duration = 'Hourly'
#pair = 'EUR_USD'
#lookback = 1440 # Lookback 1440 observations [60 min], which is one day.
#step = 60 # Step of 60 observations [60 min]
#delay = 60 # Predict 60 observations [60 min] into future
batch_size = 128
mod_layers = 1
mod_nodes = 16
dropout_rate = [0.1, 0.3, 0.5]
recurrent_dropout_rate = [0.1, 0.3, 0.5]
optimizer = 'adam'
mod_epochs = 5

#train_gen = train_gen_list[2] 
#val_gen = val_gen_list[2]  
#val_steps = val_steps_list[2]
#dataset = datasets[2]
#duration = duration_list[2]
#pair = pair_list[2]
#lookback = lookback_list[2]
#step = step_list[2]
#delay = delay_list[2]


# In[125]:


# Create lists to hold the results of each iteration.
model_id = 1 # initial model ID
model_id_list = []
model_type_list = []
duration_list = []
pair_list = []
lookback_list = []
step_list = []
delay_list = []
n_layer_list = []
node_list = []
dropout_rate_list = []
recurrent_dropout_rate_list = []
optimizer_list = []
train_loss_list = []
valid_loss_list = []
max_epoch_list = []
epoch_list = []
batch_size_list = []
processing_time = []


# In[126]:


# Create a loop to iterate through model variations
for dropout in dropout_rate:
    for recurrent_dropout in recurrent_dropout_rate:
        for train_gen, val_gen, val_steps, dataset, duration, pair, lookback, step, delay in zip(train_gen_list, 
                                                                                                 val_gen_list,  
                                                                                                 val_steps_list, 
                                                                                                 datasets,
                                                                                                 duration_data_list,
                                                                                                 pair_data_list,
                                                                                                 lookback_data_list,
                                                                                                 step_data_list,
                                                                                                 delay_data_list
                                                                                                ):

            # Print the basic parameters of the model being iterated
            print('---------------------------------------------------------------')    
            print("Model No.: %r" %model_id)
            print("Model Type: %r" %model_type)
            print("Dataset: %r" %dataset)
            print("Duration: %r" %duration)
            print("Lookback: %r" %lookback)
            print("Step: %r" %step)
            print("Delay: %r" %delay)
            print("Number of Layers: %r" %mod_layers)
            print("Number of Nodes: %r" %mod_nodes)
            print("Dropout Rate: %r" %dropout)
            print("Recurrent Dropout Rate: %r" %recurrent_dropout)
            print("Hidden Layer Optimizer: %r" %optimizer)

            start_time = time.clock() # start timer to evaluate how long it takes to train this base model

            # Set seeds for reproducible results
            seed(1)
            set_random_seed(2)
            rn.seed(3)


            # Define the model structure that is to be evaluated.
            model = Sequential()
            model.add(SimpleRNN(mod_nodes, input_shape=(None, train[:,1:6].shape[-1]),
                                dropout=dropout,
                                recurrent_dropout=recurrent_dropout))
            model.add(Dense(1))
            #print(model.summary())
            model.compile(optimizer=optimizer,
                          loss='mae')

            history = model.fit_generator(train_gen,
                                          steps_per_epoch=500,
                                          epochs=mod_epochs,
                                          validation_data=val_gen,
                                          validation_steps=val_steps)

            file_name = ('Exp_%s_Mod_%s_Results' %(exp_no, model_id))
            model.save_weights(file_name)

            # Stop the timer and append time to list
            end_time = time.clock() # end timer
            runtime = end_time - start_time  # seconds of wall-clock time 
            print("\nProcessing time (seconds): %f" % runtime)  # print process time to train model        
            processing_time.append(runtime) # append process time results to list for comparison table

            # Identify and print maximum validation loss and epoch where that accuracy was realized
            epoch_max = np.argmin(history.history['val_loss'])
            val_loss_max = min(history.history['val_loss'])
            print('Maximum Validation Loss: %r, Epochs Required to Achieve this Loss: %r' %(val_loss_max, epoch_max))

            # Print validation accuracy
            train_loss_max = min(history.history['loss'])
            print("Maximum Train Loss: %r" %train_loss_max) 

            # Plot the accuracy and loss of each model iteration
            plot_figs(model_id, model_type, mod_nodes)

            # Append lists with values for this model ID
            model_id_list.append(model_id)
            model_type_list.append(model_type)
            duration_list.append(duration)
            pair_list.append(pair)
            lookback_list.append(lookback)
            step_list.append(step)
            delay_list.append(delay)
            n_layer_list.append(mod_layers)
            node_list.append(mod_nodes)
            dropout_rate_list.append(dropout)
            recurrent_dropout_rate_list.append(recurrent_dropout)
            optimizer_list.append(optimizer)
            train_loss_list.append(train_loss_max)
            valid_loss_list.append(val_loss_max)
            max_epoch_list.append(epoch_max)
            epoch_list.append(mod_epochs)
            batch_size_list.append(batch_size)

            model_id += 1


# In[127]:


# round numbers in lists for readability
processing_time_formatted = [ '%.3f' % elem for elem in processing_time]
train_loss_list_formatted = [ '%.4f' % elem for elem in train_loss_list]
valid_loss_list_formatted = [ '%.4f' % elem for elem in valid_loss_list]


# In[128]:


# Create a dictionary to store the results of the trained models from code above.
data = OrderedDict([('Model ID', model_id_list),
                    ('Model Type', model_type_list),
                    ('Duration', duration_list),
                    ('Pair', pair_list),
                    ('Lookback', lookback_list),
                    ('Step', step_list),
                    ('Delay', delay_list),
                    ('No. of Layers', n_layer_list),
                    ('Nodes', node_list),
                    ('Dropout Rate', dropout_rate_list),
                    ('Recurrent Dropout Rate', recurrent_dropout_rate_list),
                    ('Optimizer', optimizer_list),
                    ('Process Time', processing_time_formatted),
                    ('Train Acc.', train_loss_list_formatted),
                    ('Val Acc.', valid_loss_list_formatted),
                    ('No. of Epochs to Max Acc.',epoch_max),
                    ('Batch Size', batch_size_list)])


# In[129]:


# Pickle the data for use in the future without having to run models again.  Write to binary file
file_name = ('Exp_%s_Results.pickle' %exp_no)
with open(file_name, 'wb') as f:
    # Pickle the 'data' dictionary using the highest protocol available.
    pickle.dump(data, f, pickle.HIGHEST_PROTOCOL)

print('\n Run complete. Data objects sent to binary file %r' %file_name)


# In[130]:


# Extract objects from the dictionary object data
file_name = 'Exp_4_Results.pickle'
pickle_in = open(file_name,"rb")
data = pickle.load(pickle_in)

print('\n Loading of data complete. The binary file %r has been loaded' %file_name)


# In[131]:


## Create a table to display results of this experiment
results = pd.DataFrame(data)
print('\nTable 4: Benchmark Experiment: 1-Layer RNN Comparing Dropout Rates\n')
results 


# ### Experiment #5- Evaluate Multi-Layer RNNs

# In[275]:


# Specify attributes about this model
exp_no = 5 # Experiment Number
model_type = 'RNN'
layer_type = 'RNN'
#duration = 'Hourly'
#pair = 'EUR_USD'
#lookback = 1440 # Lookback 1440 observations [60 min], which is one day.
#step = 60 # Step of 60 observations [60 min]
#delay = 60 # Predict 60 observations [60 min] into future
batch_size = 128
mod_layers = [2,4]
mod_nodes = 16
optimizer = 'adam'
mod_epochs = 5

#train_gen = train_gen_list[2] 
#val_gen = val_gen_list[2]  
#val_steps = val_steps_list[2]
#dataset = datasets[2]
#duration = duration_list[2]
#pair = pair_list[2]
#lookback = lookback_list[2]
#step = step_list[2]
#delay = delay_list[2]


# In[276]:


# Create lists to hold the results of each iteration.
model_id = 1 # initial model ID
model_id_list = []
model_type_list = []
duration_list = []
pair_list = []
lookback_list = []
step_list = []
delay_list = []
n_layer_list = []
node_list = []
hidden_activation_list = []
out_activation_list = []
optimizer_list = []
train_loss_list = []
valid_loss_list = []
max_epoch_list = []
epoch_list = []
batch_size_list = []
processing_time = []


# In[277]:


# Create a loop to iterate through model variations
for mod_layer in mod_layers:
    for train_gen, val_gen, val_steps, dataset, duration, pair, lookback, step, delay in zip(train_gen_list, 
                                                                                             val_gen_list,  
                                                                                             val_steps_list, 
                                                                                             datasets,
                                                                                             duration_data_list,
                                                                                             pair_data_list,
                                                                                             lookback_data_list,
                                                                                             step_data_list,
                                                                                             delay_data_list
                                                                                            ):

        # Print the basic parameters of the model being iterated
        print('---------------------------------------------------------------')    
        print("Model No.: %r" %model_id)
        print("Model Type: %r" %model_type)
        print("Dataset: %r" %dataset)
        print("Duration: %r" %duration)
        print("Lookback: %r" %lookback)
        print("Step: %r" %step)
        print("Delay: %r" %delay)
        print("Number of Layers: %r" %mod_layer)
        print("Number of Nodes: %r" %mod_nodes)
        print("Hidden Layer Optimizer: %r" %optimizer)

        start_time = time.clock() # start timer to evaluate how long it takes to train this base model

        # Set seeds for reproducible results
        seed(1)
        set_random_seed(2)
        rn.seed(3)


        # Define the model structure that is to be evaluated.
        model = Sequential()
        if mod_layer == 1:
            model.add(SimpleRNN(mod_nodes, input_shape=(None, train[:,1:6].shape[-1])))
        if mod_layer == 2:
            model.add(SimpleRNN(mod_nodes, input_shape=(None, train[:,1:6].shape[-1]), return_sequences=True))
            model.add(SimpleRNN(mod_nodes*2, input_shape=(None, train[:,1:6].shape[-1])))
        if mod_layer == 4:
            model.add(SimpleRNN(mod_nodes, input_shape=(None, train[:,1:6].shape[-1]), return_sequences=True))
            model.add(SimpleRNN(mod_nodes*2, input_shape=(None, train[:,1:6].shape[-1]), return_sequences=True))
            model.add(SimpleRNN(mod_nodes*2, input_shape=(None, train[:,1:6].shape[-1]), return_sequences=True))
            model.add(SimpleRNN(mod_nodes*2, input_shape=(None, train[:,1:6].shape[-1])))
        
        model.add(Dense(1))
        #print(model.summary())
        model.compile(optimizer=optimizer,
                      loss='mae')

        history = model.fit_generator(train_gen,
                                      steps_per_epoch=500,
                                      epochs=mod_epochs,
                                      validation_data=val_gen,
                                      validation_steps=val_steps)

        file_name = ('Exp_%s_Mod_%s_Results' %(exp_no, model_id))
        model.save_weights(file_name)

        # Stop the timer and append time to list
        end_time = time.clock() # end timer
        runtime = end_time - start_time  # seconds of wall-clock time 
        print("\nProcessing time (seconds): %f" % runtime)  # print process time to train model        
        processing_time.append(runtime) # append process time results to list for comparison table

        # Identify and print maximum validation loss and epoch where that accuracy was realized
        epoch_max = np.argmin(history.history['val_loss'])
        val_loss_max = min(history.history['val_loss'])
        print('Maximum Validation Loss: %r, Epochs Required to Achieve this Loss: %r' %(val_loss_max, epoch_max))

        # Print validation accuracy
        train_loss_max = min(history.history['loss'])
        print("Maximum Train Loss: %r" %train_loss_max) 

        # Plot the accuracy and loss of each model iteration
        plot_figs(model_id, model_type, mod_nodes)

        # Append lists with values for this model ID
        model_id_list.append(model_id)
        model_type_list.append(model_type)
        duration_list.append(duration)
        pair_list.append(pair)
        lookback_list.append(lookback)
        step_list.append(step)
        delay_list.append(delay)
        n_layer_list.append(mod_layer)
        node_list.append(mod_nodes)
        optimizer_list.append(optimizer)
        train_loss_list.append(train_loss_max)
        valid_loss_list.append(val_loss_max)
        max_epoch_list.append(epoch_max)
        epoch_list.append(mod_epochs)
        batch_size_list.append(batch_size)

        model_id += 1


# In[278]:


# round numbers in lists for readability
processing_time_formatted = [ '%.3f' % elem for elem in processing_time]
train_loss_list_formatted = [ '%.4f' % elem for elem in train_loss_list]
valid_loss_list_formatted = [ '%.4f' % elem for elem in valid_loss_list]


# In[279]:


# Create a dictionary to store the results of the trained models from code above.
data = OrderedDict([('Model ID', model_id_list),
                    ('Model Type', model_type_list),
                    ('Duration', duration_list),
                    ('Pair', pair_list),
                    ('Lookback', lookback_list),
                    ('Step', step_list),
                    ('Delay', delay_list),
                    ('No. of Layers', n_layer_list),
                    ('Nodes', node_list),
                    ('Optimizer', optimizer_list),
                    ('Process Time', processing_time_formatted),
                    ('Train Acc.', train_loss_list_formatted),
                    ('Val Acc.', valid_loss_list_formatted),
                    ('No. of Epochs to Max Acc.',epoch_max),
                    ('Batch Size', batch_size_list)])


# In[280]:


# Pickle the data for use in the future without having to run models again.  Write to binary file
file_name = ('Exp_%s_Results.pickle' %exp_no)
with open(file_name, 'wb') as f:
    # Pickle the 'data' dictionary using the highest protocol available.
    pickle.dump(data, f, pickle.HIGHEST_PROTOCOL)

print('\n Run complete. Data objects sent to binary file %r' %file_name)


# In[281]:


# Extract objects from the dictionary object data
file_name = 'Exp_5_Results.pickle'
pickle_in = open(file_name,"rb")
data = pickle.load(pickle_in)

print('\n Loading of data complete. The binary file %r has been loaded' %file_name)


# In[282]:


## Create a table to display results of this experiment
results = pd.DataFrame(data)
print('\nTable 5: Benchmark Experiment: Multi-Layer RNN Model Comparison\n')
results 


# ### Experiment #6- Evaluate Impacts of Bidirectional RNNs

# In[341]:


# Specify attributes about this model
exp_no = 6 # Experiment Number
model_type = 'RNN'
layer_type = 'RNN'
#duration = 'Hourly'
#pair = 'EUR_USD'
#lookback = 1440 # Lookback 1440 observations [60 min], which is one day.
#step = 60 # Step of 60 observations [60 min]
#delay = 60 # Predict 60 observations [60 min] into future
batch_size = 128
mod_layers = 1
mod_nodes = 16
optimizer = 'adam'
mod_epochs = 5

#train_gen = train_gen_list[2] 
#val_gen = val_gen_list[2]  
#val_steps = val_steps_list[2]
#dataset = datasets[2]
#duration = duration_list[2]
#pair = pair_list[2]
#lookback = lookback_list[2]
#step = step_list[2]
#delay = delay_list[2]


# In[342]:


# Create lists to hold the results of each iteration.
model_id = 1 # initial model ID
model_id_list = []
model_type_list = []
duration_list = []
pair_list = []
lookback_list = []
step_list = []
delay_list = []
n_layer_list = []
node_list = []
hidden_activation_list = []
out_activation_list = []
optimizer_list = []
train_loss_list = []
valid_loss_list = []
max_epoch_list = []
epoch_list = []
batch_size_list = []
processing_time = []


# In[343]:


# Create a loop to iterate through model variations

for train_gen, val_gen, val_steps, dataset, duration, pair, lookback, step, delay in zip(train_gen_list, 
                                                                                             val_gen_list,  
                                                                                             val_steps_list, 
                                                                                             datasets,
                                                                                             duration_data_list,
                                                                                             pair_data_list,
                                                                                             lookback_data_list,
                                                                                             step_data_list,
                                                                                             delay_data_list
                                                                                            ):

    # Print the basic parameters of the model being iterated
    print('---------------------------------------------------------------')    
    print("Model No.: %r" %model_id)
    print("Model Type: %r" %model_type)
    print("Dataset: %r" %dataset)
    print("Duration: %r" %duration)
    print("Lookback: %r" %lookback)
    print("Step: %r" %step)
    print("Delay: %r" %delay)
    print("Number of Layers: %r" %mod_layer)
    print("Number of Nodes: %r" %mod_nodes)
    print("Hidden Layer Optimizer: %r" %optimizer)

    start_time = time.clock() # start timer to evaluate how long it takes to train this base model

    # Set seeds for reproducible results
    seed(1)
    set_random_seed(2)
    rn.seed(3)


    # Define the model structure that is to be evaluated.
    model = Sequential()
    model.add(Bidirectional(SimpleRNN(mod_nodes), input_shape=(None, train[:,1:6].shape[-1])))
    #model.add(SimpleRNN(mod_nodes, input_shape=(None, train[:,1:6].shape[-1])))
    model.add(Dense(1))
    #print(model.summary())
    model.compile(optimizer=optimizer,
                  loss='mae')

    history = model.fit_generator(train_gen,
                                  steps_per_epoch=500,
                                  epochs=mod_epochs,
                                  validation_data=val_gen,
                                  validation_steps=val_steps)

    file_name = ('Exp_%s_Mod_%s_Results' %(exp_no, model_id))
    model.save_weights(file_name)

    # Stop the timer and append time to list
    end_time = time.clock() # end timer
    runtime = end_time - start_time  # seconds of wall-clock time 
    print("\nProcessing time (seconds): %f" % runtime)  # print process time to train model        
    processing_time.append(runtime) # append process time results to list for comparison table

    # Identify and print maximum validation loss and epoch where that accuracy was realized
    epoch_max = np.argmin(history.history['val_loss'])
    val_loss_max = min(history.history['val_loss'])
    print('Maximum Validation Loss: %r, Epochs Required to Achieve this Loss: %r' %(val_loss_max, epoch_max))

    # Print validation accuracy
    train_loss_max = min(history.history['loss'])
    print("Maximum Train Loss: %r" %train_loss_max) 

    # Plot the accuracy and loss of each model iteration
    plot_figs(model_id, model_type, mod_nodes)

    # Append lists with values for this model ID
    model_id_list.append(model_id)
    model_type_list.append(model_type)
    duration_list.append(duration)
    pair_list.append(pair)
    lookback_list.append(lookback)
    step_list.append(step)
    delay_list.append(delay)
    n_layer_list.append(mod_layer)
    node_list.append(mod_nodes)
    optimizer_list.append(optimizer)
    train_loss_list.append(train_loss_max)
    valid_loss_list.append(val_loss_max)
    max_epoch_list.append(epoch_max)
    epoch_list.append(mod_epochs)
    batch_size_list.append(batch_size)

    model_id += 1


# In[ ]:


# round numbers in lists for readability
processing_time_formatted = [ '%.3f' % elem for elem in processing_time]
train_loss_list_formatted = [ '%.4f' % elem for elem in train_loss_list]
valid_loss_list_formatted = [ '%.4f' % elem for elem in valid_loss_list]


# In[ ]:


# Create a dictionary to store the results of the trained models from code above.
data = OrderedDict([('Model ID', model_id_list),
                    ('Model Type', model_type_list),
                    ('Duration', duration_list),
                    ('Pair', pair_list),
                    ('Lookback', lookback_list),
                    ('Step', step_list),
                    ('Delay', delay_list),
                    ('No. of Layers', n_layer_list),
                    ('Nodes', node_list),
                    ('Optimizer', optimizer_list),
                    ('Process Time', processing_time_formatted),
                    ('Train Acc.', train_loss_list_formatted),
                    ('Val Acc.', valid_loss_list_formatted),
                    ('No. of Epochs to Max Acc.',epoch_max),
                    ('Batch Size', batch_size_list)])


# In[ ]:


# Pickle the data for use in the future without having to run models again.  Write to binary file
file_name = ('Exp_%s_Results.pickle' %exp_no)
with open(file_name, 'wb') as f:
    # Pickle the 'data' dictionary using the highest protocol available.
    pickle.dump(data, f, pickle.HIGHEST_PROTOCOL)

print('\n Run complete. Data objects sent to binary file %r' %file_name)


# In[ ]:


# Extract objects from the dictionary object data
file_name = 'Exp_6_Results.pickle'
pickle_in = open(file_name,"rb")
data = pickle.load(pickle_in)

print('\n Loading of data complete. The binary file %r has been loaded' %file_name)


# In[ ]:


## Create a table to display results of this experiment
results = pd.DataFrame(data)
print('\nTable 6: Benchmark Experiment: Bidirectional RNN Model\n')
results 


# ### Experiment #7- Compare RNN to 1-D CNN

# In[302]:


# Specify attributes about this model
exp_no = 7 # Experiment Number
model_type = ['CNN', 'CNN + LSTM']
layer_type = ['1D CNN', '1D CNN + LSTM']
#duration = 'Hourly'
#pair = 'EUR_USD'
#lookback = 1440 # Lookback 1440 observations [60 min], which is one day.
#step = 60 # Step of 60 observations [60 min]
#delay = 60 # Predict 60 observations [60 min] into future
batch_size = 128
mod_layers = [1,2]
mod_nodes = 16
hidden_activation = 'relu'
optimizer = 'adam'
mod_epochs = 5

#train_gen = train_gen_list[2] 
#val_gen = val_gen_list[2]  
#val_steps = val_steps_list[2]
#dataset = datasets[2]
#duration = duration_list[2]
#pair = pair_list[2]
#lookback = lookback_list[2]
#step = step_list[2]
#delay = delay_list[2]


# In[303]:


# Create lists to hold the results of each iteration.
model_id = 1 # initial model ID
model_id_list = []
model_type_list = []
duration_list = []
pair_list = []
lookback_list = []
step_list = []
delay_list = []
n_layer_list = []
node_list = []
hidden_activation_list = []
out_activation_list = []
optimizer_list = []
train_loss_list = []
valid_loss_list = []
max_epoch_list = []
epoch_list = []
batch_size_list = []
processing_time = []


# In[313]:


# Create a loop to iterate through model variations

for mod_type, layer in zip(model_type, layer_type):
    for train_gen, val_gen, val_steps, dataset, duration, pair, lookback, step, delay in zip(train_gen_list, 
                                                                                     val_gen_list,  
                                                                                     val_steps_list, 
                                                                                     datasets,
                                                                                     duration_data_list,
                                                                                     pair_data_list,
                                                                                     lookback_data_list,
                                                                                     step_data_list,
                                                                                     delay_data_list
                                                                                    ):
                                                                                    
        


        # Print the basic parameters of the model being iterated
        print('---------------------------------------------------------------')    
        print("Model No.: %r" %model_id)
        print("Model Type: %r" %mod_type)
        print("Dataset: %r" %dataset)
        print("Duration: %r" %duration)
        print("Lookback: %r" %lookback)
        print("Step: %r" %step)
        print("Delay: %r" %delay)
        print("Number of Layers: %r" %mod_layer)
        print("Number of Nodes: %r" %mod_nodes)
        print("Hidden Layer Optimizer: %r" %optimizer)

        start_time = time.clock() # start timer to evaluate how long it takes to train this base model

        # Set seeds for reproducible results
        seed(1)
        set_random_seed(2)
        rn.seed(3)
        
        #model.add(SimpleRNN(mod_nodes, input_shape=(None, train[:,1:6].shape[-1])))
        # Define the model structure that is to be evaluated.
        model = Sequential()
        model.add(Conv1D(mod_nodes, 7, activation=hidden_activation, input_shape=(None, train[:,1:6].shape[-1])))
        model.add(MaxPooling1D(5))
        model.add(Conv1D(mod_nodes, 3, activation=hidden_activation))
        model.add(GlobalMaxPooling1D())
        if mod_type == 'CNN + LSTM':
            model.add(BasicRNN(mod_nodes))
        
        model.add(Dense(1))
        #print(model.summary())
        model.compile(optimizer=optimizer,
                      loss='mae')

        history = model.fit_generator(train_gen,
                                      steps_per_epoch=500,
                                      epochs=mod_epochs,
                                      validation_data=val_gen,
                                      validation_steps=val_steps)

        file_name = ('Exp_%s_Mod_%s_Results' %(exp_no, model_id))
        model.save_weights(file_name)

        # Stop the timer and append time to list
        end_time = time.clock() # end timer
        runtime = end_time - start_time  # seconds of wall-clock time 
        print("\nProcessing time (seconds): %f" % runtime)  # print process time to train model        
        processing_time.append(runtime) # append process time results to list for comparison table

        # Identify and print maximum validation loss and epoch where that accuracy was realized
        epoch_max = np.argmin(history.history['val_loss'])
        val_loss_max = min(history.history['val_loss'])
        print('Maximum Validation Loss: %r, Epochs Required to Achieve this Loss: %r' %(val_loss_max, epoch_max))

        # Print validation accuracy
        train_loss_max = min(history.history['loss'])
        print("Maximum Train Loss: %r" %train_loss_max) 

        # Plot the accuracy and loss of each model iteration
        plot_figs(model_id, model_type, mod_nodes)

        # Append lists with values for this model ID
        model_id_list.append(model_id)
        model_type_list.append(mod_type)
        duration_list.append(duration)
        pair_list.append(pair)
        lookback_list.append(lookback)
        step_list.append(step)
        delay_list.append(delay)
        n_layer_list.append(mod_layer)
        node_list.append(mod_nodes)
        optimizer_list.append(optimizer)
        train_loss_list.append(train_loss_max)
        valid_loss_list.append(val_loss_max)
        max_epoch_list.append(epoch_max)
        epoch_list.append(mod_epochs)
        batch_size_list.append(batch_size)

        model_id += 1


# In[ ]:


# round numbers in lists for readability
processing_time_formatted = [ '%.3f' % elem for elem in processing_time]
train_loss_list_formatted = [ '%.4f' % elem for elem in train_loss_list]
valid_loss_list_formatted = [ '%.4f' % elem for elem in valid_loss_list]


# In[ ]:


# Create a dictionary to store the results of the trained models from code above.
data = OrderedDict([('Model ID', model_id_list),
                    ('Model Type', model_type_list),
                    ('Duration', duration_list),
                    ('Pair', pair_list),
                    ('Lookback', lookback_list),
                    ('Step', step_list),
                    ('Delay', delay_list),
                    ('No. of Layers', n_layer_list),
                    ('Nodes', node_list),
                    ('Optimizer', optimizer_list),
                    ('Process Time', processing_time_formatted),
                    ('Train Acc.', train_loss_list_formatted),
                    ('Val Acc.', valid_loss_list_formatted),
                    ('No. of Epochs to Max Acc.',epoch_max),
                    ('Batch Size', batch_size_list)])


# In[ ]:


# Pickle the data for use in the future without having to run models again.  Write to binary file
file_name = ('Exp_%s_Results.pickle' %exp_no)
with open(file_name, 'wb') as f:
    # Pickle the 'data' dictionary using the highest protocol available.
    pickle.dump(data, f, pickle.HIGHEST_PROTOCOL)

print('\n Run complete. Data objects sent to binary file %r' %file_name)


# In[ ]:


# Extract objects from the dictionary object data
file_name = 'Exp_7_Results.pickle'
pickle_in = open(file_name,"rb")
data = pickle.load(pickle_in)

print('\n Loading of data complete. The binary file %r has been loaded' %file_name)


# In[206]:


## Create a table to display results of this experiment
results = pd.DataFrame(data)
print('\nTable 7: Benchmark Experiment: Comparison of 1D CNN and 1D CNN + LSTM Models\n')
results 


# In[ ]:





# #### 5.1 EDA Results
# ***

# ### 6.0 Conclusions
# ***

# In[405]:


# Develop list of results from each experiment to tabulate overall findings
exp_list= ['base','base',1,2,2,3,3,4,4,5,5]
model_type_list = ['Naive','Naive','Dense NN','RNN','RNN','GRU','GRU','RNN','RNN','RNN','RNN']
lookback_list = [60,1440,1140,1140,1140,1440,1440,1440,1440,1440,10080]
step_list = [60]*11
delay_list = [60,1440,60,60,1440,60,1440,60,1440,60,1440]
n_layer_list = ['N/A','N/A',1,1,1,1,1,1,1,2,4]
node_list = ['N/A','N/A',32,16,16,16,16,16,16,16,16]
hidden_activation_list = ['N/A','N/A','sigmoid','N/A','N/A','N/A','N/A','N/A','N/A','N/A','N/A']
out_activation_list = ['N/A','N/A','softmax','N/A','N/A','N/A','N/A','N/A','N/A','N/A','N/A']
dropout_list = ['N/A','N/A','N/A','N/A','N/A','N/A','N/A','0.5/0.1','0.1/0.1','N/A','N/A']
optimizer_list = ['N/A','N/A','rmsprop','adam','rmsprop','adam','adam','adam','adam','adam','adam']
processing_time_formatted = [297.375,294.451,738.842,906.634,940.195,1575.713,1614.887,997.491,851.622,1683.399,9467.630]
train_loss_list_formatted = ['N/A','N/A',0.0224,0.0025,0.0176,0.0023,0.0049,0.0315,0.0316,0.0087,0.0067]
valid_loss_list_formatted = [0.0008,0.0028,0.002,0.0009,0.0026,0.0011,0.0026,0.0245,0.0261,0.0016,0.0025]
epoch_list = [5]*11
batch_size_list = [128]*11


# In[406]:


# Create a dictionary to store the results of the trained models from code above.
data = OrderedDict([('Model Type', model_type_list),
                    ('Lookback', lookback_list),
                    ('Step', step_list),
                    ('Delay', delay_list),
                    ('No. of Layers', n_layer_list),
                    ('Nodes', node_list),
                    ('Hidden Activ.', hidden_activation_list),
                    ('Output Activ.', out_activation_list),
                    ('Dropout / Recurrent Dropout Rate', dropout_list),
                    ('Optimizer', optimizer_list),
                    ('Process Time', processing_time_formatted),
                    ('Train Acc.', train_loss_list_formatted),
                    ('Val Acc.', valid_loss_list_formatted),
                    ('No. of Epochs to Max Acc.',epoch_list),
                    ('Batch Size', batch_size_list)])


# In[407]:


# Pickle the data for use in the future without having to run models again.  Write to binary file
file_name = ('Exp_%s_Results.pickle' %'Summary')
with open(file_name, 'wb') as f:
    # Pickle the 'data' dictionary using the highest protocol available.
    pickle.dump(data, f, pickle.HIGHEST_PROTOCOL)

print('\n Run complete. Data objects sent to binary file %r' %file_name)


# In[408]:


# Extract objects from the dictionary object data
file_name = 'Exp_Summary_Results.pickle'
pickle_in = open(file_name,"rb")
data = pickle.load(pickle_in)

print('\n Loading of data complete. The binary file %r has been loaded' %file_name)


# In[409]:


## Create a table to display results of this experiment
results = pd.DataFrame(data)
print('\nTable 8: Benchmark Experiment Summary: Comparison of Model Architectures and Hyperparameters Tested\n')
results 


# #### Use Best 1-Hr Delay Model with Test Data

# In[448]:


# Run Naive model on test data
for step, gen, dataset_name in zip(test_steps_list, test_gen_list, datasets):
        
    val_steps = step
    val_gen = gen
    dataset = dataset_name
    
    print('\nDataset: ', dataset)

    start_time = time.clock() # start timer to evaluate how long it takes to train this base model

    evaluate_naive_method()

    end_time = time.clock() # end timer
    runtime = end_time - start_time  # seconds of wall-clock time 
    print("Processing time (seconds): %f" % runtime)  # print process time to train model


# In[449]:


# Specify attributes about this model
exp_no = 'Best 1_Hr' # Experiment Number
model_type = 'RNN'
#duration = 'Hourly'
#pair = 'EUR_USD'
#lookback = 1440 # Lookback 1440 observations [60 min], which is one day.
#step = 60 # Step of 60 observations [60 min]
#delay = 60 # Predict 60 observations [60 min] into future
batch_size = 128
mod_layers = 1
mod_nodes = 16
hidden_activation = 'N/A'
out_activation = 'N/A'
optimizer = 'adam'
mod_epochs = 5

# Just use EUR_USD_min_hour_day_hour dataset
train_gen = train_gen_list[0] 
test_gen = test_gen_list[0]  
test_steps = test_steps_list[0]
dataset = datasets[0]
duration = duration_data_list[0]
pair = pair_data_list[0]
lookback = lookback_data_list[0]
step = step_data_list[0]
delay = delay_data_list[0]


# In[450]:


# Create lists to hold the results of each iteration.
model_id = 1 # initial model ID
model_id_list = []
model_type_list = []
#duration_list = []
#pair_list = []
#lookback_list = []
#step_list = []
#delay_list = []
n_layer_list = []
node_list = []
hidden_activation_list = []
out_activation_list = []
optimizer_list = []
train_loss_list = []
valid_loss_list = []
max_epoch_list = []
epoch_list = []
batch_size_list = []
processing_time = []


# In[451]:


# Create a loop to iterate through model variations


# Print the basic parameters of the model being iterated
print('---------------------------------------------------------------')    
print("Model No.: %r" %model_id)
print("Model Type: %r" %model_type)
print("Dataset: %r" %dataset)
print("Duration: %r" %duration)
print("Lookback: %r" %lookback)
print("Step: %r" %step)
print("Delay: %r" %delay)
print("Number of Layers: %r" %mod_layers)
print("Number of Nodes: %r" %mod_nodes)
print("Hidden Layer Optimizer: %r" %opt)

start_time = time.clock() # start timer to evaluate how long it takes to train this base model

# Set seeds for reproducible results
seed(1)
set_random_seed(2)
rn.seed(3)


# Define the model structure that is to be evaluated.
model = Sequential()
model.add(SimpleRNN(mod_nodes, input_shape=(None, train[:,1:6].shape[-1])))
model.add(Dense(1))
#print(model.summary())
model.compile(optimizer=optimizer,
              loss='mae')

history = model.fit_generator(train_gen,
                              steps_per_epoch=500,
                              epochs=mod_epochs,
                              validation_data=test_gen,
                              validation_steps=test_steps)

file_name = ('Exp_%s_Mod_%s_Results' %(exp_no, model_id))
model.save_weights(file_name)

# Stop the timer and append time to list
end_time = time.clock() # end timer
runtime = end_time - start_time  # seconds of wall-clock time 
print("\nProcessing time (seconds): %f" % runtime)  # print process time to train model        
processing_time.append(runtime) # append process time results to list for comparison table

# Identify and print maximum validation loss and epoch where that accuracy was realized
epoch_max = np.argmin(history.history['val_loss'])
val_loss_max = min(history.history['val_loss'])
print('Maximum Validation Loss: %r, Epochs Required to Achieve this Loss: %r' %(val_loss_max, epoch_max))

# Print validation accuracy
train_loss_max = min(history.history['loss'])
print("Maximum Train Loss: %r" %train_loss_max) 

# Plot the accuracy and loss of each model iteration
plot_figs(model_id, model_type, mod_nodes)

# Append lists with values for this model ID
model_id_list.append(model_id)
model_type_list.append(model_type)
duration_list.append(duration)
pair_list.append(pair)
lookback_list.append(lookback)
step_list.append(step)
delay_list.append(delay)
n_layer_list.append(mod_layers)
node_list.append(mod_nodes)
hidden_activation_list.append(hidden_activation)
out_activation_list.append(out_activation)
optimizer_list.append(optimizer)
train_loss_list.append(train_loss_max)
valid_loss_list.append(val_loss_max)
max_epoch_list.append(epoch_max)
epoch_list.append(mod_epochs)
batch_size_list.append(batch_size)

model_id += 1


# In[452]:


# round numbers in lists for readability
processing_time_formatted = [ '%.3f' % elem for elem in processing_time]
train_loss_list_formatted = [ '%.4f' % elem for elem in train_loss_list]
valid_loss_list_formatted = [ '%.4f' % elem for elem in valid_loss_list]


# In[479]:


# Create a dictionary to store the results of the trained models from code above.
data = OrderedDict([('Model ID', model_id_list),
                    ('Model Type', model_type_list),
                    ('Duration', duration_list),
                    ('Pair', pair_list),
                    ('Lookback', lookback_list),
                    ('Step', step_list),
                    ('Delay', delay_list),
                    ('No. of Layers', n_layer_list),
                    ('Nodes', node_list),
                    ('Hidden Layer Act.', hidden_activation_list),
                    ('Output Layer Activation', out_activation_list),
                    ('Optimizer', optimizer_list),
                    ('Process Time', processing_time_formatted),
                    ('Train Acc.', train_loss_list_formatted),
                    ('Test Acc.', valid_loss_list_formatted),
                    ('No. of Epochs to Max Acc.',epoch_max),
                    ('Batch Size', batch_size_list)])


# In[478]:


len(n_layer_list)


# In[477]:


delay_list


# In[476]:


delay_list = delay_list[11]


# In[480]:


# Pickle the data for use in the future without having to run models again.  Write to binary file
file_name = ('Exp_%s_Results.pickle' %exp_no)
with open(file_name, 'wb') as f:
    # Pickle the 'data' dictionary using the highest protocol available.
    pickle.dump(data, f, pickle.HIGHEST_PROTOCOL)

print('\n Run complete. Data objects sent to binary file %r' %file_name)


# In[481]:


# Extract objects from the dictionary object data
file_name = 'Exp_Best 1_Hr_Results.pickle'
pickle_in = open(file_name,"rb")
data = pickle.load(pickle_in)

print('\n Loading of data complete. The binary file %r has been loaded' %file_name)


# In[482]:


## Create a table to display results of this experiment
results = pd.DataFrame(data)
print('\nTable 9: Optimal 1-Hr Model Predictions on Test Data\n')
results 


# #### Use Best 1-Day Delay Model with Test Data and Compare to Baseline

# In[483]:


# Specify attributes about this model
exp_no = 'Best 1_Day' # Experiment Number
model_type = 'RNN'
#duration = 'Hourly'
#pair = 'EUR_USD'
#lookback = 1440 # Lookback 1440 observations [60 min], which is one day.
#step = 60 # Step of 60 observations [60 min]
#delay = 60 # Predict 60 observations [60 min] into future
batch_size = 128
mod_layers = 1
mod_nodes = 16
hidden_activation = 'N/A'
out_activation = 'N/A'
optimizer = 'adagrad'
mod_epochs = 5

# Just use EUR_USD_min_day_day_hour dataset
train_gen = train_gen_list[1] 
test_gen = test_gen_list[1]  
test_steps = test_steps_list[1]
dataset = datasets[1]
duration = duration_data_list[1]
pair = pair_data_list[1]
lookback = lookback_data_list[1]
step = step_data_list[1]
delay = delay_data_list[1]


# In[484]:


# Create lists to hold the results of each iteration.
model_id = 1 # initial model ID
model_id_list = []
model_type_list = []
#duration_list = []
#pair_list = []
#lookback_list = []
#step_list = []
#delay_list = []
n_layer_list = []
node_list = []
hidden_activation_list = []
out_activation_list = []
optimizer_list = []
train_loss_list = []
valid_loss_list = []
max_epoch_list = []
epoch_list = []
batch_size_list = []
processing_time = []


# In[485]:


# Create a loop to iterate through model variations


# Print the basic parameters of the model being iterated
print('---------------------------------------------------------------')    
print("Model No.: %r" %model_id)
print("Model Type: %r" %model_type)
print("Dataset: %r" %dataset)
print("Duration: %r" %duration)
print("Lookback: %r" %lookback)
print("Step: %r" %step)
print("Delay: %r" %delay)
print("Number of Layers: %r" %mod_layers)
print("Number of Nodes: %r" %mod_nodes)
print("Hidden Layer Optimizer: %r" %opt)

start_time = time.clock() # start timer to evaluate how long it takes to train this base model

# Set seeds for reproducible results
seed(1)
set_random_seed(2)
rn.seed(3)


# Define the model structure that is to be evaluated.
model = Sequential()
model.add(SimpleRNN(mod_nodes, input_shape=(None, train[:,1:6].shape[-1])))
model.add(Dense(1))
#print(model.summary())
model.compile(optimizer=optimizer,
              loss='mae')

history = model.fit_generator(train_gen,
                              steps_per_epoch=500,
                              epochs=mod_epochs,
                              validation_data=test_gen,
                              validation_steps=test_steps)

file_name = ('Exp_%s_Mod_%s_Results' %(exp_no, model_id))
model.save_weights(file_name)

# Stop the timer and append time to list
end_time = time.clock() # end timer
runtime = end_time - start_time  # seconds of wall-clock time 
print("\nProcessing time (seconds): %f" % runtime)  # print process time to train model        
processing_time.append(runtime) # append process time results to list for comparison table

# Identify and print maximum validation loss and epoch where that accuracy was realized
epoch_max = np.argmin(history.history['val_loss'])
val_loss_max = min(history.history['val_loss'])
print('Maximum Validation Loss: %r, Epochs Required to Achieve this Loss: %r' %(val_loss_max, epoch_max))

# Print validation accuracy
train_loss_max = min(history.history['loss'])
print("Maximum Train Loss: %r" %train_loss_max) 

# Plot the accuracy and loss of each model iteration
plot_figs(model_id, model_type, mod_nodes)

# Append lists with values for this model ID
model_id_list.append(model_id)
model_type_list.append(model_type)
duration_list.append(duration)
pair_list.append(pair)
lookback_list.append(lookback)
step_list.append(step)
delay_list.append(delay)
n_layer_list.append(mod_layers)
node_list.append(mod_nodes)
hidden_activation_list.append(hidden_activation)
out_activation_list.append(out_activation)
optimizer_list.append(optimizer)
train_loss_list.append(train_loss_max)
valid_loss_list.append(val_loss_max)
max_epoch_list.append(epoch_max)
epoch_list.append(mod_epochs)
batch_size_list.append(batch_size)

model_id += 1


# In[486]:


# round numbers in lists for readability
processing_time_formatted = [ '%.3f' % elem for elem in processing_time]
train_loss_list_formatted = [ '%.4f' % elem for elem in train_loss_list]
valid_loss_list_formatted = [ '%.4f' % elem for elem in valid_loss_list]


# In[558]:


# Create a dictionary to store the results of the trained models from code above.
data = OrderedDict([('Model ID', model_id_list),
                    ('Model Type', model_type_list),
                    ('Duration', duration_list),
                    ('Pair', pair_list),
                    ('Lookback', lookback_list),
                    ('Step', step_list),
                    ('Delay', delay_list),
                    ('No. of Layers', n_layer_list),
                    ('Nodes', node_list),
                    ('Hidden Layer Act.', hidden_activation_list),
                    ('Output Layer Activation', out_activation_list),
                    ('Optimizer', optimizer_list),
                    ('Process Time', processing_time_formatted),
                    ('Train Acc.', train_loss_list_formatted),
                    ('Test Acc.', valid_loss_list_formatted),
                    ('No. of Epochs to Max Acc.',epoch_max),
                    ('Batch Size', batch_size_list)])


# In[557]:


step_list


# In[554]:


out_activation_list = ['N/A']


# In[559]:


# Pickle the data for use in the future without having to run models again.  Write to binary file
file_name = ('Exp_%s_Results.pickle' %exp_no)
with open(file_name, 'wb') as f:
    # Pickle the 'data' dictionary using the highest protocol available.
    pickle.dump(data, f, pickle.HIGHEST_PROTOCOL)

print('\n Run complete. Data objects sent to binary file %r' %file_name)


# In[560]:


# Extract objects from the dictionary object data
file_name = 'Exp_Best 1_Day_Results.pickle'
pickle_in = open(file_name,"rb")
data = pickle.load(pickle_in)

print('\n Loading of data complete. The binary file %r has been loaded' %file_name)


# In[561]:


## Create a table to display results of this experiment
results = pd.DataFrame(data)
print('\nTable 10: Optimal 1-Day Model Predictions on Test Data\n')
results 


# In[ ]:





# In[ ]:





# In[ ]:





# # Sandbox
# The following code was not used as part of this assignment, but I wanted to keep it for future reference.

# In[ ]:


##########  TO BE DELETED AFTER CONFIRM NO LONGER USEFUL  ######################

# Create a function to generate batches of data that includes past exchange prices (training data) as well 
# as future prices (validation data).

'''
lookback - How many timesteps back the input data should go
delay - How many timesteps in the future the target should be
min_index and max_index - Inidices in the data array that delimit which time steps to draw from.  This is useful 
    for keeping a segment of the data for validation and another for testing.
shuffle - Whether to shuffle the samples or draw them in chronological order.  Always set as False in this experiment.
batch_size - The number of samples per batch
step - The period, in timesteps, at which you sample data.  
'''

def generator(data, lookback, delay, min_index, max_index, 
              shuffle, batch_size, step):
    if max_index is None:
        max_index = len(data) - delay - 1
    i = min_index + lookback
    while 1:
        if shuffle:
            rows = np.random.randint(min_index + lookback, max_index, size=batch_size)
        else:
            if i + batch_size >= max_index:
                i = min_index + lookback
            rows = np.arange(i, min(i + batch_size, max_index))
            i += len(rows)
        
        samples = np.zeros((len(rows),
                           lookback // step,
                           data.shape[-1]))
        
        targets = np.zeros((len(rows),))
        for j, row in enumerate(rows):
            indices = range(rows[j] - lookback, rows[j], step)
            samples[j] = data[indices]
            targets[j] = data[rows[j] + delay][1]
        yield samples, targets


# In[199]:


# Why doesn't this work?

'''
def create_generators(duration, pair, lookback, step, delay, batch_size, train, val, test):
    
    def generator(data, lookback, delay, shuffle, batch_size, step):
        #if max_index is None:
        max_index = len(data) - delay - 1
        i = lookback
        while 1:
            if shuffle:
                rows = np.random.randint(lookback, max_index, size=batch_size)
            else:
                if i + batch_size >= max_index:
                    i = lookback
                rows = np.arange(i, min(i + batch_size, max_index))
                i += len(rows)

            samples = np.zeros((len(rows),
                               lookback // step,
                               data.shape[-1]))

            targets = np.zeros((len(rows),))
            for j, row in enumerate(rows):
                indices = range(rows[j] - lookback, rows[j], step)
                samples[j] = data[indices]
                targets[j] = data[rows[j] + delay][3] # [3] defines the Bid Close price
            yield samples, targets
    
    train_gen = generator(train[:,1:6],
                          lookback = lookback,
                          delay = delay,
                          shuffle = True,
                          step = step,
                          batch_size = batch_size)

    val_gen = generator(val[:,1:6],
                        lookback = lookback,
                        delay = delay,
                        shuffle = False,
                        step = step,
                        batch_size = batch_size)

    test_gen = generator(test[:,1:6],
                         lookback = lookback,
                         delay = delay,
                         shuffle = False,
                         step = step,
                         batch_size = batch_size)

    val_steps = len(val) - lookback
    print('Validation steps:', val_steps)

    test_steps = len(test) - lookback
    print('\nTest steps:', test_steps)

    print('\n*** The following Generators have been created ***')
    print('Duration:', duration)
    print('Pair:', pair)
    print('Generators: train, val, test')
    
    return train_gen, val_gen, test_gen, val_steps, test_steps
    # Chollet Chap 6, pg 211
    
    
'''

# New Cell

'''
duration = 'Minute'
pair = 'EUR_USD'
lookback = 1440
step = 60
delay = 60
batch_size = 128

train_gen_EUR_USD_min, 
val_gen_EUR_USD_min, 
test_gen_EUR_USD_min, 
val_steps_EUR_USD_min, 
test_steps_EUR_USD_min = create_generators(duration, 
                                            pair, 
                                            lookback, 
                                            step, 
                                            delay, 
                                            batch_size, 
                                            train_EUR_USD_min, 
                                            val_EUR_USD_min, 
                                            test_EUR_USD_min)
'''

# New Cell - Get Nan's here

'''
print('Generated Sample Array Shape:', next(train_gen_EUR_USD_min)[0].shape)
print('Generated Target Array Shape', next(train_gen_EUR_USD_min)[1].shape)
print('\nGenerated Sample and Target Array Exampled\n\n', next(train_gen_EUR_USD_min))
'''


# #### Attempt to Plot the Validation Prediction vs. Target 

# In[215]:


exp_no = 2 # Experiment Number
model_type = 'RNN'
#duration = 'Hourly'
#pair = 'EUR_USD'
#lookback = 1440 # Lookback 1440 observations [60 min], which is one day.
#step = 60 # Step of 60 observations [60 min]
#delay = 60 # Predict 60 observations [60 min] into future
batch_size = 128
mod_layers = 1
mod_nodes = 16
hidden_activation = 'N/A'
out_activation = 'N/A'
optimizer = 'adam'
mod_epochs = 5

train_gen = train_gen_list[0]
val_gen = val_gen_list[0]
val_steps = val_steps_list[0]


# In[216]:


model = Sequential()
model.add(SimpleRNN(mod_nodes, input_shape=(None, train[:,1:6].shape[-1])))
model.add(Dense(1))
#print(model.summary())
model.compile(optimizer=optimizer,
              loss='mae')

history = model.fit_generator(train_gen,
                              steps_per_epoch=500,
                              epochs=mod_epochs,
                              validation_data=val_gen,
                              validation_steps=val_steps)


# In[217]:


predict = model.predict_generator(val_gen_list[0], val_steps_list[0])


# In[218]:


val_target_list=[]
for i in range(0,val_steps_list[0]):
    val_data = next(val_gen_list[0])
    if len(val_target_list) == 0:
        #print('less 0')
        val_target_list = val_data[1]
        
    else:
        #print('more 0')
        #print(len(val_target_list))
        #print(len(val_data[1]))
        #print(val_target_list)
        #print(val_data[1])
        val_target_list = np.concatenate([val_target_list,val_data[1]])
        #print(val_target_list)
        
    #print(len(val_target_list))


# In[219]:


plot_data('Hourly', 'EUR_USD', list(range(0,10000)), val_target_list[:10000], list(range(0,10000)), predict[:10000].flatten())


# In[ ]:





# In[ ]:





# In[ ]:




