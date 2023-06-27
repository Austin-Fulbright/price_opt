

from tensorflow.python.keras.wrappers.scikit_learn import KerasRegressor


from sklearn.model_selection import GridSearchCV
import pandas as pd
import numpy as np
import nltk
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Embedding, LSTM, Dense, Dropout
from keras.regularizers import l2
from keras.initializers import GlorotUniform
from keras.callbacks import EarlyStopping
from keras.optimizers import Adam
from keras.activations import sigmoid, tanh, relu, linear

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize



from keras import backend as K
from sklearn.metrics import mean_squared_log_error, mean_squared_error

def msle(y_true, y_pred):
    return K.mean(K.square(K.log(y_true + 1) - K.log(y_pred + 1)), axis=-1)
def rmse(y_true, y_pred):
    return K.sqrt(K.mean(K.square(y_true - y_pred), axis=-1))

def remove_outliers(df, column, multiplier=1.5):
    """ Remove outliers from a dataframe using IQR
    Args:
    df : DataFrame, the dataframe
    column : str, the column name
    multiplier : float, the multiplier for IQR. Default is 1.5.
    Returns:
    DataFrame, the dataframe after removing the outliers
    """
    # Calculate IQR of the column
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1

    # Define the upper and lower bounds for outliers
    lower_bound = Q1 - multiplier * IQR
    upper_bound = Q3 + multiplier * IQR

    # Filter the data frame
    df_out = df.loc[(df[column] >= lower_bound) & (df[column] <= upper_bound)]

    return df_out

# Remove outliers from 'Selling Price($)'



# Load the dataset into a pandas DataFrame
data = pd.read_csv('data/marketing_sample_for_amazon_com-ecommerce__20200101_20200131__10k_data.csv')
data = data.dropna(subset=['About Product', 'Selling Price'])
data.rename(columns={'Uniq Id':'Id', 'Shipping Weight':'Shipping Weight(Pounds)', 'Selling Price':'Selling Price($)'}, inplace=True)

# Data preprocessing
data['Selling Price($)'] = data['Selling Price($)'].str.replace('$', '').str.replace(' ', '').str.split('.').str[0] + '.'
data = data[~data['Selling Price($)'].str.contains('[a-zA-Z]', na=False)]
data['Selling Price($)'] = data['Selling Price($)'].str.replace(',', '').astype(float)
data['Selling Price($)'] = data['Selling Price($)'].apply(lambda x: "{:.2f}".format(x)).astype(float)
data = remove_outliers(data, 'Selling Price($)', multiplier=1.5)
# Normalize the target variable
scaler = MinMaxScaler()
data['Selling Price($)'] = scaler.fit_transform(data['Selling Price($)'].values.reshape(-1, 1))

# Preprocessing function
def preprocess(description):
    # Tokenization
    tokens = word_tokenize(description.lower())
    
    # Remove punctuation and stopwords
    stop_words = set(stopwords.words('english'))
    filtered_tokens = [token for token in tokens if token.isalpha() and token not in stop_words]
    
    return ' '.join(filtered_tokens) # Join the tokens back into a single string

# Apply preprocessing to the text data
data['preprocessed_description'] = data['About Product'].apply(preprocess)

# Split the dataset into training and testing sets
features = data['preprocessed_description']
target = data['Selling Price($)']
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)

# Tokenize the text data
tokenizer = Tokenizer()
tokenizer.fit_on_texts(X_train)
X_train_seq = tokenizer.texts_to_sequences(X_train)
X_test_seq = tokenizer.texts_to_sequences(X_test)

# Pad sequences to a fixed length
max_sequence_length = 100
X_train_padded = pad_sequences(X_train_seq, maxlen=max_sequence_length)
X_test_padded = pad_sequences(X_test_seq, maxlen=max_sequence_length)

# Define the model
embedding_dim = 100
lstm_units = 128

model = Sequential()
model.add(Embedding(len(tokenizer.word_index) + 1, embedding_dim, input_length=max_sequence_length))
model.add(LSTM(lstm_units, return_sequences=True, kernel_initializer=GlorotUniform(seed=42), recurrent_regularizer=l2(1e-5)))
model.add(LSTM(lstm_units, kernel_initializer=GlorotUniform(seed=42), recurrent_regularizer=l2(1e-5)))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(1, activation='linear'))

# Compile the model
learning_rate = 0.001
batch_size = 32
epochs = 100

optimizer = Adam(learning_rate=learning_rate)
model.compile(loss='mean_squared_error', optimizer='adam', metrics=['mape', msle, rmse])

# Define early stopping
early_stopping = EarlyStopping(monitor='val_loss', patience=5)

# Function to create model, required for KerasClassifier
def create_model(optimizer=Adam, learning_rate=0.01, lstm_units=128, dropout_rate=0.0):
    model = Sequential()
    model.add(Embedding(len(tokenizer.word_index) + 1, embedding_dim, input_length=max_sequence_length))
    model.add(LSTM(lstm_units, return_sequences=True, kernel_initializer=GlorotUniform(seed=42), recurrent_regularizer=l2(1e-5)))
    model.add(LSTM(lstm_units, kernel_initializer=GlorotUniform(seed=42), recurrent_regularizer=l2(1e-5)))
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(dropout_rate))
    model.add(Dense(1, activation='linear'))

    # Compile the model
    model.compile(loss='mean_squared_error', optimizer=optimizer(learning_rate=learning_rate), metrics=['mape', msle, rmse])
    
    return model

# Create a KerasRegressor instance
model = KerasRegressor(build_fn=create_model)

# Define parameter grid
param_grid = {
    'learning_rate': [0.001, 0.01, 0.1],
    'optimizer': [Adam],
    'lstm_units': [64, 128, 256],
    'dropout_rate': [0.0, 0.2, 0.5],
    'batch_size': [32, 64, 128],
    'epochs': [10, 50, 100],
}

# Create Grid Search
grid = GridSearchCV(estimator=model, param_grid=param_grid, cv=3)
grid_result = grid.fit(X_train_padded, y_train)

# Report Results
print(f"Best: {grid_result.best_score_} using {grid_result.best_params_}")
