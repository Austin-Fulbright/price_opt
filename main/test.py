import pandas as pd
import numpy as np
import nltk
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential, clone_model
from keras.layers import Embedding, LSTM, Dense, Dropout
from keras.regularizers import l2
from keras.initializers import GlorotUniform
from keras.callbacks import EarlyStopping
from keras.optimizers import Adam
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from keras import backend as K
from sklearn.metrics import mean_squared_log_error, mean_squared_error
from kerastuner.tuners import RandomSearch

def msle(y_true, y_pred):
    return K.mean(K.square(K.log(y_true + 1) - K.log(y_pred + 1)), axis=-1)

def rmse(y_true, y_pred):
    return K.sqrt(K.mean(K.square(y_true - y_pred), axis=-1))

def remove_outliers(df, column, multiplier=1.5):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - multiplier * IQR
    upper_bound = Q3 + multiplier * IQR
    df_out = df.loc[(df[column] >= lower_bound) & (df[column] <= upper_bound)]
    return df_out

data = pd.read_csv('data/marketing_sample_for_amazon_com-ecommerce__20200101_20200131__10k_data.csv')
data = data.dropna(subset=['About Product', 'Selling Price'])
data.rename(columns={'Uniq Id': 'Id', 'Shipping Weight': 'Shipping Weight(Pounds)', 'Selling Price': 'Selling Price($)'}, inplace=True)

data['Selling Price($)'] = data['Selling Price($)'].str.replace('$', '').str.replace(' ', '').str.split('.').str[0] + '.'
data = data[~data['Selling Price($)'].str.contains('[a-zA-Z]', na=False)]
data['Selling Price($)'] = data['Selling Price($)'].str.replace(',', '').astype(float)
data['Selling Price($)'] = data['Selling Price($)'].apply(lambda x: "{:.2f}".format(x)).astype(float)
data = remove_outliers(data, 'Selling Price($)', multiplier=1.5)

scaler = MinMaxScaler()
data['Selling Price($)'] = scaler.fit_transform(data['Selling Price($)'].values.reshape(-1, 1))

def preprocess(description):
    tokens = word_tokenize(description.lower())
    stop_words = set(stopwords.words('english'))
    filtered_tokens = [token for token in tokens if token.isalpha() and token not in stop_words]
    return ' '.join(filtered_tokens)

data['preprocessed_description'] = data['About Product'].apply(preprocess)

features = data['preprocessed_description']
target = data['Selling Price($)']
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)

tokenizer = Tokenizer()
tokenizer.fit_on_texts(X_train)
X_train_seq = tokenizer.texts_to_sequences(X_train)
X_test_seq = tokenizer.texts_to_sequences(X_test)

max_sequence_length = 100
X_train_padded = pad_sequences(X_train_seq, maxlen=max_sequence_length)
X_test_padded = pad_sequences(X_test_seq, maxlen=max_sequence_length)
# Define the base model
def build_model(hp):
    model = Sequential()
    model.add(Embedding(len(tokenizer.word_index) + 1, hp.Int('embedding_dim', min_value=50, max_value=200, step=50), input_length=max_sequence_length))
    model.add(LSTM(hp.Int('lstm_units', min_value=64, max_value=256, step=64), return_sequences=True, kernel_initializer=GlorotUniform(seed=42), recurrent_regularizer=l2(1e-5)))
    model.add(LSTM(hp.Int('lstm_units', min_value=64, max_value=256, step=64), kernel_initializer=GlorotUniform(seed=42), recurrent_regularizer=l2(1e-5)))
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(hp.Float('dropout_rate', min_value=0.3, max_value=0.7, step=0.1)))
    model.add(Dense(1, activation='linear'))
    
    optimizer = Adam(learning_rate=hp.Float('learning_rate', min_value=0.001, max_value=0.1, step=0.001))
    model.compile(loss='mean_squared_error', optimizer=optimizer, metrics=['mape', msle, rmse])
    
    return model

# Perform hyperparameter tuning
tuner = RandomSearch(
    build_model,
    objective='val_loss',
    max_trials=5,
    executions_per_trial=3,
    directory='my_dir',
    project_name='helloworld')

tuner.search(X_train_padded, y_train,
             epochs=5,
             validation_data=(X_test_padded, y_test))

best_model = tuner.get_best_models(num_models=1)[0]

# Evaluate the best model
loss = best_model.evaluate(X_test_padded, y_test)
print('Best Model Loss:', loss)

y_test = y_test.values.flatten()
# Calculate MAPE
y_pred = best_model.predict(X_test_padded).flatten() # Flatten the output
mape = np.mean(np.abs(y_test - y_pred) / y_test) * 100
print('Best Model MAPE:', mape)

# Ensemble methods
num_models = 5
ensemble_predictions = []

for _ in range(num_models):
    model_clone = clone_model(best_model)
    model_clone.set_weights(best_model.get_weights())
    model_clone.fit(X_train_padded, y_train, batch_size=32, epochs=100, validation_data=(X_test_padded, y_test), callbacks=[EarlyStopping(monitor='val_loss', patience=5)])
    ensemble_predictions.append(model_clone.predict(X_test_padded).flatten())

ensemble_predictions = np.array(ensemble_predictions)
ensemble_mean = np.mean(ensemble_predictions, axis=0)

# Calculate MAPE for ensemble mean
ensemble_mape = np.mean(np.abs(y_test - ensemble_mean) / y_test) * 100
print('Ensemble Mean MAPE:', ensemble_mape)
