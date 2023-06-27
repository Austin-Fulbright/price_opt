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


from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize


# Load the dataset into a pandas DataFrame
data = pd.read_csv('data/marketing_sample_for_amazon_com-ecommerce__20200101_20200131__10k_data.csv')
data = data.dropna(subset=['About Product', 'Selling Price'])
data.rename(columns={'Uniq Id':'Id', 'Shipping Weight':'Shipping Weight(Pounds)', 'Selling Price':'Selling Price($)'}, inplace=True)


# Data preprocessing
data['Selling Price($)'] = data['Selling Price($)'].str.replace('$', '').str.replace(' ', '').str.split('.').str[0] + '.'
data = data[~data['Selling Price($)'].str.contains('[a-zA-Z]', na=False)]
data['Selling Price($)'] = data['Selling Price($)'].str.replace(',', '').astype(float)
data['Selling Price($)'] = data['Selling Price($)'].apply(lambda x: "{:.2f}".format(x)).astype(float)


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


# Save the preprocessed data into a CSV file
data.to_csv('preprocessed_data.csv', index=False)


# Split the dataset into training and testing sets
# Split the dataset into features and target
features = data['preprocessed_description']
target = data['Selling Price($)']


# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)


# Save the training data into a CSV file
train_data = pd.concat([X_train, y_train], axis=1)
train_data.to_csv('training_data.csv', index=False)


# Tokenize the text data
tokenizer = Tokenizer()
tokenizer.fit_on_texts(X_train)


X_train_seq = tokenizer.texts_to_sequences(X_train)
X_test_seq = tokenizer.texts_to_sequences(X_test)


# Pad sequences to a fixed length
max_sequence_length = 100
X_train_padded = pad_sequences(X_train_seq, maxlen=max_sequence_length)
X_test_padded = pad_sequences(X_test_seq, maxlen=max_sequence_length)


# Convert the target variable to numpy arrays
y_train_np = np.array(y_train)
y_test_np = np.array(y_test)


# Build the RNN model with increased complexity
embedding_dim = 100

model = Sequential()
model.add(Embedding(len(tokenizer.word_index) + 1, embedding_dim, input_length=max_sequence_length))
model.add(LSTM(256, return_sequences=True, kernel_initializer=GlorotUniform(seed=42), recurrent_regularizer=l2(1e-5)))
model.add(Dropout(0.5))
model.add(LSTM(128, kernel_initializer=GlorotUniform(seed=42), recurrent_regularizer=l2(1e-5)))
model.add(Dropout(0.5))
model.add(Dense(64, activation='relu', kernel_regularizer=l2(1e-5)))
model.add(Dropout(0.5))
model.add(Dense(1, activation='linear'))


model.compile(loss='mean_squared_error', optimizer='adam')


# Train the model with early stopping
batch_size = 64
epochs = 100
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)


model.fit(X_train_padded, y_train_np, batch_size=batch_size, epochs=epochs, validation_data=(X_test_padded, y_test_np), callbacks=[early_stopping])


# Evaluate the model
loss = model.evaluate(X_test_padded, y_test_np)
print("Model loss:", loss)


# Predict on test data
y_pred = model.predict(X_test_padded)


# Rescale the predicted prices back to the original scale
y_pred = scaler.inverse_transform(y_pred)


# Calculate the mean absolute percentage error (MAPE)
mape = np.mean(np.abs((scaler.inverse_transform(y_test_np.reshape(-1, 1)) - y_pred) / scaler.inverse_transform(y_test_np.reshape(-1, 1)))) * 100
print("MAPE:", mape)
