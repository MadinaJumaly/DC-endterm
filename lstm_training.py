import tensorflow as tf
import numpy as np

def create_lstm_model(input_shape):
    model = tf.keras.models.Sequential([
        tf.keras.layers.LSTM(50, return_sequences=True, input_shape=input_shape),
        tf.keras.layers.LSTM(50, return_sequences=False),
        tf.keras.layers.Dense(25),
        tf.keras.layers.Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse')
    return model

def train_model(data):
    X_train, y_train = [], []
    for i in range(50, len(data)):
        X_train.append(data[i-50:i])
        y_train.append(data[i])
    X_train, y_train = np.array(X_train), np.array(y_train)

    model = create_lstm_model((X_train.shape[1], 1))
    model.fit(X_train, y_train, epochs=10, batch_size=32, verbose=1)
    return model

if __name__ == "__main__":
    data = np.random.rand(200)  # Replace with actual data
    model = train_model(data)
