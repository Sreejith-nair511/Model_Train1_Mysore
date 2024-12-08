import tensorflow as tf

def evaluate_model(model, X_test, y_test):
    predictions = model.predict(X_test)
    mse = tf.keras.losses.mean_squared_error(y_test, predictions)
    print(f'Mean Squared Error: {mse}')
