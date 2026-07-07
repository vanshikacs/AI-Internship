import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split

# Load dataset
data = pd.read_csv("data/winequality-red.csv", sep=";")

# Display first 5 rows
print(data.head())

# Features (X) and Target (y)
X = data.drop("quality", axis=1)
y = data["quality"]

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Build the neural network
model = tf.keras.Sequential([
    tf.keras.layers.Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dense(1)
])

# Compile the model
model.compile(
    optimizer='adam',
    loss='mse',
    metrics=['mae']
)

# Train the model
model.fit(
    X_train,
    y_train,
    epochs=50,
    validation_split=0.2
)

# Evaluate the model
loss, mae = model.evaluate(X_test, y_test)

print(f"\nTest MAE: {mae:.2f}")