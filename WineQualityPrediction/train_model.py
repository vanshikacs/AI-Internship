import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier

# Load dataset
data = pd.read_csv("data/winequality-red.csv", sep=";")

# Features and target
X = data.drop("quality", axis=1)
y = data["quality"]

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42
)

# Train model
model = MLPClassifier(
    hidden_layer_sizes=(64, 32),
    activation="tanh",
    max_iter=500,
    random_state=42
)

model.fit(X_train, y_train)

print("Accuracy:", model.score(X_test, y_test))

# Save model
joblib.dump(model, "wine_model.pkl")

print("Model saved successfully!")