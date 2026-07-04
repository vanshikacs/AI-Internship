import pandas as pd
import joblib

# Load saved files
m = joblib.load("house_rent_prediction.pkl")
e = joblib.load("encoder.pkl")
s = joblib.load("scaler.pkl")

# New house data
new_house = pd.DataFrame({
    'BHK': [2],
    'Size': [1200],
    'Area Type': ['Super Area'],
    'City': ['Delhi'],
    'Furnishing Status': ['Semi-Furnished'],
    'Tenant Preferred': ['Family'],
    'Bathroom': [2],
    'Point of Contact': ['Contact Owner']
})

# Encode
new_encoded = e.transform(new_house)

# Scale
new_scaled = s.transform(new_encoded)

# Predict
predicted_rent = m.predict(new_scaled)

print("Predicted Rent:", predicted_rent[0])