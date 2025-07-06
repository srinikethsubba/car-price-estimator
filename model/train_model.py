import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
import joblib
import os

df = pd.read_csv("data/vehicles.csv")

df.dropna(subset=['price', 'year', 'manufacturer', 'odometer', 'transmission'], inplace=True)

df = df[['price', 'year', 'manufacturer', 'odometer', 'transmission']]

le_make = LabelEncoder()
df['manufacturer'] = le_make.fit_transform(df['manufacturer'])

le_transmission = LabelEncoder()
df['transmission'] = le_transmission.fit_transform(df['transmission'])

X = df[['year', 'odometer', 'manufacturer', 'transmission']]
y = df['price']

model = RandomForestRegressor()
model.fit(X, y)

os.makedirs("model", exist_ok=True)
joblib.dump(model, "model/car_price_model.pkl")
joblib.dump(le_make, "model/make_encoder.pkl")
joblib.dump(le_transmission, "model/transmission_encoder.pkl")

print("Model and encoders saved to 'model/'")


