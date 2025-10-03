import json
import pickle
from pathlib import Path

import numpy as np
import streamlit as st


APP_DIR = Path(__file__).parent
COLUMNS_JSON_PATH = APP_DIR / "columns.json"
MODEL_PATH = APP_DIR / "mumbai_home_prices_model.pickle"


with open(COLUMNS_JSON_PATH, "r", encoding="utf-8") as f:
	_columns = json.load(f)["data_columns"]

with open(MODEL_PATH, "rb") as f:
	_model = pickle.load(f)


def build_feature_row(columns, area_sqft, bathrooms, bhk, location_name):
	row = np.zeros(len(columns))
	try:
		row[columns.index("builduparea_sqft")] = float(area_sqft)
	except ValueError:
		pass
	try:
		row[columns.index("bathrooms")] = float(bathrooms)
	except ValueError:
		pass
	try:
		row[columns.index("bhk")] = float(bhk)
	except ValueError:
		pass

	row[columns.index(location_name)] = 1
	return row.reshape(1, -1)


def predict_price():
	st.title("Mumbai Property Price Predictor")
	st.write("Enter property details")

	area_sqft = st.number_input("Total area (sqft)", min_value=100.0, max_value=20000.0, step=10.0)
	bhk = st.number_input("BHK", min_value=1, max_value=12, step=1)
	bathrooms = st.number_input("Bathrooms", min_value=1, max_value=10, step=1)

	location_columns = [c for c in _columns if c not in ("builduparea_sqft", "bathrooms", "bhk", "other", "others")]
	location = st.selectbox("Location", options=sorted(location_columns))

	if st.button("Predict Price"):
		features = build_feature_row(_columns, area_sqft, bathrooms, bhk, location)
		try:
			prediction = _model.predict(features)
			price_value = float(prediction[0])
			st.success(f"Estimated Price: {price_value:.2f} Cr")
		except Exception as e:
			st.error(f"Prediction failed: {e}")


if __name__ == "__main__":
	predict_price()
