from fastapi import FastAPI, Request
from pydantic import BaseModel
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.utils import to_categorical
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
import json

# Initialize FastAPI app
app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Adjust for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load translations
try:
    with open("translations.json", "r", encoding="utf-8") as f:
        translation_map = json.load(f)
except FileNotFoundError:
    translation_map = {"categorical": {}, "diseases": {}}

# Create reverse translation map (Khmer to English)
reverse_translation_map = {
    "categorical": {
        column: {khmer: eng for eng, khmer in values.items()}
        for column, values in translation_map["categorical"].items()
    },
    "diseases": {khmer: eng for eng, khmer in translation_map["diseases"].items()}
}

# Load and preprocess dataset
dataset_path = "dataset.csv"
symptom_data = pd.read_csv(dataset_path)

# Define features and target dynamically
target = 'Disease'
features = [col for col in symptom_data.columns if col != target and col != 'Outcome Variable']
categorical_columns = [col for col in features if symptom_data[col].dtype == 'object']  # Detect categorical columns
numerical_columns = [col for col in features if col not in categorical_columns]  # e.g., Age

# Encode categorical variables with LabelEncoder for initial mapping
label_encoders = {}
for column in categorical_columns:
    le = LabelEncoder()
    symptom_data[column] = le.fit_transform(symptom_data[column])
    label_encoders[column] = le

# Encode target variable
target_encoder = LabelEncoder()
symptom_data[target] = target_encoder.fit_transform(symptom_data[target])

# One-hot encode categorical features for neural network
X = pd.get_dummies(symptom_data[features], columns=categorical_columns)
y = to_categorical(symptom_data[target])  # Convert target to one-hot for neural network

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define and train neural network
model = Sequential([
    Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
    Dense(32, activation='relu'),
    Dense(y.shape[1], activation='softmax')  # Output layer matches number of diseases
])
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(X_train, y_train, epochs=50, batch_size=32, verbose=0)

# Evaluate model
test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0)
print(f"Test Accuracy: {test_accuracy:.2f}")

# Dynamic Pydantic model for request body
class SymptomInput(BaseModel):
    class Config:
        arbitrary_types_allowed = True
        extra = 'allow'  # Allow extra fields for new symptoms

# Endpoint to get valid input options in Khmer and English
@app.get("/valid_inputs")
async def get_valid_inputs(language: str = "khmer"):
    valid_inputs = {}
    for column in categorical_columns:
        valid_english = list(label_encoders[column].classes_)
        if language.lower() == "khmer":
            valid_inputs[column] = [
                translation_map["categorical"].get(column, {}).get(eng, eng)
                for eng in valid_english
            ]
        else:
            valid_inputs[column] = valid_english
    for column in numerical_columns:
        valid_inputs[column] = {"type": "integer", "example": symptom_data[column].mean()}
    return {"valid_inputs": valid_inputs}

# Prediction endpoint
@app.post("/predict")
async def predict_disease(symptoms: SymptomInput, language: str = "khmer"):
    try:
        # Translate Khmer inputs to English and build input data
        input_dict = symptoms.dict()
        input_data = {}
        for column in features:
            if column in numerical_columns:
                input_data[column] = [input_dict.get(column, symptom_data[column].mean())]
            else:  # Categorical
                value = input_dict.get(column, list(label_encoders[column].classes_)[0])
                input_data[column] = [reverse_translation_map["categorical"].get(column, {}).get(value, value)]

        input_data = pd.DataFrame(input_data)

        # Encode categorical inputs
        for column in categorical_columns:
            try:
                input_data[column] = label_encoders[column].transform(input_data[column])
            except ValueError:
                valid_english = list(label_encoders[column].classes_)
                valid_khmer = [translation_map["categorical"].get(column, {}).get(eng, eng) for eng in valid_english]
                return {
                    "error": f"Invalid value for {column}. Expected one of {valid_khmer} (Khmer) or {valid_english} (English)",
                    "status": "error"
                }

        # One-hot encode input
        input_encoded = pd.get_dummies(input_data, columns=categorical_columns)
        # Align input with training features
        for col in X_train.columns:
            if col not in input_encoded.columns:
                input_encoded[col] = 0
        input_encoded = input_encoded[X_train.columns]

        # Make prediction
        probabilities = model.predict(input_encoded, verbose=0)[0]
        prediction = np.argmax(probabilities)
        predicted_disease = target_encoder.inverse_transform([prediction])[0]
        confidence = float(probabilities[prediction])

        # Translate output
        if language.lower() == "khmer":
            predicted_disease_output = translation_map["diseases"].get(predicted_disease, predicted_disease)
        else:
            predicted_disease_output = predicted_disease

        return {
            "predicted_disease": predicted_disease_output,
            "confidence": confidence,
            "status": "success"
        }
    except Exception as e:
        return {"error": str(e), "status": "error"}

# Endpoint to get column names
@app.get("/columns")
async def get_columns():
    return {"columns": symptom_data.columns.tolist()}

# Root endpoint
@app.get("/")
async def root():
    return {"message": "Welcome to the Disease Prediction API (Khmer & English Support)"}