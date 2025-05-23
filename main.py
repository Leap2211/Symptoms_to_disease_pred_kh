from fastapi import FastAPI, Request
from pydantic import BaseModel
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from fastapi.middleware.cors import CORSMiddleware
import numpy as np


app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

#read dataset
dataset_path = ("dataset.csv")
symptom_data = pd.read_csv(dataset_path)
print("Dataset Columns:", symptom_data.columns.tolist())

#define feature and target
features = [
    'Fever',
    'Cough',
    'Fatigue',
    'Difficulty Breathing',
    'Age',
    'Gender',
    'Blood Pressure',
    'Cholesterol Level'
]

target = 'Disease'

#Encode categorical variables

label_encoders = {}

#These columns contain non-numerical values:
# Fever, Cough, Fatigue, Difficulty Breathing: Yes or No
# Gender: Male or Female
# Blood Pressure, Cholesterol Level: Low, Normal, or High

#encoding categorical only string, age is already numerical 

for column in ['Fever', 'Cough', 'Fatigue', 'Difficulty Breathing', 'Gender', 'Blood Pressure', 'Cholesterol Level']:
    #create label encoder
    le = LabelEncoder()
    #fit (learn unique categories and assign each a number) and transform(replace original string value with the number)
    symptom_data[column] = le.fit_transform(symptom_data[column])
    print(f"Symptop_data_column: {symptom_data}")
    #store encoder to dictionary
    label_encoders[column] = le
    print(f"\n Label Encoders: {label_encoders} \n")

#encode the target columns
target_encoder = LabelEncoder()
#fit(Learns all unique disease names and assign unique integer) and transform(replace original string with unique integer)
symptom_data[target] = target_encoder.fit_transform(symptom_data[target])

#split data 
#Split dataset into training and testing sets to prepare for model training and evaluation

#X is a DataFrame containing the features with encoded value
X = symptom_data[features]
print(f"X: {X}")
#y is a Series containing the target with encoded value
y = symptom_data[target]
print(f"y: {y}")
#test 80%, test 20% test_size=0.2
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#train model 
#initialize and trains the DecisionTreeClassifier to learn patterns for predicting disease from input symptoms
#random_state = 42, ensures consistent results 
model = RandomForestClassifier(n_estimators=100, random_state=42)
#Trained once in setup in one go
model.fit(X_train, y_train)

#class Symtom Input

class SymptomInput(BaseModel):
    Fever: str
    Cough: str
    Fatigue: str
    Difficulty_Breathing: str
    Age: int
    Gender: str
    Blood_Pressure: str
    Cholesterol_Level: str

#prediction endpoint
@app.post("/predict")

async def predict_disease(symptom: SymptomInput):
    try:
        #convert input data to DataFrame
        input_data = pd.DataFrame({
            'Fever': [symptom.Fever],
            'Cough': [symptom.Cough],
            'Fatigue': [symptom.Fatigue],
            'Difficulty Breathing': [symptom.Difficulty_Breathing],
            'Age': [symptom.Age],
            'Gender': [symptom.Gender],
            'Blood Pressure': [symptom.Blood_Pressure],
            'Cholesterol Level': [symptom.Cholesterol_Level]
        })

        print(f"DataFrame: {input_data}")

        #encode input data

        for column in ['Fever', 'Cough', 'Fatigue', 'Difficulty Breathing', 'Gender', 'Blood Pressure', 'Cholesterol Level']:
            try:
                input_data[column] = label_encoders[column].transform(input_data[column])
            except:
                return {"error": f"Invalid value for {column}. Expected one of {list(label_encoders[column].classes_)} "}

        #make prediction 
        prediction = model.predict(input_data)
        predicted_disease = target_encoder.inverse_transform(prediction)[0]

        return {
            "predicted_disease" : predicted_disease,
            "status" : "success"
        }

    except:
        return {"error": str(e), "status": "error"}


#Root endpoint

@app.get("/")
async def root():
    return {"message": "Welcome to the disease prediction API"}