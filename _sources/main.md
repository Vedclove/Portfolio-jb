# Iris Prediction API Code

Below is the FastAPI code to create an API for predicting Iris species using a trained model.

```python
from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
import joblib

# Load the trained model
model = joblib.load('Iris-model.pkl') 

# Initialize FastAPI app
app = FastAPI()

# Define input data schema using Pydantic
class IrisInput(BaseModel):
    sepal_length: float
    sepal_width: float
    petal_length: float
    petal_width: float

# Map predicted classes to species names
species_dict = {0: "Iris-Setosa", 1: "Iris-Versicolor", 2: "Iris-Virginica"}

# Define Root
@app.get("/")
def root():
    return {"message": "Welcome to the Iris Prediction API!"}

# Define a prediction endpoint
@app.post("/predict")
def predict(input_data: IrisInput):
    # Convert input data to a NumPy array
    input_array = np.array([[input_data.sepal_length, input_data.sepal_width,
                             input_data.petal_length, input_data.petal_width]])
    
    # Make a prediction
    prediction = model.predict(input_array)
    
    # Get the species name 
    predicted_species = species_dict[prediction[0]]
    
    return {predicted_species}
