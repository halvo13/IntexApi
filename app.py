import pickle
import pandas as pd
from sklearn import preprocessing
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/", tags=["Root"])
async def read_root():
    return {"message": "Welcome to Jake's API!"}

# Define endpoint for making predictions
@app.post('/predict')
def predict(data:dict):
  # Load model from .pkl file
  with open('./model.pkl','rb') as file:
    model = pickle.load(file)
    # Convert input data to DataFrame
    df = pd.DataFrame(data, index=[0])
    # Make prediction
    prediction = model.predict(df)
    # Return Prediction as JSON response
    return {'prediction': prediction[0]}
  

#   Elastic Ip Address = 3.220.72.68

# ORDER FOR POSTMAN :
# 'squarenorthsouth'
# 'squareeastwest'
# 'adultsubadult_A'
# 'adultsubadult_C'
# 'adultsubadult_N LL'
# 'area_NE'
# 'area_NNW'
# 'area_NW'
# 'area_SE'
# 'area_SW'
# 'depthlevel_Deep'
# 'depthlevel_MidDeep'
# 'depthlevel_MidShallow'
# 'depthlevel_Shallow'