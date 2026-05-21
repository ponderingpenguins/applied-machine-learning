from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()
model = 

class InputGaitData(BaseModel):
    #TODO: verify data type for the input
    gyr_x: float
    gyr_y: float
    gyr_z: float
    
    acc_x: float
    acc_y: float
    acc_z: float
    
@app.post("/encode_gait")
async def encode_gait(data: InputGaitData):
    embedding = model.encode(data)
    return {"embedding": embedding.tolist()}

@app.post("/classify_gait")
async def classify_gait(data: InputGaitData):
    prediction = model.predict(data)
    return {"prediction": prediction}