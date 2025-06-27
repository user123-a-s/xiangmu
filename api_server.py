from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import os

app = FastAPI()

# 指定模型文件的路径
model_path = r"E:\py\bishe\weather_predict_api\model\weather_model.pkl"
model = joblib.load(model_path)  # 加载模型

class InputData(BaseModel):
    temperature: float
    humidity: float
    wind_speed: float

@app.post("/predict")
def predict(data: InputData):
    features = [[data.temperature, data.humidity, data.wind_speed]]
    result = model.predict(features)[0]
    return {"prediction": str(result)}