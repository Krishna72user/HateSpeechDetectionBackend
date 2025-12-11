from fastapi import FastAPI
from tensorflow.keras.models import load_model
import numpy as np
from fastapi import HTTPException
from pydantic import BaseModel
from utils.preprocess import preprocess

app = FastAPI()
model = load_model("model/hate_speech_model.h5")
threshold = 0.86

class TextInput(BaseModel):
    text: str

class ListInput(BaseModel):
    data: list


@app.get("/")
async def root():
    return {"success": True}

@app.post('/api/predict_one/')
async def predict(data:TextInput):
    try:
        test = preprocess(data.text)
        pred =1  if model.predict(test)>=threshold else 0
        return {"query": data.text,"prediction":"HATE SPEECH" if pred==1 else "NOT HATE SPEECH"}
    except Exception as e:
        raise HTTPException(
        status_code=500,
        detail="Internal Server error"
    )

@app.post('/api/predict_many')
async def predict_many(data:ListInput):
    try:
        processed=[]
        for text in data.data:
            pro_text = preprocess(text)
            pred =1  if model.predict(pro_text)>=threshold else 0
            processed.append({"query": text,"prediction":"HATE SPEECH" if pred==1 else "NOT HATE SPEECH"})
        return {"predictions":processed}
    except Exception as e:
        raise HTTPException(
        status_code=500,
        detail="Internal Server error"
    )
