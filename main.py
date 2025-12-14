from fastapi import FastAPI
from tensorflow.keras.models import load_model
import numpy as np
from fastapi import HTTPException
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from controllers.predict import predict_hate

app = FastAPI()


class TextInput(BaseModel):
    text: str

class ListInput(BaseModel):
    data: list

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def root():
    return {"success": True}

@app.post('/api/predict_one/')
async def predict(data:TextInput):
    try:
        pred = predict_hate(data.text)
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
            pred = predict_hate(text)
            processed.append({"query": text,"prediction":"HATE SPEECH" if pred==1 else "NOT HATE SPEECH"})
        return {"predictions":processed}
    except Exception as e:
        raise HTTPException(
        status_code=500,
        detail="Internal Server error"
    )
