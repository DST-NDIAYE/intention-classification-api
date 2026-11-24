import pickle
from fastapi import FastAPI
from pydantic import BaseModel


import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("Prediction ")

from datetime import datetime, timezone
datetime.now(timezone.utc).isoformat()

import json




app =  FastAPI( title= "Classification d'intentions"  )

with open( "model.pkl" , "rb" ) as f :
    model = pickle.load(f)



class Requete(BaseModel):
    text: str

@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict")
def predict(req: Requete):
    intention = model.predict([req.text])[0]
    confiance = model.predict_proba([req.text]).max()

    date = datetime.now(timezone.utc).isoformat()

    logs = {
        "text" : req.text , 
        "intention" : intention , 
        "confiance" : confiance , 
        "date" : date
    }

    logger.info(json.dumps(logs , ensure_ascii=False))

    return {"intention": intention, "confiance": round(float(confiance), 3)}