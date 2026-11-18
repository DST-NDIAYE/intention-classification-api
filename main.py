import pickle
from fastapi import FastAPI
from pydantic import BaseModel

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
    return {"intention": intention, "confiance": round(float(confiance), 3)}