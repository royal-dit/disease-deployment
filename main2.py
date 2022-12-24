from fastapi import  FastAPI,File,UploadFile
import uvicorn
import  numpy as np
from io import  BytesIO
from PIL import  Image
import pickle


app = FastAPI()
Mod = pickle.load(open('model.pkl','rb'))
ClASS_NAMES = ["Early Blight","Late Blight","Healthy"]


@app.get("/ping")
async def ping():
    return "hellow i am alive"

def read_file_as_image(data) -> np.ndarray:
    image = np.array(Image.open(BytesIO(data)))
    return image

@app.post("/predict",method=['POST'])
async def predict(file:UploadFile = File(...)):
    image = read_file_as_image(await file.read()) #read images form user
    img_batch = np.expand_dims(image,0)     #incerease the dimension
    predictions = Mod.predict(img_batch)   #predict the image
    index = np.argmax(predictions[0])        #takes the highest value of as index
    predicted_class = ClASS_NAMES[index]     #taking class name of highst index
    confidence = np.max(predictions)         #takes the higesht value as value
    return {
        'class':predicted_class,
        'confidence':float(confidence)
    }


if __name__ == "__main__":
      uvicorn.run("main:app", port=5000, log_level="info")





