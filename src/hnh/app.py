from typing import Annotated
from fastapi import FastAPI, File, UploadFile
from fastapi.templating import Jinja2Templates
from fastapi import Request
from transformers import pipeline
import tensorflow as tf
import os
import json
import random

app = FastAPI()
html = Jinja2Templates(directory="public")

@app.get("/predict")
async def create_upload_file(
    file: UploadFile
):
    try:
        # save file
        img = await file.read()
        from PIL import Image
        model = pipeline("image-classification", model = "julien-c/hotdog-not-hotdog")
        img = Image.open(io.BytesIO(img))
        prediction = model(img)

        # 예측 결과 반환 (클라이언트가 이해하기 쉽도록)
        return {"prediction": prediction}

    except Exception as e:
        # 예외 처리
        return {"error": str(e)}

@app.get("/")
async def home(request: Request):
    hotdog = "https://encrypted-tbn3.gstatic.com/shopping?q=tbn:ANd9GcQweb_7o7OrtlTP75oX2Q_keaoVYgAhMsYVp1sCafoNEdtSSaHps3n7NtNZwT_ufZGPyH7_9MFcao_r8QWr3Fdz17RitvZXLTU4dNsxr73m6V1scsH3_ZZHRw&usqp=CAE"
    dog = "https://hearingsense.com.au/wp-content/uploads/2022/01/8-Fun-Facts-About-Your-Dog-s-Ears-1024x512.webp"
    image_url = random.choice([hotdog, dog])
    return html.TemplateResponse("index.html",{"request":request, "image_url": image_url})
