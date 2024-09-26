from typing import Annotated
from fastapi import FastAPI, File, UploadFile
from fastapi.templating import Jinja2Templates
from fastapi import Request
from transformers import pipeline
from hnh.util import get_max_score, get_max_label
import tensorflow as tf
import os
import json
import random
import io

app = FastAPI()
html = Jinja2Templates(directory="public")


@app.get("/")
async def home(request: Request):
    hotdog = "https://encrypted-tbn3.gstatic.com/shopping?q=tbn:ANd9GcQweb_7o7OrtlTP75oX2Q_keaoVYgAhMsYVp1sCafoNEdtSSaHps3n7NtNZwT_ufZGPyH7_9MFcao_r8QWr3Fdz17RitvZXLTU4dNsxr73m6V1scsH3_ZZHRw&usqp=CAE"
    dog = "https://hearingsense.com.au/wp-content/uploads/2022/01/8-Fun-Facts-About-Your-Dog-s-Ears-1024x512.webp"
    image_url = random.choice([hotdog, dog])
    return html.TemplateResponse("index.html",{"request":request, "image_url": image_url})


@app.get("/predict")
def hotdog():
    model = pipeline("image-classification", model="julien-c/hotdog-not-hotdog") 
    return {"Hello": random.choice(["hotdog", "not hotdog"])}


@app.post("/uploadfile/")
async def create_upload_file(file: UploadFile):
    # 파일 저장
    img = await file.read()
    model = pipeline("image-classification", model="julien-c/hotdog-not-hotdog")
    
    from PIL import Image
    img = Image.open(io.BytesIO(img))  # 이미지 바이트를 PIL 이미지로 변환
    
    filename = f"{uuid.uuid4()}.jpg"
    upload_folder = "uploads"
    if not os.path.exists(upload_folder):
        os.makedirs(upload_folder)
    with open(os.path.join(upload_folder, filename), "wb") as f:
        f.write(img)

    p = model(img)
    label = get_max_label(p)
    score = get_max_score(p)
    #{'label': 'hot dog', 'score': 0.54},
    #{'label': 'not hot dog', 'score': 0.46}
    return {
        "image_url": f"/uploads/{filename}",  # 이미지 URL
        "label": label, 
        "score": score
    }

@app.get("/uploads/{filename}")
async def get_uploaded_file(filename: str):
    return FileResponse(os.path.join("uploads", filename))
