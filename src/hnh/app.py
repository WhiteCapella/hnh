from typing import Annotated
from fastapi import FastAPI, File, UploadFile
from fastapi.templating import Jinja2Templates
from fastapi import Request
from transformers import pipeline
import os
import json


app = FastAPI()
html = Jinja2Templates(directory="public")

@app.get("/predict")
async def create_upload_file(
    file: UploadFile
):
    # save file
    img = await file.read()
    file_name = file.filename
    file_ext = file.content_type.split('/')[-1]

    # if not exist dir, return error
    # then, make dir

    upload_dir = os.path.expanduser("~/images/hnh/")
    if not os.path.exists(upload_dir):
        os.makedirs(upload_dir)
    import uuid
    from PIL import Image
    file_full_path = os.path.join(upload_dir, f'{uuid.uuid4()}.{file_ext}')
    with open(file_full_path, "wb") as f:
        f.write(img)
    
    model = pipeline("image-classification", model = "julien-c/hotdog-not-hotdog")
    img = Image.open(io.BytesIO(img))
    p = model(img)

    return {"Hello" : p}

@app.get("/")
async def home(request: Request):
    hotdog = "https://encrypted-tbn3.gstatic.com/shopping?q=tbn:ANd9GcQweb_7o7OrtlTP75oX2Q_keaoVYgAhMsYVp1sCafoNEdtSSaHps3n7NtNZwT_ufZGPyH7_9MFcao_r8QWr3Fdz17RitvZXLTU4dNsxr73m6V1scsH3_ZZHRw&usqp=CAE"
    dog = "https://hearingsense.com.au/wp-content/uploads/2022/01/8-Fun-Facts-About-Your-Dog-s-Ears-1024x512.webp"
    image_url = random.choice([hotdog, dog])
    return html.TemplateResponse("index.html",{"request":request, "image_url": image_url})
