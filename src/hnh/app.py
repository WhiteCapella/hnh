from typing import Annotated
from fastapi import FastAPI, File, UploadFile
import os



app = FastAPI()

@app.get("/uploadfile")
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
    file_full_path = os.path.join(upload_dir, f'{uuid.uuid4()}.{file_ext}')
    with open(file_full_path, "wb") as f:
        f.write(img)

@app.get("/predict")
async def predict():
    print("hello world!")
