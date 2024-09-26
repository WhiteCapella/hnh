from typing import Annotated
from fastapi import FastAPI, File, UploadFile
import os



app = FastAPI()

@app.get("/uploadfile")
