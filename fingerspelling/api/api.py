import numpy as np
import pandas as pd
from colorama import Fore, Style, Back
import xgboost as xgb

from fastapi import FastAPI, File, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
from typing import Annotated

from fingerspelling.params import *
from fingerspelling.api.api_code import api_get_landmark_coordinates, load_model, label_converter

from PIL import Image
import io

app = FastAPI()
alphabets_model = load_model(model_type = 'alphabets')
digits_model = load_model(model_type = 'digits')

# Allowing all middleware is optional, but good practice for dev purposes
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

@app.post("/apredict")
async def alphabets_predict(
        file: UploadFile = File(...)
    ):
    """
    Make a single alphabet prediction.
    """
    # Process X_pred
    print(Fore.RED + "\n Processing image." + Style.RESET_ALL)
    contents = await file.read()

    toimage = Image.open(io.BytesIO(contents))
    # resized_image = toimage.resize((256, 256))
    image = np.array(toimage.convert("RGB"))
    print(Back.LIGHTBLACK_EX + f"{image.shape}" + Style.RESET_ALL)

    X_pred = api_get_landmark_coordinates(image)
    print(Back.LIGHTCYAN_EX + f"{X_pred}" + Style.RESET_ALL)

    # Load model
    print(Fore.RED + "\n Loading model." + Style.RESET_ALL)
    model = alphabets_model
    print(Back.LIGHTBLACK_EX + f"{type(model)}" + Style.RESET_ALL)
    assert model is not None

    # # Predict y_pred based on X_pred
    # print(Fore.RED + "\n Predicting character." + Style.RESET_ALL)
    # y_pred = model.predict(pd.DataFrame([X_pred]))
    # sign_result = label_converter(np.asarray(y_pred, dtype=np.int)[0])

    dmatrix = xgb.DMatrix(data=pd.DataFrame([X_pred]))
    prediction = model.predict(dmatrix)
    class_pred = np.argmax(prediction)
    sign_result = label_converter(class_pred)

    print(Back.LIGHTGREEN_EX + f"{sign_result} + {type(sign_result)}" + Style.RESET_ALL)

    print(Fore.GREEN + "\n pred() done." + Style.RESET_ALL)
    print(Fore.BLUE + f"Sign is {sign_result}" + Style.RESET_ALL)

    return dict(sign = sign_result)

@app.post("/dpredict")
async def digits_predict(
        file: UploadFile = File(...)
    ):
    """
    Make a single digit prediction.
    """
    # Process X_pred
    print(Fore.RED + "\n Processing image." + Style.RESET_ALL)
    contents = await file.read()

    toimage = Image.open(io.BytesIO(contents))
    # resized_image = toimage.resize((256, 256))
    image = np.array(toimage.convert("RGB"))
    print(Back.LIGHTBLACK_EX + f"{image.shape}" + Style.RESET_ALL)

    X_pred = api_get_landmark_coordinates(image)
    print(Back.LIGHTCYAN_EX + f"{X_pred}" + Style.RESET_ALL)

    # Load model
    print(Fore.RED + "\n Loading model." + Style.RESET_ALL)
    model = digits_model
    print(Back.LIGHTBLACK_EX + f"{type(model)}" + Style.RESET_ALL)
    assert model is not None

    # Predict y_pred based on X_pred
    print(Fore.RED + "\n Predicting character." + Style.RESET_ALL)
    y_pred = model.predict(pd.DataFrame([X_pred]))
    sign_result = str(y_pred[0])
    print(Back.LIGHTGREEN_EX + f"{sign_result} + {type(sign_result)}" + Style.RESET_ALL)

    print(Fore.GREEN + "\n pred() done." + Style.RESET_ALL)
    print(Fore.BLUE + f"Sign is {sign_result}" + Style.RESET_ALL)

    return dict(sign = sign_result)

@app.get("/")
def root():
    return {'greeting': 'Hello'}
