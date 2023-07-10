import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

import numpy as np

import pickle
import os
import cv2 as cv

import xgboost

from fingerspelling.params import *

def pred_preprocessor(image):
    '''
    Parameters: image -> numpy ndarray
    Returns: final_image -> numpy ndarray
    Preprocessed by cropping the hand and padding to square followed
    by resizing to 200 x 200 x 3
    '''
    width = image.shape[1]
    height = image.shape[0]
    if height != width:
        if width > height:
            padding = int((width - height)/2)
            final_img = np.vstack((np.zeros((padding, width, 3)), image,
                                np.zeros((padding, width, 3))))
        elif height > width:
            padding = int((height-width)/2)
            final_img = np.hstack((np.zeros((height, padding, 3)), image,
                                np.zeros((height, padding, 3))))
    else:
        final_img = image
    width = final_img.shape[1]
    height = final_img.shape[0]
    final_img = np.vstack((np.zeros((100, width, 3)), final_img,
                                np.zeros((100, width, 3))))
    final_img = np.hstack((np.zeros((height+200, 100, 3)), final_img,
                                np.zeros((height+200, 100, 3))))
    final_img = cv.resize(final_img, (200, 200)).astype('uint8')
    # final_img = cv.cvtColor(final_img, cv.COLOR_BGR2RGB)
    return final_img

def cropped_image(frame, x1, x2, y1, y2):
    return frame[y1:y2, x1:x2]

def padded_image(frame):
    padded = pred_preprocessor(frame)
    results = hands.process(padded)
    if results.multi_hand_landmarks:
        final_coords = []
        for hand_landmarks in results.multi_hand_landmarks:
            for i in range(len(hand_landmarks.landmark)):
                x = hand_landmarks.landmark[i].x
                y = hand_landmarks.landmark[i].y
                # z = hand_landmarks.landmark[i].z # Comment this out if using 42
                final_coords.append(x)
                final_coords.append(y)
                # final_coords.append(z) # Comment this out if using 42
        return final_coords

def api_detection(image, training=False):
    '''
    Input image as np array.
    Output detection_result to pass into get_black_image
    '''
    # STEP 1: Create an HandLandmarker object.
    base_options = python.BaseOptions(model_asset_path='hand_landmarker.task')
    options = vision.HandLandmarkerOptions(base_options=base_options, num_hands=1)
    detector = vision.HandLandmarker.create_from_options(options)

    # STEP 2: Load the input image.
    # image_processed = mp.Image.create_from_file(image_path)
    if training:
        image_processed = mp.Image.create_from_file(image)
    else:
        image_processed = mp.Image(image_format=mp.ImageFormat.SRGB, data=image)

    # STEP 3: Detect hand landmarks from the input image.
    detection_result = detector.detect(image_processed)

    return detection_result

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence = 0.3,
                       max_num_hands=3)

def api_get_landmark_coordinates(image, training=False):
    '''
    Takes one image path and
    Returns a list of landmark coordinates
    '''
    coords = []
    results = hands.process(image)
    x_ = []
    y_ = []

    for hand_landmarks in results.multi_hand_landmarks:
        for i in range(len(hand_landmarks.landmark)):
            x = hand_landmarks.landmark[i].x
            y = hand_landmarks.landmark[i].y
            coords.append(x)
            coords.append(y)
            x_.append(x)
            y_.append(y)

        H, W, _ = image.shape
        pad_size = 40

        if int(min(x_) * W)-pad_size < 0 or int(max(x_) * W)+pad_size > W:
            x1, x2 = 0, W
        else:
            x1, x2 = int(min(x_) * W)-pad_size, int(max(x_) * W)+pad_size
        if int(min(y_) * H)-pad_size < 0 or int(max(y_) * H)+pad_size > W:
            y1, y2 = 0, H
        else:
            y1, y2 = int(min(y_) * H)-pad_size, int(max(y_) * H)+pad_size

        cropped = cropped_image(image, x1, x2, y1, y2)
        final_coords = padded_image(cropped)

    return final_coords

def load_model(model_type = 'alphabets'):
    if model_type == 'alphabets':
        model_path = 'alphabets_model.pkl'

    if model_type == 'digits':
        model_path = 'digits_model.pkl'

    model = pickle.load(open(model_path, "rb"))
    return model

def label_converter(raw_pred: int) -> str:
    label_dict = {
    0: 'A',
    1: 'B',
    2: 'C',
    3: 'D',
    4: 'E',
    5: 'F',
    6: 'G',
    7: 'H',
    8: 'I',
    9: 'J',
    10: 'K',
    11: 'L',
    12: 'M',
    13: 'N',
    14: 'O',
    15: 'P',
    16: 'Q',
    17: 'R',
    18: 'S',
    19: 'T',
    20: 'U',
    21: 'V',
    22: 'W',
    23: 'X',
    24: 'Y',
    25: 'Z',
}

    return label_dict[raw_pred]
