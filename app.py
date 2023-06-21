# -*- coding: utf-8 -*-

import streamlit as st
import pandas as pd
import random
import matplotlib.pyplot as plt
import cv2
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torch.nn as nn
import albumentations as A
from albumentations.pytorch import ToTensorV2
from efficientnet_pytorch import EfficientNet
from tqdm.notebook import tqdm
import joblib
import googlemaps
import requests
import os

data_path = "/img"
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def inference(model, image_path):
    """
    주어진 이미지에 대해 모델을 통해 예측을 수행하고 결과를 출력하는 함수입니다.

    Args:
        model (torch.nn.Module): 사용할 모델
        image_path (str): 이미지 파일 경로

    Returns:
        str: 예측 결과로 얻은 주소 정보
    """
    model.eval()
    transform = A.Compose([
        A.Resize(600, 600),
        A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.3),
        A.VerticalFlip(p=0.2),
        A.HorizontalFlip(p=0.5),
        A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.2, rotate_limit=30, p=0.3),
        A.OneOf([A.Emboss(p=1), A.Sharpen(p=1), A.Blur(p=1)], p=0.3),
        A.PiecewiseAffine(p=0.3),
        A.Normalize(),
        ToTensorV2()
        ])

    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    preprocessed_image = transform(image=image)["image"]
    preprocessed_image = preprocessed_image.unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(preprocessed_image)
        probabilities = torch.softmax(output, dim=1)
        predicted_class = torch.argmax(probabilities, dim=1).item()

    class_names = ["spring_mountain", "summer_mountain", "autumn_mountain", "winter_mountain", "forestfires_occur_early", "forestfires_occur"]

    predicted_label = class_names[predicted_class]
    predicted_probs = probabilities.squeeze().cpu().numpy()

    # location = get_current_location(api_key)
    # latitude, longitude = location
    #
    # address = reverse_geocode(latitude, longitude, api_key)

    print("=================================================")
    top1_labels = [class_names[i] for i in np.argsort(predicted_probs)[-1:]]
    if "forestfires_occur_early" in top1_labels or "forestfires_occur" in top1_labels:
        print("WARNING: Forest fires occur suspected!")
        # print("=================================================")
        # print(f"{address}, ({latitude}, {longitude})")
        # print("=================================================")

    else:
        print("Non Forest fires occur suspected.")
        # print("=================================================")
        # print(f"{address}, ({latitude}, {longitude})")
        # print("=================================================")

    print("Predicted Class and Probabilities:")
    for label, prob in zip(class_names, predicted_probs):
        print(f"{label}: {prob*100:.2f}%")

    plt.imshow(image)
    plt.title(predicted_label)
    plt.axis("off")
    plt.show()

if __name__ == "__main__":
    model = joblib.load('EfficientNet_7.pkl')
    simulation = pd.read_csv(data_path + "/simulation.csv")

    random_number = random.randint(0, 19)
    image_path = data_path + '/simulation/' + simulation["image_name"][random_number]

    inference(model, image_path)