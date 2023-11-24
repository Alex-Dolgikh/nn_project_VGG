import numpy as np
import pandas as pd

import matplotlib as mpl
import matplotlib.pyplot as plt

import streamlit as st
# import plotly_express as px
from plotly.tools import mpl_to_plotly
import io
import imageio
import requests
from PIL import Image
import torch 
import torch.nn as nn

from models.preprocess import preprocess
# -------- coffee -------------


from torchvision.models import resnet50, ResNet50_Weights
model_coffe = resnet50(weights=ResNet50_Weights.DEFAULT)
model_coffe.fc = nn.Linear(2048,4)
model_coffe.load_state_dict(torch.load('models/coffee_save.pt', map_location=torch.device('cpu')))
model_coffe.eval()
coffee_dict = {0: 'Dark', 1: 'Green', 2: 'Light', 3: 'Medium'}

# --------- crops ---------------





st.title('ПРОЕКТ: Определение агро-культур и цвета зёрен кофе')

st.sidebar.header('Выберите страницу')
page = st.sidebar.radio("Выберите страницу", ["Вводная информация", "Кофе", "Агрокультуры", "Магическая страница"])

if page == "Вводная информация":
        
        st.header('Задачи:')
        st.subheader('*Задача №1*: Классификация кофе')
        st.write('Текст задания: Сделайте страницу, позволяющую загрузить пользовательскую фотографию зерна кофе и получить класс. Для обучения используйте датасет Coffee Beans и модель ResNet18/50/101/152')

        st.subheader('*Задача №2*: Классификация агро-культур')
        st.write('Текст задания: Сделайте страницу, позволяющую классифицировать фотографию агрокультуры. В качестве обучающих данных используйте датасет изображений Agricultural crops. Модель может быть разработана самостоятельно, либо аналогично предыдущему заданию.')

        st.subheader('*Задача №3*: Классификация всего подряд')
        st.write('Текст задания: Сделайте страницу, в которую можно загрузить произвольное изображение и получить результат классификации сразу двумя полностью предобученными (= самостоятельно в модели ничего не меняем) моделями. На странице должно быть отображено топ-5 предсказанных категорий и вероятности классов. Также рядом должно быть подписано время, за которое получен результат.')

        st.header('Выполнила команда "VGG":')
        st.subheader('Алексей Долгих')
        st.subheader('Алиса Жгир')


if page == "Кофе":

    image_url = st.text_input("Введите URL картинки кофейного зерна")

    if image_url:
                response = requests.get(image_url)
                image = Image.open(io.BytesIO(response.content))
                st.subheader('Загруженная картинка')
                st.image(image)

    uploaded_file = st.file_uploader("Перетащите изображение сюда или кликните для выбора файла", type=['png', 'jpg', 'jpeg'])

    if uploaded_file is not None:
                image = Image.open(uploaded_file)
                st.subheader('Загруженная картинка')
                st.image(image)

    image = preprocess(image)
    prediction = model_coffe(image.unsqueeze(0)).softmax(dim=1).argmax().item()

    
    st.write('Предсказанный вид кофе: ', coffee_dict[prediction])
    

if page == "Агрокультуры":
    
    image_url = st.text_input("Введите URL картинки агрокультуры")

    if image_url:
                response = requests.get(image_url)
                image = Image.open(io.BytesIO(response.content))
                st.subheader('Загруженная картинка')
                st.image(image)
    
    uploaded_file = st.file_uploader("Перетащите изображение сюда или кликните для выбора файла", type=['png', 'jpg', 'jpeg'])

    if uploaded_file is not None:
                image = Image.open(uploaded_file)
                st.subheader('Загруженная картинка')
                st.image(image)

if page == "Магическая страница":
    image_url = st.text_input("Введите URL изображения чего угодно")

    if image_url:
                response = requests.get(image_url)
                image = Image.open(io.BytesIO(response.content))
                st.subheader('Загруженная картинка')
                st.image(image)

    uploaded_file = st.file_uploader("Перетащите изображение сюда или кликните для выбора файла", type=['png', 'jpg', 'jpeg'])

    if uploaded_file is not None:
                image = Image.open(uploaded_file)
                st.subheader('Загруженная картинка')
                st.image(image)




