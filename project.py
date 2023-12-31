import numpy as np
import pandas as pd
import streamlit as st
import io
import imageio
import requests
from PIL import Image
import torch 
import torch.nn as nn
import json

from models.preprocess import preprocess



# --------- double --------------

from torchvision.models import vgg19, VGG19_Weights
model_1 = vgg19(weights=VGG19_Weights.DEFAULT)
model_2 = resnet50(weights=ResNet50_Weights.DEFAULT)
def double_classify(img): 
    model_1.eval()
    model_2.eval()
    pred1 = model_1(img.unsqueeze(0)).softmax(dim=1)
    pred2 = model_2(img.unsqueeze(0)).softmax(dim=1)
    pred_vote = (pred1 + pred2)/2
    sorted, indices = torch.sort(pred_vote, descending=True)
    top_5 = (sorted*100).tolist()[0][:5]
    top_5_i = indices.tolist()[0][:5]
    top_5_n = list(map(decode, top_5_i))
    return top_5_n, top_5
labels = json.load(open('models/imagenet_class_index.json'))
decode = lambda x: labels[str(x)][1]



# ----------- Streamlit --------------------------


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

    # -------- coffee -------------


    from torchvision.models import resnet50, ResNet50_Weights
    model_coffe = resnet50(weights=ResNet50_Weights.DEFAULT)
    model_coffe.fc = nn.Linear(2048,4)
    model_coffe.load_state_dict(torch.load('models/coffee_save.pt', map_location=torch.device('cpu')))
    model_coffe.eval()
    coffee_dict = {0: 'Dark', 1: 'Green', 2: 'Light', 3: 'Medium'}

    image_url = st.text_input("Введите URL картинки кофейного зерна")
    image = None

    if image_url:
                response = requests.get(image_url)
                image = Image.open(io.BytesIO(response.content))
                st.subheader('Загруженная картинка')
                st.image(image)

    uploaded_file = st.file_uploader("Перетащите картинку сюда или кликните для выбора файла", type=['png', 'jpg', 'jpeg'])

    if uploaded_file is not None:
                image = Image.open(uploaded_file)
                st.subheader('Загруженная картинка')
                st.image(image)

    
    if image is not None:
        image = preprocess(image)
        prediction = model_coffe(image.unsqueeze(0)).softmax(dim=1).argmax().item()
        st.write('Предсказанный вид кофе: ', coffee_dict[prediction])

    

if page == "Агрокультуры":
    
    # --------- agriculture ---------------

    model_agri = resnet50(weights=ResNet50_Weights.DEFAULT)
    model_agri.fc = nn.Linear(2048,30)
    model_agri.load_state_dict(torch.load('models/agriculture.pt', map_location=torch.device('cpu')))
    model_agri.eval()
    agri_dict = {0: 'almond', 1: 'banana', 2: 'cardamon', 3: 'cherry', 4: 'chilli', 5: 'clove', 6: 'coconut', 7: 'coffee-plant', 8: 'cotton', 9: 'cucumber', 10: 'fox_nut(Makhana)', 11: 'gram', 12: 'jowar', 13: 'jute', 14: 'lemon', 15: 'maize', 16: 'mustard-oil', 17: 'olive-tree', 18: 'papaya', 19: 'pearl_millet(bajra)', 20: 'pineapple', 21: 'rice', 22: 'soyabean', 23: 'sugarcane', 24: 'sunflower', 25: 'tea', 26: 'tobacco-plant', 27: 'tomato', 28: 'vigna-radiati(Mung)', 29: 'wheat'}


    
    image_url = st.text_input("Введите URL картинки агрокультуры")
    image = None

    if image_url:
                response = requests.get(image_url)
                image = Image.open(io.BytesIO(response.content))
                st.subheader('Загруженная картинка')
                st.image(image)
    
    uploaded_file = st.file_uploader("Перетащите картинку сюда или кликните для выбора файла", type=['png', 'jpg', 'jpeg'])

    if uploaded_file is not None:
                image = Image.open(uploaded_file)
                st.subheader('Загруженная картинка')
                st.image(image)
    
    if image is not None:
        image = preprocess(image)
        prediction = model_agri(image.unsqueeze(0)).softmax(dim=1).argmax().item()
        st.write('Предсказанный вид агрокультуры: ', agri_dict[prediction])

if page == "Магическая страница":
    
    from torchvision.models import resnet50, ResNet50_Weights
    from torchvision.models import vgg19, VGG19_Weights
    model_1 = vgg19(weights=VGG19_Weights.DEFAULT)
    model_2 = resnet50(weights=ResNet50_Weights.DEFAULT)
    def double_classify(img): 
        model_1.eval()
        model_2.eval()
        pred1 = model_1(img.unsqueeze(0)).softmax(dim=1)
        pred2 = model_2(img.unsqueeze(0)).softmax(dim=1)
        pred_vote = (pred1 + pred2)/2
        sorted, indices = torch.sort(pred_vote, descending=True)
        top_5 = (sorted*100).tolist()[0][:5]
        top_5_i = indices.tolist()[0][:5]
        top_5_n = list(map(decode, top_5_i))
        return top_5_n, top_5
    labels = json.load(open('models/imagenet_class_index.json'))
    decode = lambda x: labels[str(x)][1]
    
    image = None
    image_url = st.text_input("Введите URL изображения чего угодно")

    if image_url:
                response = requests.get(image_url)
                image = Image.open(io.BytesIO(response.content))
                st.subheader('Загруженная картинка')
                st.image(image)

    uploaded_file = st.file_uploader("Перетащите картинку сюда или кликните для выбора файла", type=['png', 'jpg', 'jpeg'])

    if uploaded_file is not None:
                image = Image.open(uploaded_file)
                st.subheader('Загруженная картинка')
                st.image(image)

    if image is not None:
            image = preprocess(image)
            classes_pred, prob_pred = double_classify(image)
            for i in range(5): 
                    st.write(f'С вероятностью {prob_pred[i]}% это {classes_pred[i]}')
    # chart_data = pd.DataFrame(prob_pred)



