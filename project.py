import torch
import torchvision
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision import transforms as T
from torchvision import io
import torchutils as tu
import json
import streamlit as st

def intro(): 
    pass

def beans(): 
    pass

def crops(): 
    pass

def double_class(): 
    pass





page_names_to_funcs = {
    "—": intro,
    "Coffee beans classifier": beans,
    "Crops classifier": crops,
    "Multi NN classifier": double_class

demo_name = st.sidebar.selectbox("Choose a demo", page_names_to_funcs.keys())
page_names_to_funcs[demo_name]()
