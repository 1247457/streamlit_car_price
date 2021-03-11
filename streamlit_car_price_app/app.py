import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense # 레이어
import h5py
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.callbacks import CSVLogger
import pickle
from sklearn.preprocessing import MinMaxScaler
from EDA_app import run_eda_app
from ML_app import run_ml_app
import joblib


def main():
    st.title('자동차 가격 예측')
    menu = ['home', 'EDA', 'ML']

    choice = st.sidebar.selectbox('Menu', menu)

    if choice == 'home':
        st.write('이 앱은 고객데이터와 자동차 구매에 대한 내용이다. 해당 고객의 정보를 입력하면 얼마정도의 차를 구매할수 있는지 예측한다.')
    elif choice =='EDA':
        run_eda_app()
    elif choice == 'ML':
        run_ml_app()







if __name__ == '__main__':
    main()