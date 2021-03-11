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
import joblib

def run_ml_app() :
    st.subheader('머신 러닝 화면입니다.')

    # 1. 유저에게 입력을 받는다. 
    gender = st.radio('성별을 선택하세요',['남자','여자'])
    if gender == '남자':
        gender =1
    else :
        gender =0

    age = st.number_input('나이 입력', min_value=0, max_value=120)

    salary = st.number_input('연봉입력', min_value=0)

    debt = st.number_input('빚 입력', min_value=0)

    worth = st.number_input('자산입력', min_value=0)


    # 2. 예측한다.

    # 2-1 모델 가져오기
    # data폴더 안에 아까 저장한 모델 파일 있지요?
    model = tf.keras.models.load_model('data/car_ai.h5')

   
    # 2-2 넘파이 어레이 제작
    new_data = np.array( [gender, age, salary, debt, worth] )

    # 2-3 x 데이터 피쳐스케일링
    new_data = new_data.reshape(1,-1)

    sc_X = joblib.load('data/sc_X.pkl')

    new_data = sc_X.transform(new_data)


    # 2-4 예측한다.
    y_pred = model.predict(new_data)

    # 2-5 예측결과는 스케일링된 결과이므로 다시 원래값으로 돌린다.
    # st.write(y_pred[0][0])
    sc_y = joblib.load('data/sc_y.pkl')

    y_pred_original = sc_y.inverse_transform(y_pred)

    # 3. 결과를 화면에 띄운다
    btn = st.button('결과보기')
    if btn :
        st.write('예측결과입니다. {:,.2f} $의 차를 살 수 있습니다.'.format(y_pred_original[0,0]))
        # st.write( y_pred_original )
    






