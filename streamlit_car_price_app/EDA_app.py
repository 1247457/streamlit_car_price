import streamlit as st
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import h5py
import tensorflow.keras
# from tensorflow import keras
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.callbacks import CSVLogger
import pickle


def run_eda_app() :
    st.subheader('EDA 화면입니다.')

    car_df = pd.read_csv('data/Car_Purchasing_Data.csv', encoding='ISO-8859-1')
    radio_menu = ['데이터프레임', '통계치']
    selected_radio = st.radio('선택하세요', radio_menu)

    if selected_radio == '데이터프레임':
        st.dataframe(car_df)
    elif selected_radio =='통계치':
        st.dataframe(car_df.describe())


    st.dataframe(car_df)
    st.dataframe(car_df.describe())

    columns = car_df.columns
    columns = list(columns)

    multi=st.multiselect('컬럼을 선택해주세요', columns)
    st.dataframe(car_df[multi])

    #상관계수를 확인.
    #멀티셀렉트에 선택
    #해당컬럼에대한 상관계수
    #단,숫자데이터에 대한 상관관계
    # st.dataframe(car_df)


    # print(car_df.dtypes != object)
    corr_columns = car_df.columns[ car_df.dtypes != object ]
    selected_corr = st.multiselect('상관계수 컬럼 선택', corr_columns)

    if len(selected_corr) > 0 :
        st.dataframe(car_df[selected_corr].corr())
        # 위의 선택한 컬럼들을 이용해 씨본의 페어플롯을 그린다.
        fig = sns.pairplot( data = car_df[selected_corr] )
        st.pyplot(fig)

    else :
        st.write('선택한 컬럼이 없습니다.')

    
    # 컬럼을 1개만 선택하면 해당 컬럼의 max와 min의 
    # 해당하는 사람의 데이터를 화면에 보여주는 기능
    col_min_max = car_df.columns[ car_df.dtypes != object ]

    picked_col = st.selectbox('최대 최소를 보고싶은 컬럼 선택', col_min_max)
    min_columns = car_df[ car_df[picked_col] == car_df[picked_col].min() ]    
    max_columns = car_df[ car_df[picked_col] == car_df[picked_col].max() ]

    st.write(picked_col," Min Value")
    st.dataframe(min_columns[ ['Customer Name',picked_col] ])

    st.write(picked_col," Max Value")
    st.dataframe(max_columns[ ['Customer Name',picked_col] ])


    # 고객이름을 검색할 수 있는 기능 개발

    word = st.text_input('이름을 입력하세요')
    
    result = car_df.loc[ car_df['Customer Name'].str.contains(word, case=False),]
    st.dataframe(result)