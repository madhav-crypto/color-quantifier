# -*- coding: utf-8 -*-
"""
Created on Tue Jul 13 22:14:41 2021

@author: Madhav

"""
import streamlit as st
import matplotlib.pyplot as plt
import matplotlib.image as img
from PIL import Image as im
from sklearn.cluster import KMeans
import numpy as np

st.header('Image Quantifier')
st.write("""
### Upload the image file , smaller the file size fater the process :)

""")

col1, mid, col2 = st.beta_columns([15,1,23])
with col1:
    st.image('palm_trees.jpg', width=200,caption='Image before, file size: 172KB')
with col2:
    st.image('download.png',width=270,caption='Image after, file size : 36KB')
    
input_img = st.file_uploader('Upload your file here',type=('PNG','JPG','JPEG'),
            accept_multiple_files=False)
cluster = st.text_input('number of clusters (number of color clusters in your pic)',
                        "4")
cluster_new = int(cluster)

def process(df,f):
    
    image = img.imread(df)
    (h,w,s) = image.shape
    image_2d = image.reshape(h*w,s)
    model = KMeans(n_clusters=f)
    lebs = model.fit_predict(image_2d)
    pos = model.cluster_centers_.round(0).astype('int')
    final_image = np.reshape(pos[lebs],(h,w,s))
    x = im.fromarray((final_image * 1).astype(np.uint8))
    st.image(x)
    
if input_img:
    process(input_img,cluster_new)
    st.write('Download option will be provided in coming updates , till then take a screenshot ;)')






















