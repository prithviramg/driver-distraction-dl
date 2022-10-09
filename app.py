import streamlit as st
import numpy as np
from PIL import Image
import time
import tensorflow as tf

ActionText = {0 : 'SafeDriving', 1 : 'TextingRight', 2 : 'CellphoneTalkingRight', 3 : 'TextingLeft', 
              4 : 'CellphoneTalkingLeft', 5 : 'OperatingRadio', 6 : 'Drinking', 7 : 'ReachingBehind', 
              8 : 'SelfGrooming', 9 : 'TalkingToOthers'}
model = tf.keras.models.load_model('custom_model.h5')

st.title("Driver Distraction detection")
uploaded_file = st.file_uploader("Upload a file")
col1, col2 = st.columns([0.7,0.3])
if uploaded_file is not None:
    img = Image.open(uploaded_file)
    with col1:
        st.image(img,width=470)  
    start = time.time()
    img = img.convert("RGB")
    img = img.resize((320,240), Image.NEAREST)
    x = np.array(img, dtype=np.float)
    x -= np.mean(x, keepdims=True)
    x /= np.std(x, keepdims=True) + 1e-6
    x = np.expand_dims(x, axis = 0)
    y = model.predict(x)[0]
    end = time.time()
    y1, y2 = np.argsort(y)[::-1][:2]
    with col2:
        st.write(f"Action performed by the driver - \"{ActionText[y1]}\"")
        st.write(f"Action performed by the driver - \"{ActionText[y2]}\"")
        st.write(f"time taken to find the Action - {(end-start)*10**3:.03f}ms")





