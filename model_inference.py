import tensorflow as tf
import numpy as np

ActionText = {0 : 'SafeDriving', 1 : 'TextingRight', 2 : 'CellphoneTalkingRight', 3 : 'TextingLeft', 
              4 : 'CellphoneTalkingLeft', 5 : 'OperatingRadio', 6 : 'Drinking', 7 : 'ReachingBehind', 
              8 : 'SelfGrooming', 9 : 'TalkingToOthers'}
model = tf.keras.models.load_model('custom_model.h5')

img = tf.keras.utils.load_img(
    path = r"C:\Users\uie54988\Downloads\imgs\test\img_33.jpg",
    color_mode='rgb',
    target_size=(240,320),
    interpolation='nearest'
)
x = np.array(img, dtype=np.float)
x -= np.mean(x, keepdims=True)
x /= np.std(x, keepdims=True) + 1e-6
x = np.expand_dims(x, axis = 0)
y = ActionText[np.argmax(model.predict(x)[0])]
print(y)


