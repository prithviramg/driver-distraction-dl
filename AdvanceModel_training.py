import tensorflow as tf
import numpy as np

ActionText = {"c0" : 'SafeDriving', "c1" : 'TextingRight', "c2" : 'CellphoneTalkingRight', 
              "c3" : 'TextingLeft', "c4" : 'CellphoneTalkingLeft', "c5" : 'OperatingRadio',
              "c6" : 'Drinking', "c7" : 'ReachingBehind', "c8" : 'SelfGrooming', "c9" : 'TalkingToOthers'}

dataGenerator = tf.keras.preprocessing.image.ImageDataGenerator(
    samplewise_center=True,
    samplewise_std_normalization=True,
    fill_mode='nearest',
    validation_split=0.25
)
trainGenerator = dataGenerator.flow_from_directory(
    directory=r"C:\Users\uie54988\Downloads\imgs\train",
    target_size=(240, 320),
    color_mode='rgb',
    class_mode='categorical',
    batch_size=32,
    shuffle=True,
    subset="training"
)
valGenerator = dataGenerator.flow_from_directory(
    directory=r"C:\Users\uie54988\Downloads\imgs\train",
    target_size=(240, 320),
    color_mode='rgb',
    class_mode='categorical',
    batch_size=32,
    shuffle=True,
    subset="validation"
)

custom_model = tf.keras.Sequential([
    tf.keras.layers.InputLayer(input_shape=(240,320,3)),
    tf.keras.layers.Conv2D(4,kernel_size=(3,3),activation='relu'),
    tf.keras.layers.Conv2D(8,kernel_size=(3,3)),
    tf.keras.layers.MaxPool2D(pool_size=(3,3)),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Activation("relu"),
    tf.keras.layers.Conv2D(16,kernel_size=(3,3)),
    tf.keras.layers.MaxPool2D(pool_size=(3,3)),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Activation("relu"),
    tf.keras.layers.Conv2D(32,kernel_size=(3,3)),
    tf.keras.layers.MaxPool2D(pool_size=(3,3)),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Activation("relu"),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(256, activation = 'relu'),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(128, activation = 'relu'),
    tf.keras.layers.Dense(10, activation = 'softmax')
])
custom_model.compile(optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001),
                    loss = tf.keras.losses.CategoricalCrossentropy(),
                    metrics = "accuracy")
custom_model.summary()
custom_model_history = custom_model.fit(trainGenerator, validation_data = valGenerator, epochs = 10)

custom_model.save('custom_model.h5')