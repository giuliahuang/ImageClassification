import os
import tensorflow as tf
from tensorflow import keras as tfk
from tensorflow.keras import layers as tfkl
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator



class model:
    def __init__(self, path):
        tf.experimental.numpy.experimental_enable_numpy_behavior()
        self.model = tf.keras.models.load_model(os.path.join(path, 'SubmissionModel'))
        self.preprocessing = tfk.Sequential([
            tfkl.RandomFlip('horizontal', name='RandomFlip_horizontal'),
            tfkl.RandomZoom(0.2),
            tfkl.RandomTranslation(0.2,0.2),
            tfkl.RandomRotation(0.2)
        ], name = 'preprocessing')

        
    def predict(self, X):
        
        # Note: this is just an example.
        # Here the model.predict is called, followed by the argmax
        
        out = []
        
        for idx in range(X.shape[0]):
            
            x = X[idx]
            
            images_to_predict = []
            images_to_predict.append(x)
            
            for i in range(3):
                #augment_image in range [0,255]
                augment_image = self.preprocessing(x)
                images_to_predict.append(augment_image)
                
            images_to_predict = np.array(images_to_predict)
            predictions = self.model.predict(images_to_predict, verbose=0)
            mean_prediction = np.mean(predictions, axis=0)
            out.append(mean_prediction)
        
        out = tf.argmax(out, axis=-1)
            
        return out
