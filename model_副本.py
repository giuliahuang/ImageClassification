import os
import tensorflow as tf

class model:
    def __init__(self, path):
        self.model = tf.keras.models.load_model(os.path.join(path, 'SubmissionModel'))

    def predict(self, X):

        # Insert your preprocessing here

        assert X.ndim == 4

        # X = X/255.
        X = tf.keras.applications.efficientnet_v2.preprocess_input(X)

        predicts = []

        fs = [tf.image.flip_left_right, tf.image.flip_up_down, tf.image.transpose]
        for f in fs:
            data = f(X)
            pred = self.model.predict(data)
            predicts.append(pred)

        prediction = self.model.predict(X)

        for j in range(0, len(predicts)):
            prediction += predicts[j]
        prediction = prediction / (1 + len(predicts))

        output = tf.argmax(prediction, axis=-1)

        return output
