import numpy as np
import cv2


def preprocess_image_for_tflite32(image, model_image_size=300):
    try:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (model_image_size, model_image_size))
        image = np.expand_dims(image, axis=0)
        image = (2.0 / 255.0) * image - 1.0
        image = image.astype('float32')
    except cv2.error as e:
        print(str(e))
    return image