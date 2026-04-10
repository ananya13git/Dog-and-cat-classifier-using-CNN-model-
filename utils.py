import numpy as np
from PIL import Image

IMG_SIZE = 128

def preprocess_image(image):
    image = image.convert("RGB")
    image = image.resize((IMG_SIZE, IMG_SIZE))
    image_array = np.array(image) / 255.0
    image_array = np.expand_dims(image_array, axis=0)
    return image_array

def predict(model, image):
    image_array = preprocess_image(image)
    prediction = model.predict(image_array)[0][0]
    
    confidence = prediction if prediction > 0.5 else (1 - prediction)

    return prediction, confidence