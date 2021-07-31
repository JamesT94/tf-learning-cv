import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image, ImageOps


def image_to_array(image):
    im_arr = np.frombuffer(image.tobytes(), dtype=np.uint8)
    im_arr = im_arr.reshape((image.size[1], image.size[0], 3))

    return im_arr


def predict_image(image, model):
    # image should be a 150, 150, 3 array
    # model is a hdf5 model stored in the Models folder

    image = np.resize(image, (150, 150, 3))
    image = np.expand_dims(image, axis=0)
    result = model.predict(image)

    return result


@st.cache(allow_output_mutation=True)
def load_model(name):
    model = tf.keras.models.load_model(name)
    return model


model = load_model('Models/cat_dog_classifier.hdf5')

file = st.file_uploader('Please upload your image', type=['jpg', 'png'])

if file is None:
    st.text('Please upload an image file')
else:
    image = Image.open(file)
    image_array = image_to_array(image)
    st.image(image_array, use_column_width=True)
    prediction = predict_image(image, model)
    if prediction > 0:
        st.write("It's a dog")
    else:
        st.write("It's a cat")

#
# st.write("""
# #TEST TITLE
#
# Test smaller words
#
# """)
#

#
# st.write(file)
#
#
# def import_and_predict(image_data, model_data):
#     img = ImageOps.fit(image_data, (150, 150), Image.ANTIALIAS)
#     img = np.asarray(img)
#     st.write(img)
#
#     img = np.expand_dims(img, axis=0)
#     img = np.vstack([img])
#
#     result = model_data.predict(img)
#
#     return result
#
#
# if file is None:
#     st.text('Please upload an image file')
# else:
#     image = Image.open(file)
#     st.image(image, use_column_width=True)
#     prediction = import_and_predict(image, model)
#     if prediction > 0:
#         st.write("It's a dog")
#     else:
#         st.write("It's a cat")
