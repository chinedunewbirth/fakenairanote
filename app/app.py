from flask import Flask, request, jsonify
from werkzeug.utils import secure_filename
import os, numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras import layers, models
from PIL import Image

app = Flask(__name__)

#Load developed machine learning modea and parameter settings

tf.function(
    func=None,
    input_signature=None,
    autograph=True,
    jit_compile=None,
    reduce_retracing=False,
    experimental_implements=None,
    experimental_autograph_options=None,
    experimental_attributes=None,
    experimental_relax_shapes=None,
    experimental_compile=None,
    experimental_follow_type_hints=None
) # tensorflow parameter settings

model = tf.keras.models.load_model('currency_detections.h5') #load pre-trained saved model

@app.route('/upload', methods=['GET', 'POST'])
def upload():
    d = {}
    if request.method == 'POST':
        if 'image' not in request.files:
            return 'No image file provided', 400

        image = request.files['image']
        if image.filename == '':
            return 'No selected image file', 400

        # Process the image file as required
        filename = secure_filename(image.filename)
        filename, extension = os.path.splitext(filename)
        count = len('upload/') + 1 # count the numbers of files in the directory
        new_name = f'{count}{extension}' # rename image 
        image.save('uploads/'+new_name) # save the file to directory
        
        # Define the classes (labels)
        classes = ['fake', 'genuine']    
        
        new_image = Image.open('uploads/'+str(count)+extension)
        new_image = new_image.resize((128, 128))
        new_image = np.array(new_image) / 255.0
        new_image = np.expand_dims(new_image, axis=0)
        predictions = model.predict(new_image)
        predicted_label = classes[np.argmax(predictions)]
        d['value'] = predicted_label
        print('Predicted label:', d['value'])
    return jsonify(d['value'])

if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=False)
