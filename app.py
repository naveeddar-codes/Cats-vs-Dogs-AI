import os
from flask import Flask, request, render_template, url_for
from werkzeug.utils import secure_filename
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np

# 1. Setup Flask
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True) # Create upload folder if it doesn't exist

# 2. Load the trained model
model = tf.keras.models.load_model('model.h5')

# 3. The main route
@app.route('/', methods=['GET', 'POST'])
def home():
    if request.method == 'POST':
        # Get the uploaded file
        file = request.files['file']
        if file:
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)

            # Prepare the image for the model
            img = image.load_img(filepath, target_size=(150, 150))
            img_array = image.img_to_array(img)
            img_array = np.expand_dims(img_array, axis=0) # Add a batch dimension
            img_array /= 255.0 # Normalize (same as rescale=1./255 in training)

            # Make the prediction
            prediction = model.predict(img_array)
            score = prediction[0][0]

            # Determine label (0 = Cat, 1 = Dog)
            if score > 0.5:
                result = f"Dog (Confidence: {score*100:.2f}%)"
            else:
                result = f"Cat (Confidence: {(1-score)*100:.2f}%)"

            # Send back to HTML
            return render_template('index.html', prediction=result, image_path=filepath)
            
    return render_template('index.html')

# 4. Run the app
if __name__ == '__main__':
    app.run(debug=True)