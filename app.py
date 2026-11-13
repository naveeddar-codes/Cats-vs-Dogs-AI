import os
from flask import Flask, request, render_template
import numpy as np
from PIL import Image
import tflite_runtime.interpreter as tflite

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Load the lightweight TFLite model
interpreter = tflite.Interpreter(model_path="model.tflite")
interpreter.allocate_tensors()

# Get input and output details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

@app.route('/', methods=['GET', 'POST'])
def home():
    if request.method == 'POST':
        file = request.files['file']
        if file:
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(filepath)

            # Prepare image (Resize to 150x150, normalize)
            img = Image.open(filepath).convert('RGB').resize((150, 150))
            img_array = np.array(img, dtype=np.float32) / 255.0
            
            # Add batch dimension (1, 150, 150, 3)
            img_array = np.expand_dims(img_array, axis=0)

            # Make prediction using TFLite
            interpreter.set_tensor(input_details[0]['index'], img_array)
            interpreter.invoke()
            score = interpreter.get_tensor(output_details[0]['index'])[0][0]

            if score > 0.5:
                result = f"Dog (Confidence: {score*100:.2f}%)"
            else:
                result = f"Cat (Confidence: {(1-score)*100:.2f}%)"

            return render_template('index.html', prediction=result, image_path=filepath)
            
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)