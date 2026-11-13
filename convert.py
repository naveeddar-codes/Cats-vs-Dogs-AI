import tensorflow as tf

# Load your existing heavy model
model = tf.keras.models.load_model('model.h5')

# Create the converter
converter = tf.lite.TFLiteConverter.from_keras_model(model)

# This is the magic line: it compresses the model to use way less memory
converter.optimizations = [tf.lite.Optimize.DEFAULT]

# Convert and save
tflite_model = converter.convert()
with open('model.tflite', 'wb') as f:
    f.write(tflite_model)

print("Model successfully converted to model.tflite!")