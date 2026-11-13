import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# 1. Set up the image data
IMG_SIZE = (150, 150) # We resize all images to 150x150 pixels
BATCH_SIZE = 32

# 2. Prepare the data
datagen = ImageDataGenerator(
    rescale=1./255,      # Normalize pixel values (0-255) to (0-1)
    validation_split=0.2 # Keep 20% of data for testing
)

train_data = datagen.flow_from_directory(
    '.', # Look in the current folder
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='binary', # Cat vs Dog (Two classes)
    subset='training'
)

val_data = datagen.flow_from_directory(
    '.',
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='binary',
    subset='validation'
)

# 3. Build the Neural Network (Transfer Learning)
base_model = tf.keras.applications.MobileNetV2(
    input_shape=(150, 150, 3),
    include_top=False,
    weights='imagenet'
)
base_model.trainable = False # Freeze the pre-trained brain

model = models.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dense(1, activation='sigmoid') # 1 output (Cat=0, Dog=1)
])

# 4. Compile and Train
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.fit(train_data, validation_data=val_data, epochs=5)

# 5. Save the brain
model.save('model.h5')
print("Model saved successfully!")