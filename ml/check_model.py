import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model

# 1. Load MobileNetV2 pre-trained on ImageNet, but without the top layer
base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

# 2. Add custom layers for our Disease Detection (Binary: Normal vs Disease)
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(1024, activation='relu')(x)
predictions = Dense(2, activation='softmax')(x) # 2 classes: Normal, Disease

# 3. Create the final model
model = Model(inputs=base_model.input, outputs=predictions)

# 4. Compile it
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

print("\n--- Success! Model Initialized ---")
model.summary() # This prints the giant list of layers in your AI