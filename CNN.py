import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, BatchNormalization, Add
from tensorflow.keras.optimizers import Adam
import tensorflow as tf

# Function to apply gamma correction
def apply_gamma_correction(image, gamma):
    # Apply gamma correction
    corrected_image = np.power(image, gamma)
    # Normalize pixel values
    corrected_image = corrected_image / np.max(corrected_image)
    return corrected_image

# Function to apply unsharp mask filter
def unsharp_mask(image, sigma=1.0, strength=1.3):
    blurred = cv2.GaussianBlur(image, (0, 0), sigma)
    sharpened = cv2.addWeighted(image, 1.0 + strength, blurred, -strength, 0)
    return sharpened

# Function to adjust saturation
def adjust_saturation(image, factor):
    # Convert image to HSV color space
    hsv_image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    # Modify saturation channel
    hsv_image[:, :, 1] = np.clip(hsv_image[:, :, 1] * factor, 0, 255)
    # Convert back to RGB
    adjusted_image = cv2.cvtColor(hsv_image, cv2.COLOR_HSV2RGB)
    return adjusted_image

def calculate_psnr(img1, img2):
    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        return float('inf')
    PIXEL_MAX = 1.0  # Assuming your images are normalized to [0,1]
    psnr = 20 * np.log10(PIXEL_MAX / np.sqrt(mse))
    return psnr

# Function to adjust exposure
def adjust_exposure(image, exposure):
    # Convert image to LAB color space
    lab_image = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
    # Split LAB channels
    l_channel, a_channel, b_channel = cv2.split(lab_image)
    # Apply exposure adjustment to L channel
    l_channel_adjusted = cv2.add(l_channel, exposure)
    # Clamp values to [0, 255]
    l_channel_adjusted = np.clip(l_channel_adjusted, 0, 255)
    # Merge LAB channels back
    adjusted_lab_image = cv2.merge([l_channel_adjusted, a_channel, b_channel])
    # Convert back to RGB
    adjusted_image = cv2.cvtColor(adjusted_lab_image, cv2.COLOR_LAB2RGB)
    return adjusted_image

# Function to load and preprocess images
def load_images(hazy_folder, clean_folder):
    hazy_images = []
    clean_images = []

    for filename in os.listdir(hazy_folder):
        hazy_path = os.path.join(hazy_folder, filename)
        clean_path = os.path.join(clean_folder, filename)

        if os.path.isfile(hazy_path) and os.path.isfile(clean_path):
            hazy_img = cv2.imread(hazy_path)
            clean_img = cv2.imread(clean_path)

            hazy_img = cv2.resize(hazy_img, (256, 256)) / 255.0
            clean_img = cv2.resize(clean_img, (256, 256)) / 255.0

            hazy_images.append(hazy_img)
            clean_images.append(clean_img)

    hazy_images = np.array(hazy_images)
    clean_images = np.array(clean_images)

    return hazy_images, clean_images

# Load data
hazy_folder = "/Users/saijayaamruth/PycharmProjects/BDAEndsem4/DLSIPendsem/Images/hazy_test"
clean_folder = "/Users/saijayaamruth/PycharmProjects/BDAEndsem4/DLSIPendsem/Images/clean_test"
hazy_images, clean_images = load_images(hazy_folder, clean_folder)

# Split data into train and test sets
train_hazy, test_hazy, train_clean, test_clean = train_test_split(hazy_images, clean_images, test_size=0.2, random_state=40)

# Define the CNN model
def build_cnn(input_shape):
    inputs = Input(shape=input_shape)
    conv1 = Conv2D(64, 3, activation='relu', padding='same')(inputs)
    conv1 = BatchNormalization()(conv1)
    conv2 = Conv2D(64, 3, activation='relu', padding='same')(conv1)
    conv2 = BatchNormalization()(conv2)
    conv3 = Conv2D(64, 3, activation='relu', padding='same')(conv2)
    conv3 = BatchNormalization()(conv3)
    conv4 = Conv2D(64, 3, activation='relu', padding='same')(conv3)
    conv4 = BatchNormalization()(conv4)
    conv5 = Conv2D(64, 3, activation='relu', padding='same')(conv4)
    conv5 = BatchNormalization()(conv5)
    conv6 = Conv2D(64, 3, activation='relu', padding='same')(conv5)
    conv6 = BatchNormalization()(conv6)
    conv7 = Conv2D(3, 3, activation='tanh', padding='same')(conv6)
    outputs = Add()([inputs, conv7])
    model = Model(inputs=inputs, outputs=outputs)
    return model

# Build and compile the model
model = build_cnn(input_shape=(256, 256, 3))
model.compile(optimizer=Adam(lr=1e-4), loss='mse')

# Train the model
history = model.fit(train_hazy, train_clean, validation_data=(test_hazy, test_clean), epochs=10, batch_size=40)

# Evaluate the model using Mean Squared Error (MSE)
loss = model.evaluate(test_hazy, test_clean)
print("Test Loss (MSE):", loss)

# Calculate SSIM for the test dataset
ssim_scores = []
for i in range(len(test_hazy)):
    predicted = model.predict(np.expand_dims(test_hazy[i], axis=0))
    # Convert the tensors to double data type
    clean_tensor = tf.cast(tf.convert_to_tensor(test_clean[i:i+1]), dtype=tf.float64)
    predicted_tensor = tf.cast(tf.convert_to_tensor(predicted), dtype=tf.float64)
    # Calculate SSIM
    ssim_score = tf.image.ssim(clean_tensor, predicted_tensor, max_val=1.0)
    ssim_scores.append(ssim_score)

average_ssim = np.mean(ssim_scores)
print(f"Average SSIM: {average_ssim}")

# Choose a sample index from the test set
sample_index = 1

# Select the sample hazy and clean images from the test set
sample_hazy = test_hazy[sample_index]
sample_clean = test_clean[sample_index]

# Use the model to enhance the sample hazy image
enhanced_image = model.predict(np.expand_dims(sample_hazy, axis=0)).squeeze()

# Apply gamma correction to the enhanced image
gamma = 1.5
enhanced_image_gamma_corrected = apply_gamma_correction(enhanced_image, gamma)

# Apply unsharp mask filter to the enhanced image
sharpened_image = unsharp_mask(enhanced_image_gamma_corrected)

# Adjust saturation of the enhanced image with post-processing
saturation_factor = 0.95
enhanced_image_with_saturation = adjust_saturation(sharpened_image, saturation_factor)

# Adjust exposure of the enhanced image with saturation
exposure_adjustment = 1.6
enhanced_image_with_saturation_and_exposure = adjust_exposure(enhanced_image_with_saturation, exposure_adjustment)

# Calculate PSNR between the original hazy image and the enhanced image
psnr_value = calculate_psnr(sample_clean, enhanced_image_with_saturation_and_exposure)
print(f"PSNR between the original clean image and the enhanced image: {psnr_value:.2f} dB")

# Print training loss and validation loss
print("Training Loss:", history.history['loss'])
print("Validation Loss:", history.history['val_loss'])


plt.plot(history.history['loss'], label='Training loss')
plt.plot(history.history['val_loss'], label='Validation loss')
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(loc="upper left")
plt.show()

# Plot the original hazy image, the original clean image, and the enhanced image with all post-processing steps including saturation and exposure adjustments
plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
plt.imshow(sample_hazy)
plt.title('Original Hazy Image')

plt.subplot(1, 3, 2)
plt.imshow(sample_clean)
plt.title('Original Clean Image')

plt.subplot(1, 3, 3)
plt.imshow(enhanced_image_with_saturation_and_exposure)
plt.title('Enhanced Image')

plt.show()
