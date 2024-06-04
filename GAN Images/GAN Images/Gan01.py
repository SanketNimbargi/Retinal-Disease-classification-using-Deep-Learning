import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.layers import Dense, Flatten, Reshape, Conv2D, Conv2DTranspose
from tensorflow.keras.models import Sequential

# Generate random noise for the generator
def generate_noise(n_samples, noise_dim):
    return np.random.normal(0, 1, (n_samples, noise_dim))

# Define the generator model
def build_generator(noise_dim):
    model = Sequential()
    model.add(Dense(8*8*128, input_dim=noise_dim, activation='relu'))
    model.add(Reshape((8, 8, 128)))
    model.add(Conv2DTranspose(128, (4,4), strides=(2,2), padding='same', activation='relu'))
    model.add(Conv2DTranspose(64, (4,4), strides=(2,2), padding='same', activation='relu'))
    model.add(Conv2DTranspose(1, (4,4), strides=(2,2), padding='same', activation='sigmoid'))
    return model

# Define the discriminator model
def build_discriminator(input_shape):
    model = Sequential()
    model.add(Conv2D(64, (3, 3), strides=(2, 2), padding='same', input_shape=input_shape, activation='relu'))
    model.add(Conv2D(128, (3, 3), strides=(2, 2), padding='same', activation='relu'))
    model.add(Flatten())
    model.add(Dense(1, activation='sigmoid'))
    return model

# Compile the discriminator
def compile_discriminator(discriminator):
    discriminator.compile(loss='binary_crossentropy', optimizer=tf.keras.optimizers.Adam(lr=0.0002, beta_1=0.5), metrics=['accuracy'])
    return discriminator

# Compile the combined GAN model
def compile_gan(generator, discriminator):
    discriminator.trainable = False
    gan = Sequential([generator, discriminator])
    gan.compile(loss='binary_crossentropy', optimizer=tf.keras.optimizers.Adam(lr=0.0002, beta_1=0.5))
    return gan

# Train the GAN
def train_gan(generator, discriminator, gan, X_train, epochs, batch_size, noise_dim, output_dir):
    os.makedirs(output_dir, exist_ok=True)  # Create output directory if it doesn't exist
    for epoch in range(epochs):
        # Generate fake images
        noise = generate_noise(batch_size, noise_dim)
        fake_images = generator.predict(noise)
        
        # Save generated images
        save_generated_images(fake_images, epoch, output_dir)
        
        # Select a random batch of real images
        idx = np.random.randint(0, X_train.shape[0], batch_size)
        real_images = X_train[idx]
        
        # Train the discriminator
        discriminator_loss_real = discriminator.train_on_batch(real_images, np.ones((batch_size, 1)))
        discriminator_loss_fake = discriminator.train_on_batch(fake_images, np.zeros((batch_size, 1)))
        discriminator_loss = 0.5 * np.add(discriminator_loss_real, discriminator_loss_fake)
        
        # Train the generator
        noise = generate_noise(batch_size, noise_dim)
        valid_y = np.array([1] * batch_size)
        generator_loss = gan.train_on_batch(noise, valid_y)
        
        # Print progress
        print(f"Epoch {epoch+1}/{epochs} | Discriminator Loss: {discriminator_loss[0]}, Generator Loss: {generator_loss}")

def save_generated_images(images, epoch, output_dir):
    for i, img in enumerate(images):
        filename = os.path.join(output_dir, f"generated_image_epoch_{epoch+1}_index_{i+1}.png")
        cv2.imwrite(filename, img * 255.0)  # Save images after denormalizing

def load_retinal_images(directory, target_size):
    images = []
    for filename in os.listdir(directory):
        if filename.endswith('.jpg') or filename.endswith('.png'):  # Assuming images are in JPG or PNG format
            image_path = os.path.join(directory, filename)
            image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)  # Read image in grayscale
            image = cv2.resize(image, target_size)  # Resize image to desired dimensions
            images.append(image)
    return np.array(images)

# Example usage
if __name__ == '__main__':
    # Load and preprocess data (e.g., retinal images)
    retinal_images_dir = 'C:/Users/sachi/OneDrive/Desktop/BEPROJECT/Retinal Diseases Detection/dataset/glaucoma'

    # Replace the following with your data loading/preprocessing code
    target_size = (64, 64)  # Adjust the target size as needed
    retinal_images = load_retinal_images(retinal_images_dir, target_size)
    # Ensure that retinal images are normalized and reshaped appropriately
    # Here's a placeholder loading code
   
    retinal_images = retinal_images / 255.0  # Normalize pixel values
    retinal_images = np.expand_dims(retinal_images, axis=-1)  # Add channel dimension if necessary
    
    # Set parameters
    img_shape = retinal_images[0].shape
    noise_dim = 100
    batch_size = 32
    epochs = 10
    output_dir = "C:/Users/sachi/OneDrive/Desktop/BEPROJECT/Retinal Diseases Detection/GAN Images/GAN Images/gan_generated"  # Directory to save generated images
    
    # Build and compile the models
    generator = build_generator(noise_dim)
    discriminator = build_discriminator(img_shape)
    discriminator = compile_discriminator(discriminator)
    gan = compile_gan(generator, discriminator)
    
    # Train the GAN
    train_gan(generator, discriminator, gan, retinal_images, epochs, batch_size, noise_dim, output_dir)









# import os
# import cv2
# import numpy as np
# import tensorflow as tf
# from tensorflow.keras.layers import Dense, Flatten, Reshape, LeakyReLU, BatchNormalization, Conv2D, UpSampling2D
# from tensorflow.keras.models import Sequential, Model
# from skimage.metrics import structural_similarity as ssim

# def generate_noise(n_samples, noise_dim):
#     return np.random.normal(0, 1, (n_samples, noise_dim))

# def build_generator(noise_dim, img_shape):
#     model = Sequential()
#     model.add(Dense(64 * 64 * 256, input_dim=noise_dim))  # Adjusted input dimension
#     model.add(Reshape((64, 64, 256)))  # Adjusted reshape size
#     model.add(LeakyReLU(alpha=0.2))
#     model.add(BatchNormalization())
    
#     model.add(Conv2D(256, kernel_size=3, padding='same'))
#     model.add(LeakyReLU(alpha=0.2))
#     model.add(BatchNormalization())
    
#     model.add(UpSampling2D())  # Adjusted upsampling
#     model.add(Conv2D(128, kernel_size=3, padding='same'))
#     model.add(LeakyReLU(alpha=0.2))
#     model.add(BatchNormalization())
    
#     model.add(UpSampling2D())  # Adjusted upsampling
#     model.add(Conv2D(64, kernel_size=3, padding='same'))
#     model.add(LeakyReLU(alpha=0.2))
#     model.add(BatchNormalization())
    
#     model.add(Conv2D(3, kernel_size=3, padding='same', activation='sigmoid'))  # Adjusted output shape
    
#     return model

# def build_discriminator(input_shape):
#     print("Discriminator Input Shape:", input_shape)
#     model = Sequential()
#     model.add(Conv2D(64, kernel_size=3, strides=2, padding='same', input_shape=input_shape))
#     model.add(LeakyReLU(alpha=0.2))
    
#     model.add(Conv2D(128, kernel_size=3, strides=2, padding='same'))
#     model.add(LeakyReLU(alpha=0.2))
    
#     model.add(Conv2D(256, kernel_size=3, strides=2, padding='same'))
#     model.add(LeakyReLU(alpha=0.2))
    
#     model.add(Conv2D(512, kernel_size=3, strides=2, padding='same'))
#     model.add(LeakyReLU(alpha=0.2))
    
#     model.add(Flatten())
#     model.add(Dense(1, activation='sigmoid'))
    
#     return model

# def compile_discriminator(discriminator):
#     discriminator.compile(loss='binary_crossentropy', optimizer=tf.keras.optimizers.Adam(lr=0.0002, beta_1=0.5), metrics=['accuracy'])
#     return discriminator

# def compile_gan(generator, discriminator):
#     discriminator.trainable = False
    
#     # Define input shape for the generator
#     noise_shape = generator.input_shape[1:]
#     print("Generator Input Shape:", noise_shape[0])  # Print generator input shape
    
#     # Define input shape for the discriminator
#     discriminator_input_shape = generator.output_shape[1:]
#     print("Generator Output Shape:", discriminator_input_shape)  # Print generator output shape
    
#     # Define input shape for the discriminator
#     discriminator_input_shape = generator.output_shape[1:]
#     print("Discriminator Input Shape:", discriminator_input_shape)  # Print discriminator input shape
    
#     # Define GAN input
#     gan_input = tf.keras.Input(shape=noise_shape)
    
#     # Connect the generator and discriminator
#     generated_image = generator(gan_input)
#     gan_output = discriminator(generated_image)
    
#     # Create the GAN model
#     gan = tf.keras.Model(inputs=gan_input, outputs=gan_output)
#     gan.compile(loss='binary_crossentropy', optimizer=tf.keras.optimizers.Adam(lr=0.0002, beta_1=0.5))
    
#     return gan

# def train_gan(generator, discriminator, gan, X_train, epochs, batch_size, noise_dim, output_dir, window_size):
#     os.makedirs(output_dir, exist_ok=True)
#     for epoch in range(epochs):
#         noise = generate_noise(batch_size, noise_dim)
#         fake_images = generator.predict(noise)
#         if (epoch + 1) % 75 == 0:
#             save_generated_images(fake_images, epoch, output_dir)
        
#         idx = np.random.randint(0, X_train.shape[0], batch_size)
#         real_images = X_train[idx]
        
#         discriminator_loss_real = discriminator.train_on_batch(real_images, np.ones((batch_size, 1)))
#         discriminator_loss_fake = discriminator.train_on_batch(fake_images, np.zeros((batch_size, 1)))
#         discriminator_loss = 0.5 * np.add(discriminator_loss_real, discriminator_loss_fake)
        
#         noise = generate_noise(batch_size, noise_dim)
#         valid_y = np.array([1] * batch_size)
#         generator_loss = gan.train_on_batch(noise, valid_y)
        
#         # Calculate SSIM for each generated image
#         avg_ssim = 0
#         for i in range(len(fake_images)):
#          real_image = real_images[i].squeeze()
#          generated_image = fake_images[i].squeeze()
#          ssim_val = ssim(real_image, generated_image, win_size=window_size, multichannel=True)
#          avg_ssim += ssim_val
#          avg_ssim /= len(fake_images)

        
#         print(f"Epoch {epoch+1}/{epochs} | Discriminator Loss: {discriminator_loss[0]}, Generator Loss: {generator_loss}, Avg SSIM: {avg_ssim}")

# def save_generated_images(images, epoch, output_dir):
#     for i, img in enumerate(images):
#         filename = os.path.join(output_dir, f"generated_image_epoch_{epoch+1}index{i+1}.png")
#         cv2.imwrite(filename, img * 255.0)

# def load_retinal_images(directory, target_size):
#     images = []
#     for filename in os.listdir(directory):
#         if filename.endswith('.jpg') or filename.endswith('.png'):
#             image_path = os.path.join(directory, filename)
#             image = cv2.imread(image_path)  # Load as color image
#             image = cv2.resize(image, target_size)
#             images.append(image)
#     return np.array(images)

# if __name__ == '__main__':
#     retinal_images_dir = 'C:/Users/sachi/OneDrive/Desktop/BEPROJECT/Retinal Diseases Detection/dataset/glaucoma'
#     target_size = (256, 256)  # Adjusted to the specified dimensions
#     retinal_images = load_retinal_images(retinal_images_dir, target_size)
#     retinal_images = retinal_images / 255.0
#     print("\nShape of retinal_images:", retinal_images.shape)
    
#     noise_dim = 100
#     batch_size = 32
#     epochs = 10
#     output_dir = "C:/Users/sachi/OneDrive/Desktop/GAN Images/GAN Images/gan_generated"
    
#     # Build and compile the models
#     generator = build_generator(noise_dim, retinal_images[0].shape)
#     discriminator = build_discriminator(retinal_images[0].shape)
    
#     # Calculate window size for SSIM based on image dimensions
#     # window_size = min(retinal_images.shape[1:3]) // 32 * 2 + 1
#     window_size = 7

    
#     # Print shapes
#     print("Discriminator Input Shape:", retinal_images[0].shape)
#     print("Generator Input Shape:", noise_dim)
#     print("Generator Output Shape:", retinal_images[0].shape)
    
#     discriminator = compile_discriminator(discriminator)
#     gan = compile_gan(generator, discriminator)
    
#     # Train the GAN
#     train_gan(generator, discriminator, gan, retinal_images, epochs, batch_size, noise_dim, output_dir, window_size)
    
    
    
    
    
    
    
