# -*- coding: utf-8 -*-

import tkinter as tk
from tkinter import ttk, LEFT, END
from PIL import Image, ImageTk
from tkinter.filedialog import askopenfilename
from tkinter import messagebox as ms
import cv2
import sqlite3
import os
import numpy as np
import time
#import video_capture as value
#import lecture_details as detail_data
#import video_second as video1

#import lecture_video  as video

global fn
fn = ""
##############################################+=============================================================
root = tk.Tk()
root.configure(background="brown")
# root.geometry("1300x700")


w, h = root.winfo_screenwidth(), root.winfo_screenheight()
root.geometry("%dx%d+0+0" % (w, h))
root.title("Retinal Disease Classification Using Deep Learning")

# 43

# ++++++++++++++++++++++++++++++++++++++++++++
#####For background Image
image2 = Image.open('r1.jpeg')
image2 = image2.resize((1500, 710), Image.LANCZOS)

background_image = ImageTk.PhotoImage(image2)

background_label = tk.Label(root, image=background_image)

background_label.image = background_image

background_label.place(x=0, y=0)  # , relwidth=1, relheight=1)
#
canvas = tk.Canvas(root, width=900, height=50)
canvas.pack()

text = "Welcome to Retinal Disease Detection using Deep Learning"
x = canvas.create_text(0, 25, text=text, anchor="w", font=('times',25,'bold'))

def scroll():
    canvas.move(x, -1, 0)  # Move text to the left
    if canvas.bbox(x)[0] < -canvas.winfo_width():
        canvas.move(x, canvas.winfo_width(), 0)  # Reset text position
    root.after(20, scroll)  # Schedule next scroll after 50 milliseconds

scroll()

#T1.tag_configure("center", justify='center')
#T1.tag_add("center", 1.0, "end")

################################$%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#def clear_img():
#    img11 = tk.Label(root, background='bisque2')
#    img11.place(x=0, y=0)


#################################################################$%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


################################$%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

# def cap_video():
    
#     video1.upload()
#     #from subprocess import call
#     #call(['python','video_second.py'])

def reg():
    from subprocess import call
    call(["python","registration.py"])

def log():
    from subprocess import call
    call(["python","login.py"])

    
def window():
  root.destroy()


button1 = tk.Button(root, text="Login", command=log, width=12, height=1,font=('times', 20, ' bold '), bg="blue", fg="white")
button1.place(x=1130, y=0)

button2 = tk.Button(root, text="Register",command=reg,width=12, height=1,font=('times', 20, ' bold '), bg="green", fg="white")
button2.place(x=1130, y=60)

button3 = tk.Button(root, text="Exit",command=window,width=12, height=1,font=('times', 20, ' bold '), bg="red", fg="white")
button3.place(x=1130, y=120)

root.mainloop()








# import os
# import cv2
# import numpy as np
# import matplotlib.pyplot as plt
# import tensorflow as tf
# from tensorflow.keras.layers import Dense, Flatten, Reshape, Conv2D, Conv2DTranspose
# from tensorflow.keras.models import Sequential

# # Generate random noise for the generator
# def generate_noise(n_samples, noise_dim):
#     return np.random.normal(0, 1, (n_samples, noise_dim))

# # Define the generator model
# def build_generator(noise_dim):
#     model = Sequential()
#     model.add(Dense(8*8*128, input_dim=noise_dim, activation='relu'))
#     model.add(Reshape((8, 8, 128)))
#     model.add(Conv2DTranspose(128, (4,4), strides=(2,2), padding='same', activation='relu'))
#     model.add(Conv2DTranspose(64, (4,4), strides=(2,2), padding='same', activation='relu'))
#     model.add(Conv2DTranspose(1, (4,4), strides=(2,2), padding='same', activation='sigmoid'))
#     return model

# # Define the discriminator model
# def build_discriminator(input_shape):
#     model = Sequential()
#     model.add(Conv2D(64, (3, 3), strides=(2, 2), padding='same', input_shape=input_shape, activation='relu'))
#     model.add(Conv2D(128, (3, 3), strides=(2, 2), padding='same', activation='relu'))
#     model.add(Flatten())
#     model.add(Dense(1, activation='sigmoid'))
#     return model

# # Compile the discriminator
# def compile_discriminator(discriminator):
#     discriminator.compile(loss='binary_crossentropy', optimizer=tf.keras.optimizers.Adam(lr=0.0002, beta_1=0.5), metrics=['accuracy'])
#     return discriminator

# # Compile the combined GAN model
# def compile_gan(generator, discriminator):
#     discriminator.trainable = False
#     gan = Sequential([generator, discriminator])
#     gan.compile(loss='binary_crossentropy', optimizer=tf.keras.optimizers.Adam(lr=0.0002, beta_1=0.5))
#     return gan

# # Train the GAN
# def train_gan(generator, discriminator, gan, X_train, epochs, batch_size, noise_dim, output_dir):
#     os.makedirs(output_dir, exist_ok=True)  # Create output directory if it doesn't exist
#     for epoch in range(epochs):
#         # Generate fake images
#         noise = generate_noise(batch_size, noise_dim)
#         fake_images = generator.predict(noise)
        
#         # Save generated images
#         save_generated_images(fake_images, epoch, output_dir)
        
#         # Select a random batch of real images
#         idx = np.random.randint(0, X_train.shape[0], batch_size)
#         real_images = X_train[idx]
        
#         # Train the discriminator
#         discriminator_loss_real = discriminator.train_on_batch(real_images, np.ones((batch_size, 1)))
#         discriminator_loss_fake = discriminator.train_on_batch(fake_images, np.zeros((batch_size, 1)))
#         discriminator_loss = 0.5 * np.add(discriminator_loss_real, discriminator_loss_fake)
        
#         # Train the generator
#         noise = generate_noise(batch_size, noise_dim)
#         valid_y = np.array([1] * batch_size)
#         generator_loss = gan.train_on_batch(noise, valid_y)
        
#         # Print progress
#         print(f"Epoch {epoch+1}/{epochs} | Discriminator Loss: {discriminator_loss[0]}, Generator Loss: {generator_loss}")

# def save_generated_images(images, epoch, output_dir):
#     for i, img in enumerate(images):
#         filename = os.path.join(output_dir, f"generated_image_epoch_{epoch+1}_index_{i+1}.png")
#         cv2.imwrite(filename, img * 255.0)  # Save images after denormalizing

# def load_retinal_images(directory, target_size):
#     images = []
#     for filename in os.listdir(directory):
#         if filename.endswith('.jpg') or filename.endswith('.png'):  # Assuming images are in JPG or PNG format
#             image_path = os.path.join(directory, filename)
#             image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)  # Read image in grayscale
#             image = cv2.resize(image, target_size)  # Resize image to desired dimensions
#             images.append(image)
#     return np.array(images)

# # Example usage
# if __name__ == '__main__':
#     # Load and preprocess data (e.g., retinal images)
#     retinal_images_dir = 'C:/Users/sachi/OneDrive/Desktop/BEPROJECT/Retinal Diseases Detection/dataset/glaucoma'

#     # Replace the following with your data loading/preprocessing code
#     target_size = (64, 64)  # Adjust the target size as needed
#     retinal_images = load_retinal_images(retinal_images_dir, target_size)
#     # Ensure that retinal images are normalized and reshaped appropriately
#     # Here's a placeholder loading code
   
#     retinal_images = retinal_images / 255.0  # Normalize pixel values
#     retinal_images = np.expand_dims(retinal_images, axis=-1)  # Add channel dimension if necessary
    
#     # Set parameters
#     img_shape = retinal_images[0].shape
#     noise_dim = 100
#     batch_size = 32
#     epochs = 10
#     output_dir = "C:/Users/sachi/OneDrive/Desktop/GAN Images/GAN Images/gan_generated"  # Directory to save generated images
    
#     # Build and compile the models
#     generator = build_generator(noise_dim)
#     discriminator = build_discriminator(img_shape)
#     discriminator = compile_discriminator(discriminator)
#     gan = compile_gan(generator, discriminator)
    
#     # Train the GAN
#     train_gan(generator, discriminator, gan, retinal_images, epochs, batch_size, noise_dim, output_dir)