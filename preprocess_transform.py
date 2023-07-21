import os
import cv2
import numpy as np
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt

# Define the directories
source_dir = '/mnt/c/Users/jrjen/development/apps/facegen_stylegan2/training_data/source'
preprocessed_dir = '/mnt/c/Users/jrjen/development/apps/facegen_stylegan2/training_data/preprocessed'
transformed_dir = '/mnt/c/Users/jrjen/development/apps/facegen_stylegan2/training_data/transformed'

# Load OpenCV's Haar cascade for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Define the transformations
color_jitter = transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1)
horizontal_flip = transforms.RandomHorizontalFlip(p=0.5)

#initialize image counter
counter = 0

# Go through all files in the source directory
for filename in os.listdir(source_dir):
    lower_filename = filename.lower()
    if lower_filename.endswith('.jpeg') or lower_filename.endswith('.jpg'):  # Check that the file is an image
        #increment counter
        counter += 1
        
        # Load the image
        img = Image.open(os.path.join(source_dir, filename))

        # Convert the image to a numpy array and convert it to grayscale
        img_np = np.array(img)
        gray = cv2.cvtColor(img_np, cv2.COLOR_BGR2GRAY)

        # Detect faces in the image
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        # Check that exactly one face is detected
        if len(faces) != 1:
            print(f"Expected one face but found {len(faces)} in {filename}")
        else:
            # Get the coordinates of the face
            (x, y, w, h) = faces[0]
            face_coords = (x, y, x+w, y+h)

            # Crop to the face
            img_cropped = img.crop(face_coords)

            # Resize to 1024x1024
            img_resized = img_cropped.resize((1024, 1024))

            # Save the preprocessed image
            preprocessed_filename = os.path.join(preprocessed_dir, 'preprocessed_' + filename)
            img_resized.save(preprocessed_filename)
            
            print(f"Saved preprocessed image as {preprocessed_filename}")

            # Apply the transformations
            img_transformed = color_jitter(img_resized)
            img_transformed = horizontal_flip(img_transformed)

            # Save the transformed image
            transformed_filename = os.path.join(transformed_dir, 'transformed_' + filename)
            img_transformed.save(transformed_filename)

            print(f"Saved transformed image as {transformed_filename}")

        # Display the original image, the preprocessed image, and the transformed image
        if counter % 50 == 0:
            plt.figure(figsize=(12, 4))
            plt.subplot(1, 3, 1)
            plt.imshow(img)
            plt.title('Original Image')
            plt.subplot(1, 3, 2)
            plt.imshow(img_resized)
            plt.title('Preprocessed Image')
            plt.subplot(1, 3, 3)
            plt.imshow(img_transformed)
            plt.title('Transformed Image')
            plt.show()

    elif lower_filename.endswith('.mov'):  # Check that the file is a MOV video
        # Open the video file
        cap = cv2.VideoCapture(os.path.join(source_dir, filename))

        # Check if video opened successfully
        if not cap.isOpened():
            print(f"Could not open video file {filename}")
            continue

        # Go through all frames in the video
        frame_number = 0
        while True:
            # Capture frame-by-frame
            ret, frame = cap.read()

            # If the frame was not successfully read, then we have reached the end of the video
            if not ret:
                break

            # Convert the frame to grayscale
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # Detect faces in the frame
            faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

            # Check that exactly one face is detected
            if len(faces) != 1:
                print(f"Expected one face but found {len(faces)} in frame {frame_number} of {filename}")
            else:
                # Get the coordinates of the face
                (x, y, w, h) = faces[0]
                face_coords = (x, y, x+w, y+h)

                # Convert the frame to a PIL Image
                img = Image.fromarray(frame)

                # Crop to the face
                img_cropped = img.crop(face_coords)

                # Resize to 1024x1024
                img_resized = img_cropped.resize((1024, 1024))

                # Save the preprocessed image
                preprocessed_filename = os.path.join(preprocessed_dir, f'preprocessed_{filename}_{frame_number}.jpeg')
                img_resized.save(preprocessed_filename)

                print(f"Saved preprocessed image as {preprocessed_filename}")

                # Apply the transformations
                img_transformed = color_jitter(img_resized)
                img_transformed = horizontal_flip(img_transformed)

                # Save the transformed image
                transformed_filename = os.path.join(transformed_dir, f'transformed_{filename}_{frame_number}.jpeg')
                img_transformed.save(transformed_filename)

                print(f"Saved transformed image as {transformed_filename}")

                # Display the original frame, the preprocessed frame, and the transformed frame
                if frame_number % 50 == 0:
                    plt.figure(figsize=(12, 4))
                    plt.subplot(1, 3, 1)
                    plt.imshow(img)
                    plt.title('Original Frame')
                    plt.subplot(1, 3, 2)
                    plt.imshow(img_resized)
                    plt.title('Preprocessed Frame')
                    plt.subplot(1, 3, 3)
                    plt.imshow(img_transformed)
                    plt.title('Transformed Frame')
                    plt.show()

            frame_number += 1

        # Release the video capture object
        cap.release()