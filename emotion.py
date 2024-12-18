import cv2
from deepface import DeepFace
import time
import os
import numpy as np
import tkinter as tk
from tkinter import Label
from PIL import Image, ImageTk
import matplotlib.pyplot as plt

# Create directories to save the emotion images and graphs if they don't exist
output_dir = "emotion_images"
graphs_dir = "emotion_graphs"

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

if not os.path.exists(graphs_dir):
    os.makedirs(graphs_dir)

# Set up Tkinter window
root = tk.Tk()
root.title("Real-Time Emotion Detection")

# Set window dimensions
window_width = 800
window_height = 600
root.geometry(f"{window_width}x{window_height}")

# Create frames for the webcam feed and captured images
frame_left = tk.Frame(root, width=window_width // 2, height=window_height, bg="white")
frame_left.grid(row=0, column=0)

frame_right = tk.Frame(root, width=window_width // 2, height=window_height, bg="black")
frame_right.grid(row=0, column=1)

# List to store captured emotion images as numpy arrays
captured_images = []
captured_emotions = []  # To store emotion labels
target_size = (125, 125)  # Desired size for all images in the collage

# Emotion count dictionary
emotion_count = {'angry': 0, 'disgust': 0, 'fear': 0, 'happy': 0, 'sad': 0, 'surprise': 0, 'neutral': 0}

# Create Label widget for webcam feed
webcam_label = Label(frame_right)
webcam_label.pack()

# Create a Label widget to display captured emotion images
captured_images_label = Label(frame_left)
captured_images_label.pack()

# Start capturing video from the default camera
cap = cv2.VideoCapture(0)

# Initialize the last processed time globally
last_processed_time = time.time()

# Function to update the webcam feed
def update_webcam_feed():
    global last_processed_time, captured_images, captured_emotions, emotion_count

    ret, frame = cap.read()

    # If frame is not captured properly, break the loop
    if not ret:
        print("Failed to grab frame")
        return

    # Convert frame to grayscale for face detection (MTCNN works well with color images too)
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Only process the frame for emotion analysis if enough time has passed to reduce lag
    if time.time() - last_processed_time > 0.2:  # Process every 0.2 seconds (5 FPS)

        # Perform emotion analysis using MTCNN for face detection and DeepFace for emotion detection
        try:
            # MTCNN is automatically used with the 'detector_backend' parameter
            result = DeepFace.analyze(frame, actions=['emotion'], enforce_detection=False, detector_backend='mtcnn')

            # Iterate over detected faces and draw rectangles around them with emotions
            for face in result:
                # Get the face region coordinates
                x, y, w, h = face['region']['x'], face['region']['y'], face['region']['w'], face['region']['h']
                emotion = face['dominant_emotion']

                # Draw rectangle around face and label with predicted emotion
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
                cv2.putText(frame, emotion, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

                # Extract the face ROI and resize it to the target size
                face_roi = frame[y:y + h, x:x + w]
                
                # Add emotion label to the face image
                cv2.putText(face_roi, emotion, (5, face_roi.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

                # Resize the face ROI
                resized_face_roi = cv2.resize(face_roi, target_size)

                # Save the resized emotion image
                emotion_image_filename = f"{output_dir}/{emotion}_{int(time.time())}.jpg"
                cv2.imwrite(emotion_image_filename, resized_face_roi)

                # Add the resized image and emotion label to the lists
                captured_images.append(resized_face_roi)
                captured_emotions.append(emotion)

                # Update the emotion count
                if emotion in emotion_count:
                    emotion_count[emotion] += 1

        except Exception as e:
            print(f"Error analyzing emotion: {e}")

        # Update the last processed time
        last_processed_time = time.time()

    # Convert frame to RGB for displaying in Tkinter
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame_image = Image.fromarray(frame_rgb)
    frame_image = ImageTk.PhotoImage(frame_image)

    # Update the webcam feed label
    webcam_label.config(image=frame_image)
    webcam_label.image = frame_image

    # Update the captured images display
    if captured_images:
        last_captured_image = captured_images[-1]
        last_captured_image_rgb = cv2.cvtColor(last_captured_image, cv2.COLOR_BGR2RGB)
        last_captured_image_pil = Image.fromarray(last_captured_image_rgb)
        last_captured_image_pil = ImageTk.PhotoImage(last_captured_image_pil)

        captured_images_label.config(image=last_captured_image_pil)
        captured_images_label.image = last_captured_image_pil

    # Continue updating the webcam feed
    root.after(1, update_webcam_feed)

# Start the webcam feed update
update_webcam_feed()

# Function to save collage when capture is stopped
def save_collage():
    global emotion_count, captured_images, captured_emotions

    if captured_images:
        # Determine the collage dimensions
        collage_width = 5  # Number of images per row
        collage_height = len(captured_images) // collage_width + (len(captured_images) % collage_width > 0)
        
        # Get the size of each captured image (all images are now resized to target size)
        img_height, img_width, _ = captured_images[0].shape

        # Create a blank canvas for the collage
        collage = np.zeros((collage_height * img_height, collage_width * img_width, 3), dtype=np.uint8)

        # Place the images on the collage canvas
        for i, img in enumerate(captured_images):
            row = i // collage_width
            col = i % collage_width
            collage[row * img_height: (row + 1) * img_height, col * img_width: (col + 1) * img_width] = img

            # Add the emotion label below the image
            emotion_label = captured_emotions[i]
            cv2.putText(collage, emotion_label, (col * img_width + 5, (row + 1) * img_height - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)

        # Save the collage image
        collage_filename = "emotion_collage_with_labelss.jpg"
        cv2.imwrite(collage_filename, collage)
        print(f"Collage saved as {collage_filename}")

        # Create graphs after saving collage
        create_graphs()

        # End the program after saving the collage and graphs
        root.quit()

# Function to create graphs (Pie chart and Bar chart)
def create_graphs():
    global emotion_count

    # Create pie chart
    fig, ax = plt.subplots(figsize=(5, 5))
    emotions = list(emotion_count.keys())
    counts = list(emotion_count.values())
    
    ax.pie(counts, labels=emotions, autopct='%1.1f%%', startangle=90, colors=plt.cm.Paired.colors)
    ax.set_title("Emotion Distribution")
    pie_chart_filename = f"{graphs_dir}/emotion_pie_chart.jpg"
    fig.savefig(pie_chart_filename)
    print(f"Pie chart saved as {pie_chart_filename}")

    # Create bar chart
    fig, ax = plt.subplots(figsize=(5, 3))
    ax.bar(emotions, counts)
    ax.set_title("Emotion Frequency")
    ax.set_xlabel("Emotions")
    ax.set_ylabel("Frequency")
    bar_chart_filename = f"{graphs_dir}/emotion_bar_chart.jpg"
    fig.savefig(bar_chart_filename)
    print(f"Bar chart saved as {bar_chart_filename}")

# Add a button to save the collage and graphs
save_button = tk.Button(root, text="Save Collage", command=save_collage, height=2, width=15)
save_button.grid(row=1, column=0, pady=10)

# Run the Tkinter event loop
root.mainloop()

# Release the capture and close all windows
cap.release()
cv2.destroyAllWindows()
