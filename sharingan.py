import cv2
from ultralytics import YOLO
import numpy as np
import os
import pygame
import time



# Initialize Pygame mixer for audio
pygame.mixer.init()

# Load the YOLO model
model = YOLO('/Users/mac/Desktop/Sharingan/best.pt')  
assets_folder = '/Users/mac/Desktop/Sharingan/assets/'
audio_folder = '/Users/mac/Desktop/Sharingan/audio/'

# Load all Sharingan image file paths from the "assets" folder
image_paths = []

for file in os.listdir(assets_folder):
    if file.endswith(('.png', '.jpg', '.jpeg')):
        img_path = os.path.join(assets_folder, file)
        image_paths.append(img_path)

image_paths.sort(key=lambda x: int(os.path.splitext(os.path.basename(x))[0]))

if not image_paths:
    print("Error: No Sharingan images found in the assets folder.")
    exit()

# Load the images in sorted order
sharingan_images = [cv2.imread(path, cv2.IMREAD_UNCHANGED) for path in image_paths]

# Open webcam for real-time video
cap = cv2.VideoCapture(0)

scaling_factor = 1
current_index = 0
audio_playing = False
last_transition_time = time.time()
transition_cooldown = 0.5  # Reduced cooldown time for more responsive transitions

# State tracking variables
eyes_were_open = True
consecutive_closed_frames = 0
CLOSED_FRAMES_THRESHOLD = 3  # Number of frames eyes need to be closed to count as a blink

# Rotation animation variables
rotation_duration = 2.4
rotation_speed = 720
is_rotating = False
rotation_start_time = 0
current_angle = 0

# Load audio files
audio_files = [f for f in os.listdir(audio_folder) if f.endswith('.mp3')]
audio_files.sort(key=lambda x: int(os.path.splitext(os.path.basename(x))[0]))

if not audio_files:
    print("Error: No audio files found in the audio folder.")
    exit()

def play_audio(audio_file):
    global audio_playing
    if not pygame.mixer.music.get_busy():
        pygame.mixer.music.load(audio_file)
        pygame.mixer.music.play(loops=0, start=0.0)
        audio_playing = True
    
def stop_audio():
    global audio_playing
    pygame.mixer.music.stop()
    audio_playing = False

def rotate_image(image, angle):
    """Rotate image around its center"""
    height, width = image.shape[:2]
    center = (width // 2, height // 2)
    rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated_image = cv2.warpAffine(image, rotation_matrix, (width, height), 
                                  flags=cv2.INTER_LINEAR, 
                                  borderMode=cv2.BORDER_TRANSPARENT)
    return rotated_image

def try_transition():
    global current_index, is_rotating, rotation_start_time, last_transition_time, audio_playing
    
    current_time = time.time()
    if current_time - last_transition_time >= transition_cooldown:
        # Stop current audio
        stop_audio()
        
        # Update to next image
        current_index = (current_index + 1) % len(sharingan_images)
        
        # Start rotation animation
        is_rotating = True
        rotation_start_time = current_time
        
        # Update transition time
        last_transition_time = current_time
        
        # Play new audio
        audio_file = os.path.join(audio_folder, audio_files[current_index])
        play_audio(audio_file)
        
        return True
    return False


while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame, imgsz=640)
    eyes_open = False  # Default to eyes closed

    for result in results[0].boxes.xyxy:  # Iterate over detected objects
        if len(result) == 6:
            # Assuming the result contains: [x_min, y_min, x_max, y_max, confidence, class_id]
            x_min, y_min, x_max, y_max, confidence, class_id = map(int, result[:6])
        elif len(result) == 4:
            # Assuming the result contains: [x_min, y_min, x_max, y_max]
            x_min, y_min, x_max, y_max = map(int, result[:4])
            confidence = 1.0  # Set a default confidence if not provided
            class_id = -1  # Use a placeholder class_id if not present

        if confidence > 0.5:  # Confidence threshold (adjust as needed)
            width, height = x_max - x_min, y_max - y_min
            eyes_open = True  # Eyes are detected, so they're open

            # Treat all detected objects as pupils (since all classes are "pupils")
            pupil_detected = True  # Override logic to detect pupil
            scaled_width = int(width * scaling_factor)
            scaled_height = int(height * scaling_factor)

            if scaled_width > 0 and scaled_height > 0:
                current_sharingan = sharingan_images[current_index].copy()


                if current_index < len(sharingan_images) - 1:
                    if is_rotating:
                        current_time = time.time()
                        elapsed_time = current_time - rotation_start_time
                        
                        if elapsed_time < rotation_duration:
                            current_angle = (rotation_speed * elapsed_time) % 360
                            current_sharingan = rotate_image(current_sharingan, current_angle)
                        else:
                            is_rotating = False
                            current_angle = 0

                # Resize and overlay logic
                sharingan_resized = cv2.resize(current_sharingan, (scaled_width, scaled_height))
                alpha_channel = sharingan_resized[:, :, 3]  # Extract the alpha channel
                reduced_alpha = (alpha_channel * 0.7).astype(np.uint8)  # Reduce transparency by 50%
                sharingan_resized[:, :, 3] = reduced_alpha  # Set the new alpha channel

                x_offset = x_min + (width - scaled_width) // 2
                y_offset = y_min + (height - scaled_height) // 2

                x_end = min(x_offset + scaled_width, frame.shape[1])
                y_end = min(y_offset + scaled_height, frame.shape[0])
                x_offset = max(x_offset, 0)
                y_offset = max(y_offset, 0)

                sharingan_cropped = sharingan_resized[
                    0 : (y_end - y_offset), 0 : (x_end - x_offset)
                ]

                alpha_s = sharingan_cropped[:, :, 3] / 255.0
                alpha_l = 1.0 - alpha_s

                for c in range(3):
                    frame[y_offset:y_end, x_offset:x_end, c] = (
                        alpha_s * sharingan_cropped[:, :, c]
                        + alpha_l * frame[y_offset:y_end, x_offset:x_end, c]
                    )

    # Blink detection logic
    if eyes_open:
        if not eyes_were_open:  # Eyes just opened
            if consecutive_closed_frames >= CLOSED_FRAMES_THRESHOLD:
                # This was a valid blink, try to transition
                try_transition()
        consecutive_closed_frames = 0
    else:  # Eyes are closed
        consecutive_closed_frames += 1
    
    eyes_were_open = eyes_open

    # Debug information (optional)
    #cv2.putText(frame, f"Image: {current_index}", (10, 30), 
                #cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    #cv2.putText(frame, f"Eyes: {'Open' if eyes_open else 'Closed'}", (10, 70), 
               # cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow("Real-time Sharingan Overlay", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
