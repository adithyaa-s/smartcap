from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image
import cv2
import os
from gtts import gTTS
import pygame  # Import pygame for audio playback

# Load the model and processor
model_name = "Salesforce/blip-image-captioning-large"
processor = BlipProcessor.from_pretrained(model_name)
model = BlipForConditionalGeneration.from_pretrained(model_name)

def capture_image():
    # Create the images directory if it doesn't exist
    save_directory = "I:\\VS CODE\\CCPG\\images"  # Use double backslashes
    os.makedirs(save_directory, exist_ok=True)

    # Start the webcam
    camera = cv2.VideoCapture(0)
    if not camera.isOpened():
        raise Exception("Could not open webcam")

    ret, frame = camera.read()  # Capture a frame
    camera.release()  # Release the webcam

    if ret:
        # Create the full path to save the image
        save_path = os.path.join(save_directory, 'demo.jpg')
        cv2.imwrite(save_path, frame)  # Save the captured image
        print(f"Image saved to {save_path}")
        return save_path  # Return the path of the saved image
    else:
        raise Exception("Failed to capture image")

# Capture the image
image_path = capture_image()

# Load the captured image
image = Image.open(image_path)
# Preprocess the image
inputs = processor(images=image, return_tensors="pt")

# Generate captions
output = model.generate(**inputs)

# Decode the generated caption
caption = processor.decode(output[0], skip_special_tokens=True)
print(f"Generated Caption: {caption}")

# Convert text to speech
tts = gTTS(text=caption, lang='en')
audio_file = "caption.mp3"
tts.save(audio_file)

# Initialize pygame mixer for audio playback
pygame.mixer.init()
pygame.mixer.music.load(audio_file)
pygame.mixer.music.play()

# Wait until the audio is done playing
while pygame.mixer.music.get_busy():
    continue