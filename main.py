#V_1 OG

# from transformers import BlipProcessor, BlipForConditionalGeneration
# from PIL import Image
# import cv2
# import os
# from gtts import gTTS
# import pygame  # Import pygame for audio playback
# import time
# st = time.time()
# # Load the model and processor
# model_name = "Salesforce/blip-image-captioning-large"
# processor = BlipProcessor.from_pretrained(model_name)
# model = BlipForConditionalGeneration.from_pretrained(model_name)

# def capture_image():
#     # Create the images directory if it doesn't exist
#     save_directory = "I:\\VS CODE\\CCPG\\images"  # Use double backslashes
#     os.makedirs(save_directory, exist_ok=True)

#     # Start the webcam
#     camera = cv2.VideoCapture(0)
#     if not camera.isOpened():
#         raise Exception("Could not open webcam")

#     ret, frame = camera.read()  # Capture a frame
#     camera.release()  # Release the webcam

#     if ret:
#         # Create the full path to save the image
#         save_path = os.path.join(save_directory, 'demo.jpg')
#         cv2.imwrite(save_path, frame)  # Save the captured image
#         print(f"Image saved to {save_path}")
#         return save_path  # Return the path of the saved image
#     else:
#         raise Exception("Failed to capture image")

# # Capture the image
# image_path = capture_image()

# # Load the captured image
# image = Image.open(image_path)
# # Preprocess the image
# inputs = processor(images=image, return_tensors="pt")

# # Generate captions
# output = model.generate(**inputs)

# # Decode the generated caption
# caption = processor.decode(output[0], skip_special_tokens=True)
# print(f"Generated Caption: {caption}")

# # Convert text to speech
# tts = gTTS(text=caption, lang='en')
# audio_file = "caption.mp3"
# tts.save(audio_file)

# # Initialize pygame mixer for audio playback
# pygame.mixer.init()
# pygame.mixer.music.load(audio_file)
# pygame.mixer.music.play()

# # Wait until the audio is done playing
# while pygame.mixer.music.get_busy():
#     continue
# print(time.time()-st)

#V_2 Indic TTS (In Production, Not Ready for Deployment)

# from transformers import BlipProcessor, BlipForConditionalGeneration
# from PIL import Image
# import cv2
# import os
# import time
# from gtts import gTTS
# import pygame
# import threading

# # Load the model and processor
# model_name = "Salesforce/blip-image-captioning-large"
# processor = BlipProcessor.from_pretrained(model_name)
# model = BlipForConditionalGeneration.from_pretrained(model_name)

# def capture_image():
#     # Create the images directory if it doesn't exist
#     save_directory = "I:\\VS CODE\\CCPG\\images"  # Use double backslashes
#     os.makedirs(save_directory, exist_ok=True)

#     # Start the webcam
#     camera = cv2.VideoCapture(0)
#     if not camera.isOpened():
#         raise Exception("Could not open webcam")

#     ret, frame = camera.read()  # Capture a frame
#     camera.release()  # Release the webcam

#     if ret:
#         # Create the full path to save the image
#         save_path = os.path.join(save_directory, 'demo.jpg')
#         cv2.imwrite(save_path, frame)  # Save the captured image
#         print(f"Image saved to {save_path}")
#         return save_path  # Return the path of the saved image
#     else:
#         raise Exception("Failed to capture image")

# def process_image(image_path):
#     # Load the captured image
#     image = Image.open(image_path)
#     # Preprocess the image
#     inputs = processor(images=image, return_tensors="pt")

#     # Generate captions
#     output = model.generate(**inputs)

#     # Decode the generated caption
#     caption = processor.decode(output[0], skip_special_tokens=True)
#     print(f"Generated Caption: {caption}")

#     # Convert text to speech
#     tts = gTTS(text=caption, lang='en')
#     audio_file = "caption.mp3"
#     tts.save(audio_file)

#     # Initialize pygame mixer for audio playback
#     pygame.mixer.init()
#     pygame.mixer.music.load(audio_file)
#     pygame.mixer.music.play()

# def main():
#     # Capture the image
#     image_path = capture_image()

#     # Start the image processing in a separate thread
#     processing_thread = threading.Thread(target=process_image, args=(image_path,))
#     processing_thread.start()

#     # Wait for the thread to complete
#     processing_thread.join()

# if __name__ == "__main__":
#     st = time.time()
#     main()
#     print("Time taken is ",time.time() - st)

#V_3 OCR Tesseract

import cv2
import pytesseract
# from transformers import pipeline
from gtts import gTTS
import os
import pygame
import time
from PIL import Image

st = time.time()
pytesseract.pytesseract.tesseract_cmd = "/usr/bin/tesseract" #For Linux
# Setup Tesseract path (if not in PATH, specify the path to tesseract executable)
# pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'  # For Windows ~ Change Path as needed

def capture_image():
    save_directory = "images"
    os.makedirs(save_directory, exist_ok=True)

    # Start the webcam
    camera = cv2.VideoCapture(0)
    if not camera.isOpened():
        raise Exception("Could not open webcam")

    ret, frame = camera.read()  # Capture a frame
    camera.release()  # Release the webcam

    if ret:
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

# Perform OCR (extract text from image)
extracted_text = pytesseract.image_to_string(image)
print(f"Extracted Text: {extracted_text}")

if not extracted_text.strip():
    print("No text detected in the image!")
    exit()

# Summarize the extracted text
# summarizer = pipeline("summarization")
# summary = summarizer(extracted_text, max_length=150, min_length=1, do_sample=False)

# Print the summary
# summary_text = summary[0]['summary_text']
# print(f"Summary: {summary_text}")

# Convert summary to speech
tts = gTTS(text=extracted_text, lang='en')
audio_file = "summary.mp3"
tts.save(audio_file)

# Initialize pygame mixer for audio playback
pygame.mixer.init()
pygame.mixer.music.load(audio_file)
pygame.mixer.music.play()

# Wait until the audio is done playing
while pygame.mixer.music.get_busy():
    continue

print(f"Execution Time: {time.time() - st:.2f} seconds")

#V_4 OCR Tesseract + Summary

# import cv2
# import pytesseract
# from transformers import pipeline
# from gtts import gTTS
# import os
# import pygame
# import time
# from PIL import Image

# st = time.time()
# pytesseract.pytesseract.tesseract_cmd = "/usr/bin/tesseract" #For Linux
# # Setup Tesseract path (if not in PATH, specify the path to tesseract executable)
# # pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'  # For Windows ~ Change Path as needed

# def capture_image():
#     save_directory = "images"
#     os.makedirs(save_directory, exist_ok=True)

#     # Start the webcam
#     camera = cv2.VideoCapture(0)
#     if not camera.isOpened():
#         raise Exception("Could not open webcam")

#     ret, frame = camera.read()  # Capture a frame
#     camera.release()  # Release the webcam

#     if ret:
#         save_path = os.path.join(save_directory, 'demo.jpg')
#         cv2.imwrite(save_path, frame)  # Save the captured image
#         print(f"Image saved to {save_path}")
#         return save_path  # Return the path of the saved image
#     else:
#         raise Exception("Failed to capture image")

# # Capture the image
# image_path = capture_image()

# # Load the captured image
# image = "/media/adithyaa/CodeRed/VS CODE/Projects/CCPG/img.jpg"

# # Perform OCR (extract text from image)
# extracted_text = pytesseract.image_to_string(image)
# print(f"Extracted Text: {extracted_text}")

# if not extracted_text.strip():
#     print("No text detected in the image!")
#     exit()

# # Summarize the extracted text
# summarizer = pipeline("summarization")
# summary = summarizer(extracted_text, max_length=150, min_length=1, do_sample=False)

# # Print the summary
# summary_text = summary[0]['summary_text']
# print(f"Summary: {summary_text}")

# # Convert summary to speech
# tts = gTTS(text=summary_text, lang='en')
# audio_file = "summary.mp3"
# tts.save(audio_file)

# # Initialize pygame mixer for audio playback
# pygame.mixer.init()
# pygame.mixer.music.load(audio_file)
# pygame.mixer.music.play()

# # Wait until the audio is done playing
# while pygame.mixer.music.get_busy():
#     continue

# print(f"Execution Time: {time.time() - st:.2f} seconds")
