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

# import cv2
# import pytesseract
# # from transformers import pipeline
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
# # image = Image.open(image_path)
# image = "/media/adithyaa/CodeRed/VS CODE/Projects/CCPG/img.png"

# # Perform OCR (extract text from image)
# extracted_text = pytesseract.image_to_string(image)
# print(f"Extracted Text: {extracted_text}")

# if not extracted_text.strip():
#     print("No text detected in the image!")
#     exit()

# # Summarize the extracted text
# # summarizer = pipeline("summarization")
# # summary = summarizer(extracted_text, max_length=150, min_length=1, do_sample=False)

# # Print the summary
# # summary_text = summary[0]['summary_text']
# # print(f"Summary: {summary_text}")

# # Convert summary to speech
# tts = gTTS(text=extracted_text, lang='en')
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

#V_5 Final

import cv2
import pytesseract
import os
import pygame
import time
import speech_recognition as sr
from gtts import gTTS
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration, pipeline
from threading import Thread

# Initialize Tesseract and Pygame for text-to-speech
pytesseract.pytesseract.tesseract_cmd = "/usr/bin/tesseract"  # For Linux (change for Windows or as suitable -  r"C:\Program Files\Tesseract-OCR\tesseract.exe")
pygame.mixer.init()

# Load BLIP model and processor for image captioning
model_name = "Salesforce/blip-image-captioning-large"
processor = BlipProcessor.from_pretrained(model_name)
model = BlipForConditionalGeneration.from_pretrained(model_name)

# Directory to save images
image_dir = "images"
os.makedirs(image_dir, exist_ok=True)

# Function to capture image from webcam
def capture_image():
    camera = cv2.VideoCapture(0)
    if not camera.isOpened():
        raise Exception("Could not open webcam")

    ret, frame = camera.read()
    camera.release()

    if ret:
        image_path = os.path.join(image_dir, "captured_image.jpg")
        cv2.imwrite(image_path, frame)
        print(f"Image captured and saved to {image_path}")
        return image_path
    else:
        raise Exception("Failed to capture image")

# Function to convert text to speech
def text_to_speech(text, filename="output.mp3"):
    tts = gTTS(text=text, lang='en')
    tts.save(filename)

    pygame.mixer.music.load(filename)
    pygame.mixer.music.play()
    while pygame.mixer.music.get_busy():
        continue

# Function to recognize voice commands using SpeechRecognition
def listen_for_command():
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        print("Listening for command...")
        recognizer.adjust_for_ambient_noise(source)
        audio = recognizer.listen(source)

    try:
        command = recognizer.recognize_google(audio)
        print(f"User said: {command}")
        return command.lower()
    except sr.UnknownValueError:
        print("Sorry, I didn't catch that. Please repeat.")
        return ""
    except sr.RequestError:
        print("Sorry, there seems to be an issue with the speech recognition service.")
        return ""

# Function to extract text using OCR (Tesseract)
def extract_text_from_image(image_path):
    image = Image.open(image_path)
    extracted_text = pytesseract.image_to_string(image)
    return extracted_text

# Function to generate image captions using BLIP
def generate_image_caption(image_path):
    image = Image.open(image_path)
    inputs = processor(images=image, return_tensors="pt")
    output = model.generate(**inputs)
    caption = processor.decode(output[0], skip_special_tokens=True)
    return caption

# Function to summarize the extracted text using transformers pipeline
def summarize_text(text):
    summarizer = pipeline("summarization")
    summary = summarizer(text, max_length=150, min_length=30, do_sample=False)
    return summary[0]['summary_text']

# Main function to run the assistant
def oculus_assistant():
    while True:
        print("Listening for 'Hey Oculus' to wake up...")
        command = listen_for_command()

        if "hey oculus" in command or "hello oculus" in command:
            text_to_speech("Hello Sir! I am Oculus. Capturing the image now.")

            # Capture the image in a separate thread
            image_path = capture_image()

            # Generate BLIP caption and extract text using OCR concurrently
            blip_thread = Thread(target=process_image, args=(image_path,))
            blip_thread.start()
            blip_thread.join()
            
        elif "goodbye oculus" in command:
            text_to_speech("Goodbye Sir!")
            break

# Function to process the captured image
def process_image(image_path):
    # Step 1: Generate caption using BLIP
    caption = generate_image_caption(image_path)
    print(f"BLIP Caption: {caption}")

    # Step 2: Extract text from image using OCR
    extracted_text = extract_text_from_image(image_path)
    print(f"Extracted Text: {extracted_text}")

    # Step 3: If text is found, ask whether to read or summarize
    if extracted_text.strip():
        print("Text detected!")
        text_to_speech(f"I found text in the image.")
        text_to_speech("Would you like me to read it or summarize it? Please say 'read' or 'summarize'.")

        command = listen_for_command()
        if "read" in command:
            text_to_speech(extracted_text)
        elif "summarize" in command or "summarise" in command:
            summary = summarize_text(extracted_text)
            text_to_speech(summary)
        else:
            text_to_speech("Sorry, I didn't understand your choice. I will just read the text.")
            text_to_speech(extracted_text)
    else:
        # If no text detected, just speak the BLIP caption
        print("No text detected.")
        text_to_speech("I couldn't find any text detected in the image but here's what I saw: " + caption)

if __name__ == "__main__":
    oculus_assistant()
