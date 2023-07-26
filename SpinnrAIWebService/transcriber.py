import torch
import whisper
import os
import csv

# MAKE SURE YOU"VE RUN pip install openai-whisper

# Specify the model you want to use
print("Setting up Whisper model...")
model_name = 'tiny'  # Replace with the actual model name
device = "cuda" if torch.cuda.is_available() else "cpu"

# Load the model
model = whisper.load_model(model_name).to(device)
print("Whisper model loaded.")

def get_transcription(audio_path, language=None):
    global model
    # MAKE SURE YOU HAVE FFMPEG OR ELSE THIS WILL KILL YOU
    # audio_path = "C:\\Users\\appii\\Google Drive\\Projects\\whisper-ui\\data\\media\\Ryan-Reynolds--Vasectomy\\audio.mp4"
    print(f"Checking if file exists: {audio_path}")
    print(f"File exists: {os.path.isfile(audio_path)}")

    # Specify the language
    language = "en" if model_name.endswith(".en") else None

    # Transcribe the audio file
    result = model.transcribe(audio_path, language=language, temperature=0.0)

    # The transcription result is a dictionary. Print the transcribed text:
    return result['text']

def moderate_text(text, blocked_words_file='SpinnrAIWebService/static/data/blocked-words.csv'):
    print("current working directory - ", os.getcwd())
    with open(blocked_words_file, 'r') as file:
        reader = csv.reader(file)
        blocked_words = set(row[0].lower().replace(",","") for row in reader)

    punct_to_remove = ',?!#().'
    words = text.lower()
    for punct in punct_to_remove:
        words = words.replace(punct, '')
    words = words.split()
    flagged_words = [word for word in words if word in blocked_words]

    if flagged_words:
        print("FLAGGED WORDS: ",flagged_words)
        return True  # Text contains blocked words
    else:
        return False  # Text is clean

# print(moderate_text('where the heck is my ass?'))

# url = "https://spinnrdev.sfo3.digitaloceanspaces.com/test6350/test6350_profile_16778231102625587539030912281657.mp4"
# print(get_transcription(url))

