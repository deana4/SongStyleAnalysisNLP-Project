import csv
import html

import requests
import torch
from transformers import pipeline
import os

# Set environment variable
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "C:\\PythonProjects\\NLP_STYLISH_HEBREW_SOMGS\\creds.json"

from google.cloud import translate_v2 as translate


class Translator:
    def __init__(self, sourceLanguage='he', targetLanguage='en'):
        self.sourceLan = sourceLanguage
        self.targetLan = targetLanguage
        self.model = "Helsinki-NLP/opus-mt-tc-big-he-en"
        self.maxWords = 130  # Max characters per request for MyMemory API

        self.translationClient = translate.Client()

    def translateText(self, text, name=""):
        print(f'Translating song: {name}')
        translated_chunks = []
        words = text.split()
        current_chunk = ''
        for word in words:
            if len(current_chunk) + len(word) + 1 <= self.maxWords:
                current_chunk += ' ' + word
            else:
                toBeAdded = self.translateAlternative(current_chunk.strip())
                translated_chunks.append(toBeAdded)
                current_chunk = word

        if current_chunk:
            translated_chunks.append(self.translateAlternative(current_chunk.strip()))

        fullTranslatedText = ' '.join(translated_chunks)

        return fullTranslatedText

    # def translateAlternative(self, chunk):
    #     url = f"https://api.mymemory.translated.net/get?q={chunk}&langpair={self.sourceLan}|{self.targetLan}"
    #     response = requests.get(url).json()
    #     translated_text = response['responseData']['translatedText']
    #     return translated_text
    def translateAlternative(self, text):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        pipe = pipeline("translation", model=self.model, device=device)
        return pipe(text)[0]['translation_text']

    def translateTextWithGoogle(self, text, targetLan="en", currentSong = ""):
        print(f'Started translating song: {currentSong}')
        translation = self.translationClient.translate(
            text,
            target_language=targetLan
        )
        return html.unescape(translation['translatedText'])
