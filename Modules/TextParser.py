import json
import os
import shutil

import torch
from textstat import textstat
from transformers import AutoModel, AutoTokenizer


# This class utilize DictaBert model and can be expanded for many use cases.
class TextParser:
    def __init__(self, locationAddOns=""):
        self.counterLocation = f"{locationAddOns}nb_text_follower/generatedTexts.txt"
        self.txtID = int(self.fileReader(self.counterLocation))
        self.parserOutput = f'{locationAddOns}parser_outputs'
        self.history = []  # history of the last parsed txts
        # loading the lemmatizer and tokenizer

        self.tokenizer = AutoTokenizer.from_pretrained('dicta-il/dictabert-joint')
        self.model = AutoModel.from_pretrained('dicta-il/dictabert-joint', trust_remote_code=True)

        self.model.eval()

    @staticmethod
    def fileReader(filename):
        with open(filename, 'r', encoding='utf-8') as file:
            text = file.read()

        return text

    @staticmethod
    def fileWritter(filename, newValue):
        with open(filename, 'w') as file:
            file.write(str(newValue))

    def resetCounter(self):
        self.fileWritter(self.counterLocation, 0)
        self.txtID = 0

    @staticmethod
    def removeDir(path):
        try:
            shutil.rmtree(path)
            print(f"Directory from path: '{path}' removed successfully.")
        except Exception as e:
            pass
            # do nothing
            # print(f"Error: {e}")

    def tokenize(self, text):
        return self.tokenizer(text, return_tensors="pt")

    def parseLight(self, text):
        # startTime = time.time()
        prediction = self.model.predict([text], self.tokenizer, output_style='json')[
            0]  # dict of - text, tokens, root_idx, ner_entities
        # print(f'time to predict: {round(time.time() - startTime, 2)} seconds')
        self.history.append(prediction)

    def parseText(self, name, text, toJson=False):
        # startTime = time.time()
        file = None

        if toJson:
            if not os.path.exists(self.parserOutput):
                os.makedirs(self.parserOutput)
            file = open(f'{self.parserOutput}/{name}_{self.txtID}.json', 'w', encoding='utf-8')

        self.txtID += 1
        self.fileWritter(self.counterLocation, self.txtID)
        prediction = self.model.predict([text], self.tokenizer, output_style='json')[
            0]  # dict of - text, tokens, root_idx, ner_entities
        self.history.append(prediction)

        if toJson:
            for token in prediction['tokens']:
                assert file is not None
                json.dump(token, file, ensure_ascii=False, indent=4)

        # print(f'Took {round(time.time() - startTime, 2)} seconds to complete this process \n')

        return prediction

    def lemmatizeText(self):
        lemmatizedTokens = []
        lastParsedTxt = self.history[-1]
        for token in lastParsedTxt['tokens']:
            lemmatizedTokens.append(token['lex'])

        return lemmatizedTokens

    def extractPOS(self):
        listOfPOS = []
        lastParsedTxt = self.history[-1]
        for token in lastParsedTxt['tokens']:
            listOfPOS.append(token['morph']['pos'])

        return listOfPOS

    @staticmethod
    def generateNGrams(n, tokens):
        ngrams = []
        for i in range(len(tokens)):
            if i + n - 1 < len(tokens):
                ngrams.append(tuple(tokens[i: i + n]))

        return ngrams

    @staticmethod
    def readabilityMeasurement(lyrics):
        flesch_kincaid_grade = textstat.flesch_kincaid_grade(lyrics)
        gunning_fog = textstat.gunning_fog(lyrics)
        return flesch_kincaid_grade, gunning_fog

# Example use

# lem = TextParser()
# lem.resetCounter()
# lem.removeDir(lem.parserOutput)
#
# txt = 'בשנת 1948 השלים אפרים קישון את לימודיו בפיסול מתכת ובתולדות האמנות והחל לפרסם מאמרים הומוריסטיים'
# lem.parseText("one", txt, True)
# di1 = lem.lemmatizeText()
#
# txt = 'בשנת 1948, הוקמה מדינת ישראל'
# lem.parseText("two", txt, True)
# di2 = lem.lemmatizeText()
