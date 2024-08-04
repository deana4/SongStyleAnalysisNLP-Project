from collections import Counter

from afinn import Afinn
from transformers import pipeline
import wordfreq as wf
from nltk.corpus import wordnet
import pyphen


class Sentimentor:
    def __init__(self):
        self.goodOrBad = Afinn()

    # sentimentScore will return 1 if pos, 0 if neutral and -1 if negative and its sentiment score.
    def sentimentScore(self, listOfWords):
        res = {
            'posWords': [],
            'negWords': [],
            'neuWords': []
        }
        for word in listOfWords:
            sentimentScore = self.goodOrBad.score(word)
            if sentimentScore > 0:
                res['posWords'].append(word)
            elif sentimentScore < 0:
                res['negWords'].append(word)
            else:
                res['neuWords'].append(word)

        return len(res['posWords']), len(res['negWords']), len(res['neuWords'])

    def fullSongSentimentScore(self, text):
        return self.goodOrBad.score(text)

    def wordsRhythm(self, word1, word2):
        return word1.endswith(word2[-2:])  # Simplified rule: check if last two characters match

    def RhythmDetection(self, tokens, punctuations=[]):
        rhymes = []
        for i in range(len(tokens)):
            for j in range(i + 1, len(tokens)):
                if self.wordsRhythm(tokens[i], tokens[j]) and tokens[i] not in punctuations and tokens[i] != tokens[j]:
                    rhymes.append((tokens[i], tokens[j]))

        # print(rhymes)
        return len(rhymes)

    def uniqueness(self, words):
        totalUniqueness = 0
        count = 0
        countUniqueWords = 0
        for word in words:
            wordFreq = wf.word_frequency(word, "he")
            if wordFreq != 0:
                totalUniqueness += 1 / wordFreq
                count += 1
            if wordFreq < 1e-6:
                countUniqueWords += 1

        return round(totalUniqueness / count if count != 0 else 0, 3), countUniqueWords

    # Counting the amount of words that appears more than 4 times in the lyrics implying that
    # The lyrics are assembled of repeated words -> if the value is high, the song is more simple
    def repRate(self, words):
        word_counts = Counter(words)
        repeated_words = [word for word, count in word_counts.items() if count > 4]
        repetition_rate = (len(repeated_words) / len(words)) * 100 if len(words) > 0 else 0

        uniquenessOfRepeated, _ = self.uniqueness(repeated_words)

        return repetition_rate, uniquenessOfRepeated
