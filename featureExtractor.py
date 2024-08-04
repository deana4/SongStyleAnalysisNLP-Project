import collections
import csv
from scipy.stats import entropy
import ast
import numpy as np
from tqdm import tqdm
from Modules import TextParser, WordCategorizer, Word2Vec
import pandas as pd
import re
import threading


class Artist:
    def __init__(self, name, musicStyle, yearOfBirth, artistKey):
        self.name = name
        self.songs = {}
        self.birthYear = yearOfBirth
        self.musicStyle = musicStyle
        self.artistKey = artistKey

    def addSong(self, song):
        if song.name not in self.songs:
            self.songs[song.name] = song
            return True
        return False

    def getArtistSongs(self):
        return self.songs


class Song:
    def __init__(self, name_param, artist_param, words_param, wordCount_param, uniqueWords_param, releaseYear,
                 englishLyrics):
        self.name = name_param
        self.artist = artist_param
        self.words = words_param
        self.wordCount = wordCount_param
        self.uniqueWords = uniqueWords_param
        self.setOfWords = set(words_param)
        self.releaseYear = releaseYear
        self.songInEnglish = englishLyrics
        self.translatedWords = DataFetcher.splitText(self.songInEnglish)

        self.bigrams = []
        self.trigrams = []

        # new features
        self.numberOfRepeatedWords: int = 0
        self.ratioOfTotalWordsToUnique: float = 0.0
        self.percentageOfTotalWordsToUnique: float = 0.0
        self.LemmatizedWords: list = []
        self.POSperWord: list = []

        self.sentimentScore: float = float('inf')
        self.positiveWords: int = 0
        self.negativeWords: int = 0
        self.numberOfDiffLemmas = 0
        self.numberOfDiffPOS = 0
        self.avgSetWordLength = 0
        self.avgAllWordLength = 0
        self.readabilityMeasure = 0
        self.amountOfWordsRhymes = 0
        self.ratioOfWordsToPOS = 0
        self.amountOfBiGrams = 0
        self.amountOfTriGrams = 0
        self.bigramsEntropy = -1
        self.trigramsEntropy = -1
        self.avgSimilarityMeasure: float = float('inf')

        self.numberOfUniqueRankedWords = 0
        self.avgUniquenessOfSong = 0
        # this 2 features are connected to each other by the fact that we extract the percentage of
        # repeated words that appears more than 4 times and their uniqueness level
        self.repetitionWordsPercentage = 0
        self.repetitionWordsUniqueness = 0

    def getUniqueWords(self):
        return len(self.setOfWords)

    def getRawLyricsAsSentence(self):
        rawLyrics = ""
        for word in self.words:
            rawLyrics += word + " "

        return rawLyrics.rstrip(" ")

    def toDict(self):
        return {
            'name': self.name,
            'artist': self.artist,
            'words': ' '.join(self.words),
            'translatedWords': ' '.join(self.translatedWords),
            'LemmatizedWords': ' '.join(self.LemmatizedWords),
            'POSperWord': ' '.join(self.POSperWord),
            'songInEnglish': self.songInEnglish,
            'wordCount': self.wordCount,
            'uniqueWords': self.uniqueWords,
            'releaseYear': self.releaseYear,
            'numberOfRepeatedWords': self.numberOfRepeatedWords,
            'ratioOfTotalWordsToUnique': self.ratioOfTotalWordsToUnique,
            'percentageOfTotalWordsToUnique': self.percentageOfTotalWordsToUnique,
            'DiffLemmas': self.numberOfDiffLemmas,
            'DiffPOS': self.numberOfDiffPOS,
            'numberOfBiGrams': self.amountOfBiGrams,
            'numberOfTriGrams': self.amountOfTriGrams,
            'bigramsEntropy': self.bigramsEntropy,
            'trigramsEntropy': self.trigramsEntropy,
            'sentimentScore': self.sentimentScore,
            'averageSetWordLength': self.avgSetWordLength,
            'WordsRhymes': self.amountOfWordsRhymes,
            'RatioOfPOStoWords': self.ratioOfWordsToPOS,
            # 'averageAllWordLength': self.avgAllWordLength,
            'readabilityMeasure': self.readabilityMeasure,
            'positiveWords': self.positiveWords,
            'negativeWords': self.negativeWords,
            'avgSimilarityMeasure': self.avgSimilarityMeasure,
            'NumberOfUniqueWordsby1/freq': self.numberOfUniqueRankedWords,
            'AvgUniqueness': self.avgUniquenessOfSong,
            'percentageOfRepeatedWords': self.repetitionWordsPercentage,
            'theUniquenessLvlOfTheRepeatedSongs': self.repetitionWordsUniqueness,
        }


class DataFetcher:
    def __init__(self, dataCsv):
        assert dataCsv is not None or ""
        self.dataFramed = pd.read_csv(dataCsv)
        self.ArtistObjectDict = {}

        self.txtParser = TextParser.TextParser('Modules/')
        self.txtParser.removeDir(self.txtParser.parserOutput)
        self.txtParser.resetCounter()

        self.Sentimentor = WordCategorizer.Sentimentor()
        self.W2V = Word2Vec.W2V('Modules/')
        # Fetch the data into dict
        self.fetchData()

    def fetchData(self):
        for row in self.dataFramed.itertuples():
            idx, artistName, currentSongWords, currentSongName, artistKey, url, wordCount, uniqueWords, birthYear, musicStyle, releaseYear, englishLyrics, *_ = row
            if artistKey not in self.ArtistObjectDict:
                self.ArtistObjectDict[artistKey] = Artist(artistName, musicStyle, birthYear, artistKey)

            if currentSongName not in self.ArtistObjectDict[artistKey].songs:
                song = Song(currentSongName, artistName, ast.literal_eval(currentSongWords), wordCount, uniqueWords,
                            releaseYear, englishLyrics)
                if self.ArtistObjectDict[artistKey].addSong(song) is True:
                    # Do nothing
                    pass
                else:
                    print("Song could not be added to the fecther dict")

    @staticmethod
    def splitText(text):
        # Use re.split to split the text by ., :, or space
        tokens = re.split(r'[.,: ]+', text)
        tokens = [token for token in tokens if token]
        return tokens

    @staticmethod
    def EliminateRowsWithZeros(input_csv, output_csv, column_name):
        df = pd.read_csv(input_csv, encoding='utf-8-sig')
        df = df[df[column_name] != 0]
        df.to_csv(output_csv, index=False, encoding='utf-8-sig')

        print(f"Rows with value 0 in column '{column_name}' have been deleted and saved to '{output_csv}'.")

    @staticmethod
    def calcEntropy(probs):
        total_ent = 0
        for p in probs:
            ent = p * np.log2(p)  # Calculate entropy with log base 2
            total_ent += ent

        return -1 * total_ent

    def assignLightFeatures(self):
        for artist in tqdm(self.ArtistObjectDict.values(), desc="Processing artists in light features"):
            for currentSong in artist.songs.values():
                # Assign the number of repeated words in the current song to the object
                currentSong.numberOfRepeatedWords = currentSong.wordCount - currentSong.uniqueWords

                # Assign ratio
                assert currentSong.wordCount != 0
                currentSong.ratioOfTotalWordsToUnique = (currentSong.uniqueWords / currentSong.wordCount)
                currentSong.percentageOfTotalWordsToUnique = currentSong.ratioOfTotalWordsToUnique * 100

                avgLengthSet = 0
                for word in currentSong.setOfWords:
                    avgLengthSet += len(word)

                avgLength = 0
                for word in currentSong.words:
                    avgLength += len(word)

                currentSong.avgSetWordLength = avgLengthSet / currentSong.uniqueWords
                currentSong.avgAllWordLength = avgLength / currentSong.wordCount

                currentSong.bigrams = self.txtParser.generateNGrams(2, currentSong.words)
                currentSong.trigrams = self.txtParser.generateNGrams(3, currentSong.words)

                twoGrams = collections.Counter(currentSong.bigrams)
                threeGrams = collections.Counter(currentSong.trigrams)

                currentSong.amountOfBiGrams = len(twoGrams.values())
                currentSong.amountOfTriGrams = len(threeGrams.values())

                currentSong.amountOfWordsRhymes = self.Sentimentor.RhythmDetection(currentSong.words)

                nbBigrams = len(currentSong.bigrams)
                nbTrigrams = len(currentSong.trigrams)

                biProbs = np.array(list(twoGrams.values())) / nbBigrams
                triProbs = np.array(list(threeGrams.values())) / nbTrigrams

                currentSong.bigramsEntropy = DataFetcher.calcEntropy(biProbs)
                currentSong.trigramsEntropy = DataFetcher.calcEntropy(triProbs)

    def assignLemmasAndPOS(self):
        for artist in tqdm(self.ArtistObjectDict.values(), desc="Processing artists in lemmas and pos"):
            for currentSong in tqdm(artist.songs.values(),
                                    desc=f"Processing songs for {artist.name} in Lemmas and POS"):
                # self.txtParser.parseText(currentSong.name, currentSong.getRawLyricsAsSentence(), True)
                self.txtParser.parseLight(currentSong.getRawLyricsAsSentence())
                currentSong.LemmatizedWords = self.txtParser.lemmatizeText()
                currentSong.POSperWord = self.txtParser.extractPOS()
                currentSong.numberOfDiffLemmas = len(set(currentSong.LemmatizedWords))
                currentSong.numberOfDiffPOS = len(set(currentSong.POSperWord))
                currentSong.ratioOfWordsToPOS = currentSong.numberOfDiffPOS / currentSong.wordCount
                # print(f'lemmas: {currentSong.LemmatizedWords} \nPOS: {currentSong.POSperWord}\n')

    def assignSentimentFeatures(self):
        for artist in tqdm(self.ArtistObjectDict.values(), desc="Processing artists in sentiment features"):
            for currentSong in artist.songs.values():
                # Assign sentiments

                currentSong.sentimentScore = self.Sentimentor.fullSongSentimentScore(currentSong.songInEnglish)

                currentSong.positiveWords, currentSong.negativeWords, neutralWordsCount = self.Sentimentor.sentimentScore(
                    currentSong.translatedWords)

                flesch, fog = self.txtParser.readabilityMeasurement(currentSong.songInEnglish)

                currentSong.readabilityMeasure = 0.5 * flesch + 0.5 * fog

    def assignW2V(self):
        for artist in tqdm(self.ArtistObjectDict.values(), desc="Processing artists in W2V"):
            for song in artist.songs.values():
                songSimilarity = []
                countComparisons = 0
                songWords = song.translatedWords
                # print(songWords)
                for m in range(len(songWords)):
                    for n in range(m, len(songWords)):
                        if songWords[m] == songWords[n]:
                            continue
                        else:
                            countComparisons += 1
                            songSimilarity.append(self.W2V.similarity(songWords[m], songWords[
                                n]))  # Maybe set the counter to count only the real comparisons that happened

                if countComparisons == 0:
                    song.avgSimilarityMeasure = 0
                else:
                    song.avgSimilarityMeasure = sum(songSimilarity) / countComparisons
                # print(song.avgSimilarityMeasure)

    def extractFeatures(self):
        self.assignLightFeatures()
        self.assignSentimentFeatures()
        self.assignLemmasAndPOS()
        self.extractUniqueness()
        self.assignW2V()
        print("All methods have completed")

    def extractUniqueness(self):
        for artist in tqdm(self.ArtistObjectDict.values(), desc="Processing artists in light features"):
            for currentSong in artist.songs.values():
                currentSong.avgUniquenessOfSong, currentSong.numberOfUniqueRankedWords = self.Sentimentor.uniqueness(
                    currentSong.words)
                currentSong.repetitionWordsPercentage, currentSong.repetitionWordsUniqueness = self.Sentimentor.repRate(
                    currentSong.words)

        self.saveSongsToNewCSV("newFeatures.csv")

    def saveSongsToNewCSV(self, filename):
        songs = []
        for artist in self.ArtistObjectDict.values():
            for song in artist.songs.values():
                songs.append(song)

        fieldnames = songs[0].toDict().keys()

        with open(filename, 'w', newline='', encoding='utf-8-sig') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()

            for song in songs:
                writer.writerow(song.toDict())

    # def translateSongs(self):
    #     totalArts = len(self.ArtistObjectDict.keys())
    #     artistTracker = 1
    #     for artist in self.ArtistObjectDict.values():
    #         songsTracker = 0
    #         for currentSong in artist.songs.values():
    #             songsTracker += 1
    #             print(f'artist: {artistTracker}/{totalArts} song: {songsTracker}/{len(artist.songs)}')
    #
    #             currentSong.songInEnglish = self.Translator.translateTextWithGoogle(
    #                 currentSong.getRawLyricsAsSentence(), "en",
    #                 currentSong.name)  # This API is not unlimited need to fix
    #
    #         self.saveSongsToNewCSV("AllEnglish.csv")
    #         artistTracker += 1
    #
    #     self.saveSongsToNewCSV("AllEnglishBackup.csv")


if __name__ == "__main__":
    # runner = DataFetcher(sys.argv[1]) #CMD runner version
    runner = DataFetcher('latestDataset.csv')
    print(f'Total number of artists in the system: {len(runner.ArtistObjectDict.keys())} \n')
    runner.extractFeatures()

