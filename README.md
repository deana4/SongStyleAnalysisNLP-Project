# Stylistic Analysis of Hebrew Song Lyrics

## The Goal

This project aims to analyze the stylistic elements of Hebrew song lyrics. The **motivation** stems from a controversial statement made by singer Yehoram Gaon in 2021, who criticized "oriental" (Mizrahi) music for its alleged poor and ungrammatical language. This project seeks to explore stylistic differences across various groups of songs, focusing on aspects such as vocabulary richness, syntactic complexity, and thematic diversity.

## The Dataset

The project utilizes a corpus of nearly 15,000 Hebrew song lyrics, downloaded from Kaggle. For each song, additional features such as the performer's year of birth, music style, and song release year were manually associated.

Moreover, we created a module that extracts additional features to help analyze the songs. These features provide a comprehensive analysis of the song’s lyrics, covering lexical, syntactic, semantic, and sentimental aspects. We added an appendix that includes information about the features.

## Code Explanation

We divided the project into 3 parts:

1. **Features Extraction** – Mostly numerical features.
2. **Features Analyzing & Visualization** – Using the features we extracted, we tried to prove or disprove hypotheses set before the start of the project, activating ML & DL methods in addition to data visualization algorithms.
3. **Conclusions**

### Feature Extraction

We created 3 classes:

- **Class Artist**: Holds the artist properties, including a dictionary of songs where each key is a song’s name, and its value is an instance of a class Song.
- **Class Song**: Holds each song’s properties and new features extracted from the lyrics.
- **Class DataFetcher**: Uses both Artist and Song classes for new feature extraction. It utilizes modules created to focus on subsets of features like sentiment features, lexical features, etc. We used pretrained models such as DictaBERT-il, Word2Vec, GloVe, WordFreq, Afinn, and APIs such as Spotify & Google Translate to extract additional data.

### Feature Analysis & Visualizations

After extracting features and assembling the numerical dataset, we created a file called `analyzer` that includes the main class `FeatureAnalyzer`. It uses the dataset and tags it according to needs (by song styles, year of release, year of birth of the artist, and artists). This class uses methods for dimension reduction to understand the best features separating the songs and visualizes the data accordingly.

Another file named `classifier` includes the `Classifier` class, which activates different ML & DL methods for classification on our data based on current needs. We sampled different subsets of the data to validate or disprove our hypotheses.

**Important detail**: Extracting the song’s style was challenging due to a lack of APIs capable of this. We attached the artist’s main music style to each song. This resulted in some outliers for each style class. To overcome this, we batched each group into batches of different sizes.

## Hypotheses

We formulated several hypotheses to test:

1. **Creativity Measurement**: Creativity has been reduced over the years due to commercialization, technological advancements, and saturation of musical styles.
2. **Syntax Complexity**: Syntax complexity has reduced over the years.
3. **Political Events Influence**: Political and inter-country events significantly influence sentiment, with periods of conflict correlating with more negative sentiments.
4. **Vocabulary Richness**: Vocabulary richness increases over the years.
5. **Song Style Variety**: Song style variety has increased over the years due to musical genre evolution.
6. **Oriental Songs Complexity**: Oriental songs are inferior in complexity levels (syntax and lexical).
7. **Yehoram Gaon’s Creativity**: Songs by Yehoram Gaon are less creative (more banal) than other songs.
8. **Style Similarity**: Genres like ‘Folk, Pop’ and ‘Pop, Soul’ are close to each other by their syntax complexity.


For more information or to contribute to the project, please visit the [GitHub repository](https://github.com/deana4/SongStyleAnalysisNLP-Project).
