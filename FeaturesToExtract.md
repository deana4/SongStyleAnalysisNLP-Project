## List of features we want to extract:
* word count
* unique words count
* lemmatization of lyrics
* POS lyrics - number of nouns, etc
* unique to words ratio & percentage
* number of repeated words
* categorize words of the lyrics to a collections of words such - 'war', 'love'
* for each artist -> make a simple MLE which show us the amount of songs from each 'music_style'
* high lvl words vs low lvl words on each song
* number of rhyme (חרוזים) for each song


### features that we want to consider?
* *year of publication?*
* *word2vec - find how close a word to each words in the lyrics by its semantics?*
* after understanding how to get the real structure of the song:
    1. number of verses (בתים)
    2. average sentence length
    3. number of sentences


## Next Missions:
* extract more features from the list.
* we need to understand how to split the lyrics into sentences.
* then, create a parsing tree of words dependencies -> will yield more features.
* create a model that uses that parameters in order to predict the song style and more.

## List of features extracted:
* We will fill this one together

## hypothesises we want to check:
* We will fill this one together as well
* mizarhi is false love while traditional is about wars - example
* oriental songs are poor in vocabulary compared with other songs
* songs by Yehoram Gaon are less creative (more banal) than other songs
* the style of Hebrew songs changes monotonically over the years
* Songs by Younger Performers vs. Older Performers:Hypothesis: "Songs by younger performers (born after 1980) exhibit more modern slang and informal language compared to songs by older performers."
* Impact of Sociopolitical Events:Hypothesis: "Songs written during periods of significant sociopolitical events (e.g., wars, peace treaties) show a higher frequency of related themes and vocabulary compared to songs from more stable periods."
* Gender Differences in Lyrics:Hypothesis: "Songs performed by female artists use a broader emotional vocabulary compared to songs performed by male artists."
* Evolution of Song Length and Complexity:Hypothesis: "The average length of songs and their syntactic complexity has increased over the decades.
* Influence of Western Music Styles: Hypothesis: "Songs influenced by Western music styles (e.g., rock, pop) show a higher occurrence of borrowed words and phrases compared to traditional Hebrew music styles."
* Religious vs. Secular Themes:Hypothesis: "Religious songs use a more formal and classical Hebrew vocabulary compared to secular songs, which exhibit more colloquial language."
* changes in Sentiment Over Time:Hypothesis: "The overall sentiment (positive, negative, neutral) of Hebrew song lyrics has shifted towards a more positive tone over the last 50 years."