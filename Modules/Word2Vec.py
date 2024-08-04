from gensim.scripts.glove2word2vec import glove2word2vec
from gensim.models import KeyedVectors

'''
Download the GloVe pre-trained word vectors from here:
@LINK
 https://github.com/stanfordnlp/GloVe?tab=readme-ov-file#download-pre-trained-word-vectors
'''


# # Do the following only once!
#
# # Save the GloVe text file as a word2vec file for your use:
# glove2word2vec(<downloaded_text_filename>, <full_path_vector_filename>)
# # Load the file as KeyVectors:
# pre_trained_model = KeyedVectors.load_word2vec_format(<full_path_vector_filename.kv>, binary=False)
# # Save the key vectors for your use:
# pre_trained_model.save(<full_path_keyvector_filename.kv>)
#
# # Now, when handing the project, the KeyVector filename will be given as an argument. You may blacken
# # this part of the code again.

# glove2word2vec('glove.6B.300d.txt', 'vectors.kv')
# pre_trained_model = KeyedVectors.load_word2vec_format('vectors.kv', binary=False)
# pre_trained_model.save('readyFiles.kv')


class W2V:
    def __init__(self, innerPath=""):
        self.word2vec = KeyedVectors.load(f'{innerPath}readyFiles.kv', mmap='r')
        self.availableWords = self.word2vec.index_to_key

    def similarity(self, word1, word2):
        sim = 0
        try:
            sim = self.word2vec.similarity(word1, word2)
        except:
            # do nothing
            pass

        return round(sim * 100, 3)
        # if word1 not in self.availableWords or word2 not in self.availableWords:
        #     # print("dictionary doesn't contain one of the words - similarity will be 0 automatically")
        #     return float(0)
        # else:
        #     return round(self.word2vec.similarity(word1, word2) * 100, 3)
