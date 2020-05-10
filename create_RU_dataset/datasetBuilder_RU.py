import glob
import os

import spacy_udpipe
import numpy as np
import pandas as pd
import codecs
import string
import pickle
from gensim.models.doc2vec import TaggedDocument

from src.SyntaxVectorizerRU import SyntaxVectorizerRU
from src.utils import load_pickled_file
from src.settings import PATH_TO_RUS_UDPIPE_MODEL
from src.settings import PATH_TO_OUTPUT_FOLDER


from sklearn.feature_extraction import DictVectorizer
from collections import defaultdict
from tqdm import tqdm
from random import randint

PATH_TO_PREPROCESSED_TEXTS = '.\\..\\Processed\\RUS_AA\\Test_Processed\\*.pkl'
FOLDER_NAME_FOR_OUTPUT_NAMES = 'TEST_FIXED_SEPARATION_{}_SENTENCES'
COUNT_OF_SENTENCES = 150


class ProcessOneFile:
    def __init__(self, count_of_tokens_for_separate=350, ):
        self.SEPARATED_PARAM = count_of_tokens_for_separate

    def separate_list(self, list_of_dependency_trees):
        sentences_number = len(list_of_dependency_trees)
        if (self.SEPARATED_PARAM == -1) or (sentences_number // self.SEPARATED_PARAM) < 2:
            res = [list_of_dependency_trees]
        else:
            res = []
            previous_position = 0
            vertices = list(range(self.SEPARATED_PARAM, sentences_number, self.SEPARATED_PARAM))
            if len(vertices) == 1:
                res.append(list_of_dependency_trees[: vertices[0]])
                res.append(list_of_dependency_trees[vertices[0]:])
            else:
                for i in vertices[:-1]:
                    res.append(list_of_dependency_trees[previous_position: i])
                    previous_position = i
                res.append(list_of_dependency_trees[vertices[-1]:])
        return res

    def calc_separate_blocks(self, list_of_dependency_trees):
        sentences_number = len(list_of_dependency_trees)
        if (self.SEPARATED_PARAM == -1) or (sentences_number // self.SEPARATED_PARAM) < 2:
            count_of_blocks = 1
        else:
            count_of_blocks = len(list(range(self.SEPARATED_PARAM, sentences_number, self.SEPARATED_PARAM)))
        return count_of_blocks


class CreateBootstrapSample:
    def __init__(self,
                 count_of_sentences = 300,
                 sample_size = 100):
        self.SEPARATED_PARAM = count_of_sentences
        self.SAMPLE_SIZE = sample_size
    def generate_sample(self, list_of_dependency_trees):
        sentences_number = len(list_of_dependency_trees)
        res = []
        for index in range(self.SAMPLE_SIZE):
            sentence_position = randint(0, sentences_number - self.SEPARATED_PARAM)
            res.append(list_of_dependency_trees[sentence_position: sentence_position+self.SEPARATED_PARAM])



def traverse_all_dumped_files(path_to_preprocessed_file):
    count_of_sentences = []
    for fname in tqdm(glob.glob(path_to_preprocessed_file)):
        targetDictionary = load_pickled_file(fname)
        count_of_sentences.append(len(targetDictionary['Trees']))
    print('Avg count of sentences per book: {}'.format(np.mean(count_of_sentences)))
    print('Median count of sentences per book: {}'.format(np.median(count_of_sentences)))







if __name__ == "__main__":
    current_nlp_module = spacy_udpipe.load_from_path('ru-syntagrus', PATH_TO_RUS_UDPIPE_MODEL)

    PATH_TO_CURRENT_OUT_FOLDER = os.path.join(PATH_TO_OUTPUT_FOLDER,
                                              FOLDER_NAME_FOR_OUTPUT_NAMES.format(COUNT_OF_SENTENCES))

    print('We will use the next preprocessed data: {}'.format(PATH_TO_PREPROCESSED_TEXTS))
    if not os.path.exists(PATH_TO_CURRENT_OUT_FOLDER):
        print('Directory: {} was created'.format(PATH_TO_CURRENT_OUT_FOLDER))
        os.makedirs(PATH_TO_CURRENT_OUT_FOLDER)

    separator = ProcessOneFile(COUNT_OF_SENTENCES)
    syntVectorizer = SyntaxVectorizerRU(current_nlp_module)

    initDataset = []
    fullDatasetDoc2Vec = []
    for fname in glob.glob(PATH_TO_PREPROCESSED_TEXTS):
        print('Processing: {}'.format(fname))
        targetDictionary = load_pickled_file(fname)

        author = targetDictionary['Author']
        novel_name = targetDictionary['Novel']
        list_of_trees = targetDictionary['Trees']

        res = separator.separate_list(list_of_trees)
        print('In {} we\'ve {} sentences and we could separate them into {} blocks'.format(fname,
                                                                                           len(list_of_trees),
                                                                                           len(res)))
        for index, one_block in enumerate(res):
            syntVectorizer.convert_to_attributes(one_block)
            initDataset.append((author, novel_name, syntVectorizer.get_res_attributes()))

            words_in_block = [str(parsed_sentence).replace('—', '').replace('﻿i', '').translate(str.maketrans('', '', string.punctuation)).translate(str.maketrans('', '', string.ascii_letters)).lower().split()
                              for parsed_sentence in one_block]
            current_document = []
            for list_of_words_in_sentences in words_in_block:
                current_document.extend(list_of_words_in_sentences)

            class_name = '{}_{}_{}'.format(author, novel_name, index)
            fullDatasetDoc2Vec.append((author, novel_name, TaggedDocument(current_document, [class_name])))

    print('Save doc2vec preparation:')
    pickle.dump(fullDatasetDoc2Vec, open(os.path.join(PATH_TO_CURRENT_OUT_FOLDER, 'tagged_documents_dump.pkl'), 'wb'))
    print('doc2vec representation was saved')

    representationTypes = initDataset[0][2].keys()
    representationPrepared = defaultdict(list)
    answer = list()
    novel_names = list()
    for one_row in initDataset:
        answer.append(one_row[0])
        novel_names.append(one_row[1])
        for representationType in one_row[2]:
            representationPrepared[representationType].append(one_row[2][representationType])

    for representationType in representationPrepared:
        currentTypeVectorizer = DictVectorizer(sparse=False)
        X = currentTypeVectorizer.fit_transform(representationPrepared[representationType])
        df = pd.DataFrame(X, columns=currentTypeVectorizer.get_feature_names())
        resFilepath = os.path.join(PATH_TO_CURRENT_OUT_FOLDER, '{}.csv'.format(representationType))
        df['Author'] = answer
        df['Novel'] = novel_names
        df.to_csv(resFilepath, index=False )
        print('Representation type: {} was saved to {}'.format(representationType, resFilepath))

    print('Finish')