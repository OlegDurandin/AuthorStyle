import glob
import os

import spacy
import numpy as np
import pandas as pd

from preprocessor import load_pickled_file
from preprocessor import PATH_TO_RUS_LANG_MODEL
from processor import SyntaxVectorizer

from sklearn.feature_extraction import DictVectorizer
from collections import defaultdict
from tqdm import tqdm

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
            for i in vertices[:-1]:
                res.append(list_of_dependency_trees[previous_position: i])
                previous_position = i
            res.append(list_of_dependency_trees[vertices[-2]:])
        return res

    def calc_separate_blocks(self, list_of_dependency_trees):
        sentences_number = len(list_of_dependency_trees)
        count_of_blocks = 1
        if (self.SEPARATED_PARAM == -1) or (sentences_number // self.SEPARATED_PARAM) < 2:
            count_of_blocks = 1
        else:
            count_of_blocks = len(list(range(self.SEPARATED_PARAM, sentences_number, self.SEPARATED_PARAM)))
        return count_of_blocks

PATH_TO_PREPROCESSED_TEXTS = '.\\ProcessedTexts\\*.pkl'

def traverse_all_dumped_files(path_to_preprocessed_file):
    count_of_sentences = []
    for fname in tqdm(glob.glob(path_to_preprocessed_file)):
        targetDictionary = load_pickled_file(fname)
        count_of_sentences.append(len(targetDictionary['Trees']))
    print('Avg count of sentences per book: {}'.format(np.mean(count_of_sentences)))
    print('Median count of sentences per book: {}'.format(np.median(count_of_sentences)))



if __name__ == "__main__":
    ru_nlp = spacy.load(PATH_TO_RUS_LANG_MODEL)  # Загрузим языковую модель

    # Just function to print out avg count of sentences per book: avg over 3000, median 267
    # Hmmmm...
    #traverse_all_dumped_files(PATH_TO_PREPROCESSED_TEXTS)

    separator = ProcessOneFile(350)
    syntVectorizer = SyntaxVectorizer(ru_nlp)

    initDataset = []
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
        for one_block in res:
            syntVectorizer.convert_to_attributes(one_block)
            initDataset.append((author, novel_name, syntVectorizer.get_res_attributes()))


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
        resFilepath = os.path.join('OutData', '{}.csv'.format(representationType))
        df['Author'] = answer
        df['Novel'] = novel_names
        df.to_csv(resFilepath, index=False )


    print('Finish')


