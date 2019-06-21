from gensim.models import Doc2Vec
from gensim.models.doc2vec import TaggedDocument
import pickle
import glob
import os
import string
import spacy
import numpy as np
import pandas as pd

from preprocessor import load_pickled_file
from preprocessor import PATH_TO_RUS_LANG_MODEL
from datasetBuilder import PATH_TO_PREPROCESSED_TEXTS

from processor import SyntaxVectorizer

from datasetBuilder import ProcessOneFile
from collections import defaultdict
from tqdm import tqdm


def save_data_from_doc2vec(filename, model):
    data_csv = open(filename + '.csv', 'w', encoding='utf-8')
    print('Save file: {}'.format(filename + '.csv'))
    data_csv_string = ''
    for index, one_vector in tqdm(enumerate(model.docvecs.vectors_docs)):
        row_result = ','.join([str(value) for value in one_vector])
        data_csv_string += row_result + '\n'

    data_csv.write(data_csv_string[:-1])
    data_csv.close()
    print('Save file: {} DONE'.format(filename + '.csv'))

if __name__ == "__main__":
    print('Loading processed documents...')
    fullDatasetDocs = pickle.load(open('preparation4doc2vec.pkl', 'rb'))
    print('Loading processed documents... DONE')

    search_params = {'vector_size' : [50,100,150],
                     'window' : [5,10,15],
                     'min_count' : [1,3,5,10],
                     'negative' : [5,10]}

    current_params = {'min_count' : 1,
                      'negative' : 5 ,
                      'workers' : 4}

    for vector_size in search_params['vector_size']:
        for window_size in search_params['window']:
            current_params['vector_size'] = vector_size
            current_params['window'] = window_size
            model = Doc2Vec(**current_params)
            print('Model declaration: {}'.format(model))
            print('Building vocabulary for model...')
            model.build_vocab(fullDatasetDocs)
            print('Building vocabulary for model... DONE')
            print('Training model...')
            model.train(fullDatasetDocs, epochs=20, total_examples=len(fullDatasetDocs))
            print('Training model... DONE')
            save_data_from_doc2vec('doc2vec_data_size_{}_window_{}'.format(vector_size, window_size), model)
