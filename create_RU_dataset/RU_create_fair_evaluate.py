import glob
import pickle
import string

from gensim.models.doc2vec import TaggedDocument
from sklearn.feature_extraction import DictVectorizer
import pandas as pd
import os

from src.SyntaxVectorizerRU import SyntaxVectorizerRU
from src.utils import load_pickled_file
from collections import defaultdict, Counter
import numpy as np
from numpy.random import choice
import random
from collections import OrderedDict

import spacy_udpipe
from src.settings import PATH_TO_RUS_UDPIPE_MODEL
from src.settings import PATH_TO_OUTPUT_FOLDER



targetStructure = defaultdict(lambda : defaultdict(list))

path_to_preprocessed_data_with_mask = './/..//Processed//RUS_AA//Test_Processed//*.pkl'
COUNT_OF_SENTENCES = 150
PERCENTAGE = 0.1
print('Processing : {}'.format(path_to_preprocessed_data_with_mask))

current_nlp_module = spacy_udpipe.load_from_path('ru-syntagrus', PATH_TO_RUS_UDPIPE_MODEL)
syntVectorizer = SyntaxVectorizerRU(current_nlp_module)



PATH_TO_CURRENT_OUT_FOLDER = os.path.join(PATH_TO_OUTPUT_FOLDER, 'TEST_SAMPLE_{}_PERCENT_{}_SENTENCES'.format(PERCENTAGE,
                                                                                                              COUNT_OF_SENTENCES))
if not os.path.exists(PATH_TO_CURRENT_OUT_FOLDER):
    print('Directory: {} was created'.format(PATH_TO_CURRENT_OUT_FOLDER))
    os.makedirs(PATH_TO_CURRENT_OUT_FOLDER)

initDataset = []
fullDatasetDoc2Vec = []

for fname in glob.glob(path_to_preprocessed_data_with_mask):
    print('Processing: {}'.format(fname))
    targetDictionary = load_pickled_file(fname)

    author = targetDictionary['Author']
    novel_name = targetDictionary['Novel']
    list_of_trees = targetDictionary['Trees']
    targetStructure[author][novel_name] = list_of_trees
    print('For {} and {} we have {} sentences. So, we can choose {} blocks'.format(
        author, novel_name, len(list_of_trees), max([1, len(list_of_trees) - int(0.75*COUNT_OF_SENTENCES)]) ))
    if (len(list_of_trees) - int(0.75*COUNT_OF_SENTENCES)) > 3:
        count_of_sentence_for_sampling = 1+int(PERCENTAGE*(len(list_of_trees) - int(0.75*COUNT_OF_SENTENCES)))
    else:
        count_of_sentence_for_sampling = 2
    print('* For {} {} we will sampling {} blocks'.format(author, novel_name, count_of_sentence_for_sampling))
    for i in range(count_of_sentence_for_sampling):
        try:
            start_position = random.randrange(0, len(list_of_trees) - int(0.75 * COUNT_OF_SENTENCES))
            if start_position + COUNT_OF_SENTENCES < len(list_of_trees):
                one_block_sentences = list_of_trees[start_position:start_position+COUNT_OF_SENTENCES]
            else:
                one_block_sentences = list_of_trees[start_position:]
        except ValueError:
            start_position = 0
            one_block_sentences = list_of_trees
        syntVectorizer.convert_to_attributes(one_block_sentences)
        initDataset.append((author, novel_name, syntVectorizer.get_res_attributes(),
                            start_position, len(one_block_sentences) ))

        words_in_block = [str(parsed_sentence).translate(str.maketrans('', '', string.punctuation)).lower().split()
                          for parsed_sentence in one_block_sentences]
        current_document = []
        for list_of_words_in_sentences in words_in_block:
            current_document.extend(list_of_words_in_sentences)

        class_name = '{}_{}_{}'.format(author, novel_name, i)
        fullDatasetDoc2Vec.append((author, novel_name, TaggedDocument(current_document, [class_name])))

print('Save doc2vec preparation:')
pickle.dump(fullDatasetDoc2Vec, open(os.path.join(PATH_TO_CURRENT_OUT_FOLDER, 'tagged_documents_dump.pkl'), 'wb'))
print('doc2vec representation was saved')

representationTypes = initDataset[0][2].keys()
representationPrepared = defaultdict(list)
answer = list()
novel_names = list()
position_list = list()
sent_length_list = list()
for one_row in initDataset:
    answer.append(one_row[0])
    novel_names.append(one_row[1])
    for representationType in one_row[2]:
        representationPrepared[representationType].append(one_row[2][representationType])
    position_list.append(one_row[3])
    sent_length_list.append(one_row[4])

bootstrap_positions_df = pd.DataFrame(list(zip(answer, novel_names, position_list, sent_length_list)),
                                      columns=['Author', 'Novel', 'Position', 'Length'])
bootstrap_positions_df.to_csv(os.path.join(PATH_TO_CURRENT_OUT_FOLDER, 'POSITIONS_LIST.csv'), index=False)

for representationType in representationPrepared:
    currentTypeVectorizer = DictVectorizer(sparse=False)
    X = currentTypeVectorizer.fit_transform(representationPrepared[representationType])
    df = pd.DataFrame(X, columns=currentTypeVectorizer.get_feature_names())
    resFilepath = os.path.join(PATH_TO_CURRENT_OUT_FOLDER, '{}.csv'.format(representationType))
    df['Author'] = answer
    df['Novel'] = novel_names
    df.to_csv(resFilepath, index=False )
    print('Representation: {} was saved to {}'.format(representationType, resFilepath))
print('Finish')

