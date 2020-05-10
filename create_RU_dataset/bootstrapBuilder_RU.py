import glob
import pickle
import string

from gensim.models.doc2vec import TaggedDocument
from sklearn.feature_extraction import DictVectorizer
import pandas as pd
import os


from src.utils import load_pickled_file
from collections import defaultdict, Counter
import numpy as np
from numpy.random import choice
import random
from collections import OrderedDict

import spacy_udpipe
from src.settings import PATH_TO_RUS_UDPIPE_MODEL
from src.settings import PATH_TO_OUTPUT_FOLDER

from src.SyntaxVectorizerRU import SyntaxVectorizerRU



targetStructure = defaultdict(lambda : defaultdict(list))

path_to_preprocessed_data_with_mask = './/..//Processed//RUS_AA//Train_Processed//*.pkl'
sample_size = 6000
count_of_sentences = 150

print('Processing : {}'.format(path_to_preprocessed_data_with_mask))

for fname in glob.glob(path_to_preprocessed_data_with_mask):
    print('Processing: {}'.format(fname))
    targetDictionary = load_pickled_file(fname)

    author = targetDictionary['Author']
    novel_name = targetDictionary['Novel']
    list_of_trees = targetDictionary['Trees']
    targetStructure[author][novel_name] = list_of_trees



print('Short stat:')
print('Count Authors: {}'.format(len(targetStructure)))

statsDict = OrderedDict()
#defaultdict(lambda: Counter(0))

T = 0
for one_author in targetStructure.keys():
    print('Author: {} have {} novels'.format(one_author, len(targetStructure[one_author])))
    S_A = 0
    for one_novel in targetStructure[one_author]:
        S_i = len(targetStructure[one_author][one_novel])
        print('\tNovel: {} have {} sentences'.format(one_novel, S_i))
        S_A += S_i

        statsDict[one_author] = statsDict.get(one_author, OrderedDict())
        statsDict[one_author][one_novel] = S_i
    print('\tTotal sentences for author: {}'.format(S_A))
    T += S_A

print('Total sentences in corpora: {}'.format(T))


PATH_TO_CURRENT_OUT_FOLDER = os.path.join(PATH_TO_OUTPUT_FOLDER,
                                          'TRAIN_{}_BOOTSTRAP_{}_SENTENCES'.format(sample_size, count_of_sentences))
if not os.path.exists(PATH_TO_CURRENT_OUT_FOLDER):
    print('Directory: {} was created'.format(PATH_TO_CURRENT_OUT_FOLDER))
    os.makedirs(PATH_TO_CURRENT_OUT_FOLDER)

authors_list = list(statsDict.keys())
authors_sample = choice(authors_list, sample_size,replace=True)

print(Counter(authors_sample))
print('Who are absense: {}'.format(set(authors_list) - set(Counter(authors_sample).keys())))
initDataset = []

author_sample_counter = Counter(authors_sample)

current_nlp_module = spacy_udpipe.load_from_path('ru-syntagrus', PATH_TO_RUS_UDPIPE_MODEL)
syntVectorizer = SyntaxVectorizerRU(current_nlp_module)
fullDatasetDoc2Vec = []



# Forming which sentence use
for one_author in author_sample_counter:
    # Which novel use
    total_sentences_of_author = sum([statsDict[one_author][one_novel] for one_novel in statsDict[one_author]])
    probability_of_novels = [statsDict[one_author][one_novel]/total_sentences_of_author for one_novel in statsDict[one_author]]
    interesting_novels = list(statsDict[one_author].keys())

    target_novels = choice(interesting_novels,author_sample_counter[one_author], replace=True, p=probability_of_novels)
    for current_novel in target_novels:
        try:
            start_position = random.randrange(0, statsDict[one_author][current_novel] - int(0.75*count_of_sentences))
            if start_position + count_of_sentences < statsDict[one_author][current_novel]:
                one_block_sentences = targetStructure[one_author][current_novel][start_position:start_position+count_of_sentences]
            else:
                one_block_sentences = targetStructure[one_author][current_novel][start_position:]
        except ValueError:
            one_block_sentences = targetStructure[one_author][current_novel]
            start_position = 0
        syntVectorizer.convert_to_attributes(one_block_sentences)
        initDataset.append((one_author, current_novel, syntVectorizer.get_res_attributes(), start_position,
                            len(one_block_sentences)))


        words_in_block = [str(parsed_sentence).translate(str.maketrans('', '', string.punctuation)).lower().split()
                          for parsed_sentence in one_block_sentences]

        current_document = []
        for list_of_words_in_sentences in words_in_block:
            current_document.extend(list_of_words_in_sentences)
        class_name = '{}_{}_{}'.format(one_author, current_novel, len(fullDatasetDoc2Vec))
        fullDatasetDoc2Vec.append((one_author, current_novel, TaggedDocument(current_document, [class_name])))


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

print('Save doc2vec preparation:')
pickle.dump(fullDatasetDoc2Vec, open(os.path.join(PATH_TO_CURRENT_OUT_FOLDER, 'tagged_documents_dump.pkl'), 'wb'))
print('doc2vec representation was saved')


for representationType in representationPrepared:
    currentTypeVectorizer = DictVectorizer(sparse=False)
    X = currentTypeVectorizer.fit_transform(representationPrepared[representationType])
    df = pd.DataFrame(X, columns=currentTypeVectorizer.get_feature_names())
    resFilepath = os.path.join(PATH_TO_CURRENT_OUT_FOLDER, '{}.csv'.format(representationType))
    df['Author'] = answer
    df['Novel'] = novel_names
    df.to_csv(resFilepath, index=False )
print('Finish')
