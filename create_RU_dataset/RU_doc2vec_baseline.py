from gensim.models import Doc2Vec
import pickle
import os

from src.settings import PATH_TO_OUTPUT_FOLDER
from tqdm import tqdm

def save_data_from_doc2vec(filename, model,
                           authors_list, novels_list):
    data_csv = open(filename + '.csv', 'w', encoding='utf-8')
    print('Save file: {}'.format(filename + '.csv'))
    data_csv_string = ''
    for index, one_vector in tqdm(enumerate(model.docvecs.vectors_docs)):
        row_result = ';'.join([str(value) for value in one_vector])
        data_csv_string += row_result
        data_csv_string += ';'+authors_list[index]+';'+novels_list[index].replace(';','')+ '\n'
    data_csv.write(data_csv_string[:-1])
    data_csv.close()
    print('Save file: {} DONE'.format(filename + '.csv'))

def save_data_after_doc2vec_inference(filename, list_of_infered_vectors,
                           authors_list, novels_list):
    data_csv = open(filename + '.csv', 'w', encoding='utf-8')
    print('Save file: {}'.format(filename + '.csv'))
    data_csv_string = ''
    for index, one_vector in tqdm(enumerate(list_of_infered_vectors)):
        row_result = ';'.join([str(value) for value in one_vector])
        data_csv_string += row_result
        data_csv_string += ';'+authors_list[index]+';'+novels_list[index].replace(';','')+ '\n'
    data_csv.write(data_csv_string[:-1])
    data_csv.close()
    print('Save file: {} DONE'.format(filename + '.csv'))


TEST_AVALABLE = True
FAIR_TEST = True

COUNT_OF_SENTENCE = 350

if __name__ == "__main__":
    print('Loading processed documents...')
    PATH_TO_CURRENT_OUT_FOLDER = os.path.join(PATH_TO_OUTPUT_FOLDER, 'RUS_AA', '{} Sentences'.format(COUNT_OF_SENTENCE),
                                              #'TRAIN_6000_BOOTSTRAP_{}_SENTENCES'.format(COUNT_OF_SENTENCE))
                                              'TRAIN_FIXED_SEPARATION_{}_SENTENCES'.format(COUNT_OF_SENTENCE))
    fullDatasetDocs = pickle.load(open(os.path.join(PATH_TO_CURRENT_OUT_FOLDER, 'tagged_documents_dump.pkl'), 'rb'))
    print('Loading processed documents... DONE')

    PATH_TO_DOC2VEC_VECTORS = os.path.join(PATH_TO_CURRENT_OUT_FOLDER, 'doc2vec_vectors_TRAIN')
    print('We will save csv to: {}'.format(PATH_TO_DOC2VEC_VECTORS))

    if not os.path.exists(PATH_TO_DOC2VEC_VECTORS):
        print('Directory: {} was created'.format(PATH_TO_DOC2VEC_VECTORS))
        os.makedirs(PATH_TO_DOC2VEC_VECTORS)

    if TEST_AVALABLE:
        PATH_TO_CURRENT_OUT_TEST = os.path.join(PATH_TO_OUTPUT_FOLDER, 'RUS_AA', '{} Sentences'.format(COUNT_OF_SENTENCE),
                                                'TEST_FIXED_SEPARATION_{}_SENTENCES'.format(COUNT_OF_SENTENCE))
        fullTestDatasetDocs = pickle.load(open(os.path.join(PATH_TO_CURRENT_OUT_TEST, 'tagged_documents_dump.pkl'),
                                               'rb'))
        print('Loading test processed documents... DONE')
        #PATH_TO_DOC2VEC_INFER_VECTORS = os.path.join(PATH_TO_CURRENT_OUT_TEST, 'doc2vec_vectors_TEST_INFER_BOOTSTRAP')
        PATH_TO_DOC2VEC_INFER_VECTORS = os.path.join(PATH_TO_CURRENT_OUT_TEST, 'doc2vec_vectors_TEST_INFER')
        if not os.path.exists(PATH_TO_DOC2VEC_INFER_VECTORS):
            print('Directory: {} was created'.format(PATH_TO_DOC2VEC_INFER_VECTORS))
            os.makedirs(PATH_TO_DOC2VEC_INFER_VECTORS)

    if FAIR_TEST:
        PATH_TO_CURRENT_OUT_FAIR_TEST = os.path.join(PATH_TO_OUTPUT_FOLDER, 'RUS_AA', '{} Sentences'.format(COUNT_OF_SENTENCE),
                                                'TEST_SAMPLE_0.1_PERCENT_{}_SENTENCES'.format(COUNT_OF_SENTENCE))
        fullFairTestDatasetDocs = pickle.load(open(os.path.join(PATH_TO_CURRENT_OUT_FAIR_TEST, 'tagged_documents_dump.pkl'),
                                               'rb'))
        print('Loading test processed documents... DONE')
        #PATH_TO_DOC2VEC_FAIR_TEST_INFER_VECTORS = os.path.join(PATH_TO_CURRENT_OUT_FAIR_TEST, 'doc2vec_vectors_FAIR_TEST_INFER_BOOTSTRAP')
        PATH_TO_DOC2VEC_FAIR_TEST_INFER_VECTORS = os.path.join(PATH_TO_CURRENT_OUT_FAIR_TEST, 'doc2vec_vectors_FAIR_TEST_INFER')
        if not os.path.exists(PATH_TO_DOC2VEC_FAIR_TEST_INFER_VECTORS):
            print('Directory: {} was created'.format(PATH_TO_DOC2VEC_FAIR_TEST_INFER_VECTORS))
            os.makedirs(PATH_TO_DOC2VEC_FAIR_TEST_INFER_VECTORS)

    search_params = {#'vector_size' : [50,100,150],
        'vector_size': [100], #'vector_size': [50, 100, 150],
        'window': [10],                     #'window' : [5,10,15],
        'min_count' : [3],#'min_count' : [1,3,5,10],
        'negative': [5]             #'negative' : [5,10]
        }

    current_params = {'min_count' : 1,
                      'negative' : 5 ,
                      'workers' : 4}



    list_of_tagged_documents = list(map(lambda x : x[2], fullDatasetDocs))
    list_of_authors = list(map(lambda x : x[0], fullDatasetDocs))
    list_of_novels = list(map(lambda x: x[1], fullDatasetDocs))

    for vector_size in search_params['vector_size']:
        for window_size in search_params['window']:
            current_params['vector_size'] = vector_size
            current_params['window'] = window_size
            model = Doc2Vec(**current_params)
            print('Model declaration: {}'.format(model))
            print('Building vocabulary for model...')
            model.build_vocab(list_of_tagged_documents)
            print('Building vocabulary for model... DONE')
            print('Training model...')
            model.train(list_of_tagged_documents, epochs=30, total_examples=len(fullDatasetDocs))
            print('Training model... DONE')
            save_data_from_doc2vec(
                os.path.join(PATH_TO_DOC2VEC_VECTORS, 'doc2vec_data_size_{}_window_{}'.format(vector_size, window_size)),
                                   model, list_of_authors, list_of_novels)
            if TEST_AVALABLE:
                print('Model inference (Test set)...')
                test_authors_list = list(map(lambda x : x[0], fullTestDatasetDocs))
                test_novels_list = list(map(lambda x: x[1], fullTestDatasetDocs))
                list_of_vectors = [model.infer_vector(one_doc) for one_doc in map(lambda x : x[2].words,
                                                                                  fullTestDatasetDocs)]
                save_data_after_doc2vec_inference(
                    os.path.join(PATH_TO_DOC2VEC_INFER_VECTORS,
                                 'doc2vec_data_size_{}_window_{}_infered'.format(vector_size, window_size)),
                    list_of_vectors, test_authors_list, test_novels_list)
                print('Model inference... DONE')

            if FAIR_TEST:
                print('Model inference (fair test set)...')
                test_authors_list = list(map(lambda x : x[0], fullFairTestDatasetDocs))
                test_novels_list = list(map(lambda x: x[1], fullFairTestDatasetDocs))
                list_of_vectors = [model.infer_vector(one_doc) for one_doc in map(lambda x : x[2].words,
                                                                                  fullFairTestDatasetDocs)]
                save_data_after_doc2vec_inference(
                    os.path.join(PATH_TO_DOC2VEC_FAIR_TEST_INFER_VECTORS,
                                 'doc2vec_data_size_{}_window_{}_infered'.format(vector_size, window_size)),
                    list_of_vectors, test_authors_list, test_novels_list)
                print('Model inference (fair test set)... DONE')
