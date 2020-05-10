from gensim.models.doc2vec import TaggedDocument
import pickle
import glob
import string
import spacy

from preprocessor import load_pickled_file
from preprocessor import PATH_TO_RUS_LANG_MODEL
from create_RU_dataset.datasetBuilder import PATH_TO_PREPROCESSED_TEXTS

from create_RU_dataset.datasetBuilder import ProcessOneFile

if __name__ == "__main__":
    ru_nlp = spacy.load(PATH_TO_RUS_LANG_MODEL)  # Загрузим языковую модель
    separator = ProcessOneFile(350)

    fullDatasetDocs = []
    vector_of_authors = []
    vector_of_novels = []
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
            # Список списков слов
            words_in_block = [str(parsed_sentence).translate(str.maketrans('', '', string.punctuation)).lower().split() for parsed_sentence in one_block]
            current_document = []
            for list_of_words_in_sentences in words_in_block:
                current_document.extend(list_of_words_in_sentences)

            class_name = '{}_{}_{}'.format(author, novel_name, index)
            fullDatasetDocs.append(TaggedDocument(current_document, [class_name]))

            vector_of_authors.append(author)
            vector_of_novels.append(novel_name)

    print('Pickling processed documents...')
    pickle.dump(fullDatasetDocs, open('preparation4doc2vec.pkl', 'wb'))
    answer_csv_string = 'Author;Novel\n'
    for index, author_name in enumerate(vector_of_authors):
        answer_row = '{};{}'.format(vector_of_authors[index], vector_of_novels[index])
        answer_csv_string += answer_row + '\n'
    answer_csv = open('classes_data.csv', 'w', encoding='utf-8')
    answer_csv.write(answer_csv_string[:-1])
    answer_csv.close()

    print('Save results... DONE')

