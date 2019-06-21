import os
import spacy
from nltk.tokenize import sent_tokenize
import pickle
import glob
import tqdm

# Path to spacy Russian language model:
PATH_TO_RUS_LANG_MODEL = "C:/Users/Oleg_Durandin/Documents/spacy-ru-master/ru2/"
#PATH_TO_RUS_LANG_MODEL = """C:/Users/Oleg/Downloads/spacy-ru-master/spacy-ru-master/ru2/"""
ru_nlp = spacy.load(PATH_TO_RUS_LANG_MODEL) # Загрузим языковую модель

def process_one_file(input_fname):
    with open(input_fname, 'r', encoding='utf-8') as f:
        raw_text = f.read()
    # Change all \n
    raw_text = raw_text.replace('\n', ' ')

    processed_file = {}
    filename = os.path.basename(input_fname)
    processed_file['Author']= filename.split('_')[0]
    processed_file['Novel'] = filename.split('_')[1][:-4]
    #processed_file['Sentences'] = []
    processed_file['Trees'] = []

    tokenized_by_sentences = sent_tokenize(raw_text)
    for one_sentence in tqdm.tqdm(tokenized_by_sentences):
        processed_sentence = one_sentence.strip('— ').strip('- ').strip('– ')
        doc = ru_nlp(processed_sentence)
        #processed_file['Sentences'].append(processed_sentence)
        processed_file['Trees'].append(doc)
    return processed_file

def preprocessing(path_to_texts, path_to_output):
    for fname in glob.glob(path_to_texts):
        print('Processing {}'.format(fname))

        filename = os.path.basename(fname)
        output_file = os.path.join(path_to_output, filename[:-4]+'.pkl')
        if not os.path.isfile(output_file):
            #print('File {} not exists'.format(output_file))
            processed_dict = process_one_file(fname)
        else:
            print('File {} is exists'.format(output_file))
            continue

        with open(output_file, 'wb') as outfile:
            pickle.dump(processed_dict, outfile)
            print('{} pickled'.format(output_file))

def load_pickled_file(one_file):
    with open(one_file, 'rb') as f:
        obj = pickle.load(f)
    return obj

if __name__ == "__main__":
    mask_with_input_texts = ".\\Texts\\*.txt"
    output_path = '.\\ProcessedTexts\\'
    preprocessing(mask_with_input_texts, output_path)
