import spacy_udpipe
import os
import pickle
import glob
import tqdm

from settings import PATH_TO_RUS_UDPIPE_MODEL

current_nlp_module = spacy_udpipe.load_from_path('ru-syntagrus', PATH_TO_RUS_UDPIPE_MODEL)
current_nlp_module.max_length= 3000000

def process_one_file(input_fname : str) -> dict:
    with open(input_fname, 'r', encoding='utf-8-sig') as f:
        raw_text = f.read()
    raw_text = raw_text.replace('\n', ' ')

    processed_file = {}
    filename = os.path.basename(input_fname)
    processed_file['Author']= filename.split('_')[0]
    processed_file['Novel'] = filename.split('_')[1][:-4]
    #processed_file['Sentences'] = []
    processed_file['Trees'] = []

    current_text = current_nlp_module(raw_text)
    for i, one_sentence in tqdm.tqdm(enumerate(current_text.sents)):
        processed_file['Trees'].append(current_nlp_module(one_sentence.text))
    return processed_file

def preprocessing(path_to_texts : str, path_to_output_fname : str):
    for fname in glob.glob(path_to_texts):
        print('Processing {}'.format(fname))
        filename = os.path.basename(fname)
        output_file = os.path.join(path_to_output_fname, filename[:-4]+'.pkl')
        if not os.path.isfile(output_file):
            # print('File {} not exists'.format(output_file))
            processed_dict = process_one_file(fname)
        else:
            print('* File {} is exists'.format(output_file))
            continue

        with open(output_file, 'wb') as outfile:
            pickle.dump(processed_dict, outfile)
            print('* {} pickled'.format(output_file))

INPUT_FOLDER = '.\\Input\\RUS_AA\\Test\\'
PROCESSED_FOLDER = '.\\Processed\\RUS_AA\\Test_Processed\\'
MASK_FOR_TEXT = INPUT_FOLDER + "*.txt"

if __name__ == '__main__':
    output_path = PROCESSED_FOLDER

    if not os.path.exists(PROCESSED_FOLDER):
        print('Directory: {} was created'.format(PROCESSED_FOLDER))
        os.makedirs(PROCESSED_FOLDER)


    print('Next files will be processed: {}'.format(MASK_FOR_TEXT))
    print("Processed files will be save to: {}".format(output_path))
    preprocessing(MASK_FOR_TEXT, output_path)
