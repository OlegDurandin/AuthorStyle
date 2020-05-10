import glob
from create_RU_dataset.datasetBuilder import ProcessOneFile
from create_RU_dataset.datasetBuilder import PATH_TO_PREPROCESSED_TEXTS
from preprocessor import load_pickled_file

if __name__ == "__main__":
    separator = ProcessOneFile(2000)
    for fname in glob.glob(PATH_TO_PREPROCESSED_TEXTS):
        print('Processing: {}'.format(fname))
        targetDictionary = load_pickled_file(fname)

        author = targetDictionary['Author']
        novel_name = targetDictionary['Novel']
        list_of_trees = targetDictionary['Trees']

        res = separator.calc_separate_blocks(list_of_trees)
        print('In {} we\'ve {} sentences and we could separate them into {} blocks'.format(fname,
                                                                                           len(list_of_trees),
                                                                                           res))
