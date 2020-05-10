import pickle

def load_pickled_file(one_file):
    with open(one_file, 'rb') as f:
        obj = pickle.load(f)
    return obj