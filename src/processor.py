from abc import ABC, abstractmethod
import os
import spacy_udpipe
from .utils import load_pickled_file
from .settings import PATH_TO_RUS_UDPIPE_MODEL
import spacy


def processTag(tag_representation):
    res = {}
    if len(tag_representation.split('|')) > 0:
        for one_subtag in tag_representation.split('|'):
            if len(one_subtag.split('=')) > 1:
                key = one_subtag.split('=')[0]
                value = one_subtag.split('=')[1]
                res[key] = value
    return res

class SyntaxVectorizer(ABC):
    def setup_rules(self):
        pass

    def text_structures_initializer(self):
        pass

    def calculate_morpho_tags(self, current_token):
        pass

    def normalize_morpho_tags(self):
        pass


if __name__ == "__main__":
    # Just checking
    # Please, pay attention, that this class imported

    t = load_pickled_file(os.path.join('ProcessedData', 'Андреев_Ангелочек.pkl'))
    print('Pickle loaded')
    current_nlp_module = spacy_udpipe.load_from_path('ru-syntagrus', PATH_TO_RUS_UDPIPE_MODEL)

    print('Model loaded')
    hj = SyntaxVectorizerRU(current_nlp_module)
    hj.convert_to_attributes(t['Trees'])
    resAttribs = hj.get_res_attributes()
    print('Thats all')