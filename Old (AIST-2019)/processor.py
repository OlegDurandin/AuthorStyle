from preprocessor import load_pickled_file
import nltk
import pymorphy2
import os

from collections import defaultdict, Counter
from itertools import combinations

from tqdm import tqdm

def processTag(tag_representation):
    res = {}
    if len(tag_representation.split('__')) > 1:
        for one_subtag in tag_representation.split('__')[1].split('|'):
            if len(one_subtag.split('=')) > 1:
                key = one_subtag.split('=')[0]
                value = one_subtag.split('=')[1]
                res[key] = value
    return res



class SyntaxVectorizer:
    LIST_OF_TARGET_FEATURES = ['Конкретные', 'Абстрактные', 'Признак', 'Процесс-существительное',
                               'Состояние', 'Прочее', 'Местоимение', 'Признак предмета', 'Процесс', 'Число',
                               'Признак действия', 'Категория состояния', 'НезнAppealConstructionаменательные', 'Совершенный',
                               'Несовершенный', 'Изъявительное', 'Повелительное', 'Прошедшее', 'Настоящее',
                               'Будущее', 'Действительный', 'Страдательный', 'Всего']

    MORPHO_TAGS = 'MorphoTags'
    SYNTAX_DEP_TAGS = 'SyntaxDependencies'
    BIGRAMS_TREELETS = 'BigramTreelet'
    TRIGRAMS_TREELETS = 'TrigramTreelet'
    DYMARSKY_CLASSES = 'Dymarsky'

    COMPLEX_MORPHO_FEATURES_EXTERNAL = 'ExtendedMorpho'
    COMPLEX_SYNTAX_FEATURES = 'SyntaxComplex'

    # For Memo
    COMPLEX_MORPHO_FEATURES = 'ComplexMorphoFeatures'
    SYNTAX_DEP_AUX_PHRASE_BASED = 'AuxPhraseBasedLevelSyntax'
    SYNTAX_PHRASE_BASED_LEVEL = 'PhraseBasedLevelSyntax'
    SYNTAX_SENTENCE_BASED_LEVEL = 'SentenceBasedLevelSyntax'


    def __init__(self, loaded_spacy_model):
        self.ru_nlp = loaded_spacy_model  # Загрузим языковую модель
        self.morpher = pymorphy2.MorphAnalyzer()
        self.load_ext_morpholists()
        self.total_sentences = 0

    def setup_rules(self):
        self.token_based_functions = [
            (
                SyntaxVectorizer.MORPHO_TAGS,
                self.calculate_morpho_tags,
                self.normalize_morpho_tags
            ),
            (
                SyntaxVectorizer.SYNTAX_DEP_TAGS,
                self.calculate_syntax_tags,
                self.normalize_syntax_tags
            ),
            (
                SyntaxVectorizer.BIGRAMS_TREELETS,
                self.calculate_bigram_treelet,
                self.normalize_bigram_treelet
            ),
            (
                SyntaxVectorizer.TRIGRAMS_TREELETS,
                self.calculate_trigram_treelet,
                self.normalize_trigram_treelet
            ),
            (
                SyntaxVectorizer.DYMARSKY_CLASSES,
                self.calculate_syntax_features_by_dymarsky,
                self.normalize_dymarsky_features
            )

        ]
        self.sentence_based_functions = [
            (
                SyntaxVectorizer.COMPLEX_MORPHO_FEATURES_EXTERNAL,
                self.calculate_complex_Morpho,
                self.normalize_complex_morpho),
            (
                SyntaxVectorizer.COMPLEX_SYNTAX_FEATURES,
                self.calculate_complex_syntax,
                self.normalize_complex_syntax)
        ]

    def load_ext_morpholists(self):
        self.items = self.read_morpho_attributes('предметы.txt')
        self.names = self.read_morpho_attributes('Имена.txt')
        self.abstr = self.read_morpho_attributes('Абстрактные.txt')
        self.prizn = self.read_morpho_attributes('Признак.txt')
        self.proc = self.read_morpho_attributes('Процесс-существительное.txt')
        self.sost = self.read_morpho_attributes('Состояние.txt')

    def read_morpho_attributes(self, fname):
        FOLDER_NAME_WITH_MORPHO_FILES = './/FeatureExtractors//'
        target_list = []
        fullFName = os.path.join(FOLDER_NAME_WITH_MORPHO_FILES, fname)
        print('File: {}'.format(fullFName))
        with open(fullFName, 'rt', encoding='utf-8') as f:
            for l in f.readlines():
                target_list.append(l.strip())
        return target_list

    def text_structures_initializer(self):
        self.Attributes = {}
        self.Attributes[SyntaxVectorizer.MORPHO_TAGS] = Counter()
        self.Attributes[SyntaxVectorizer.SYNTAX_DEP_TAGS] = Counter()
        self.Attributes[SyntaxVectorizer.BIGRAMS_TREELETS] = Counter()
        self.Attributes[SyntaxVectorizer.TRIGRAMS_TREELETS] = Counter()
        self.Attributes[SyntaxVectorizer.DYMARSKY_CLASSES] = Counter()
        self.Attributes[SyntaxVectorizer.COMPLEX_MORPHO_FEATURES_EXTERNAL] = Counter()
        self.Attributes[SyntaxVectorizer.COMPLEX_SYNTAX_FEATURES] = Counter()

        self.Memo = {}
        self.Memo[SyntaxVectorizer.COMPLEX_MORPHO_FEATURES] = Counter()
        self.Memo[SyntaxVectorizer.SYNTAX_DEP_AUX_PHRASE_BASED] = Counter()
        self.Memo[SyntaxVectorizer.SYNTAX_PHRASE_BASED_LEVEL] = Counter()
        self.Memo[SyntaxVectorizer.SYNTAX_SENTENCE_BASED_LEVEL] = Counter()

        self.total_sentences = 0




    def calculate_morpho_tags(self, current_token):
        self.Attributes[SyntaxVectorizer.MORPHO_TAGS][current_token.pos_] += 1
    def normalize_morpho_tags(self):
        total = sum(self.Attributes[SyntaxVectorizer.MORPHO_TAGS].values(), 0.0)
        for key in self.Attributes[SyntaxVectorizer.MORPHO_TAGS]:
            self.Attributes[SyntaxVectorizer.MORPHO_TAGS][key] /= total

    def calculate_syntax_tags(self, current_token):
        self.Attributes[SyntaxVectorizer.SYNTAX_DEP_TAGS][current_token.dep_] += 1
    def normalize_syntax_tags(self):
        total = sum(self.Attributes[SyntaxVectorizer.SYNTAX_DEP_TAGS].values(), 0.0)
        for key in self.Attributes[SyntaxVectorizer.SYNTAX_DEP_TAGS]:
            self.Attributes[SyntaxVectorizer.SYNTAX_DEP_TAGS][key] /= total

    def calculate_bigram_treelet(self, current_token):
        if current_token.dep != 'punct':
            self.Attributes[SyntaxVectorizer.BIGRAMS_TREELETS][(current_token.head.pos_, current_token.dep_, current_token.pos_)] += 1
    def normalize_bigram_treelet(self):
        total = sum(self.Attributes[SyntaxVectorizer.BIGRAMS_TREELETS].values(), 0.0)
        for key in self.Attributes[SyntaxVectorizer.BIGRAMS_TREELETS]:
            self.Attributes[SyntaxVectorizer.BIGRAMS_TREELETS][key] /= total

    def calculate_trigram_treelet(self, current_token):
        for (left_child, right_child) in combinations(current_token.children, 2):
            if left_child.dep_ != 'punct' and right_child.dep_ != 'punct':
                self.Attributes[SyntaxVectorizer.TRIGRAMS_TREELETS][
                    (1, left_child.pos_, left_child.dep_, current_token.pos_, right_child.dep_, right_child.pos_)] += 1
        if current_token.dep_ != 'ROOT':
            for child in current_token.children:
                if child.dep_ == 'punct' or current_token.dep_ == 'punct':
                    continue
                self.Attributes[SyntaxVectorizer.TRIGRAMS_TREELETS][
                    (2, child.pos_, child.dep_, current_token.pos_, current_token.dep_, current_token.head.pos_)] += 1
    def normalize_trigram_treelet(self):
        # Вопрос по нормализации: должны ли они быть независимы друг от друга
        total = sum(self.Attributes[SyntaxVectorizer.TRIGRAMS_TREELETS].values(), 0.0)
        for key in self.Attributes[SyntaxVectorizer.TRIGRAMS_TREELETS]:
            self.Attributes[SyntaxVectorizer.TRIGRAMS_TREELETS][key] /= total

    def calculate_syntax_features_by_dymarsky(self, current_token):
        # P (person): nsubj, obj, iobj, vocative, dislocated, appos, nsubj:pass.
        # A (action): csubj, ccomp, xcomp, aux, cop, aux:pass, csubj:pass.
        # R (relations): ???
        # L (location): obl
        # T (time): obl
        # Circ (сирконстант, обстоятельство): advcl, advmod, discourse
        # Ch (characteristics): nmod, nummod, acl, amod, det, acl:recl
        if current_token.dep_ in ['nsubj', 'obj', 'iobj', 'vocative', 'dislocated', 'appos', 'nsubj:pass']:
            self.Attributes[SyntaxVectorizer.DYMARSKY_CLASSES]['person'] += 1
        elif current_token.dep_ in ['csubj', 'ccomp', 'xcomp', 'aux', 'cop', 'aux:pass', 'csubj:pass']:
            self.Attributes[SyntaxVectorizer.DYMARSKY_CLASSES]['action'] += 1
        elif current_token.dep_ in ['obl']:
            self.Attributes[SyntaxVectorizer.DYMARSKY_CLASSES]['location'] += 1
        elif current_token.dep_ in ['obl']:
            self.Attributes[SyntaxVectorizer.DYMARSKY_CLASSES]['time'] += 1
        elif current_token.dep_ in ['advcl', 'advmod', 'discourse']:
            self.Attributes[SyntaxVectorizer.DYMARSKY_CLASSES]['Ch'] += 1
        elif current_token.dep_ in ['nmod', 'nummod', 'acl', 'amod', 'det', 'acl:recl']:
            self.Attributes[SyntaxVectorizer.DYMARSKY_CLASSES]['Circ'] += 1
    def normalize_dymarsky_features(self):
        total = sum(self.Attributes[SyntaxVectorizer.DYMARSKY_CLASSES].values(), 0.0)
        for key in self.Attributes[SyntaxVectorizer.DYMARSKY_CLASSES]:
            self.Attributes[SyntaxVectorizer.DYMARSKY_CLASSES][key] /= total

    def calculate_complex_Morpho(self, parsed_sentence):
        current_text = parsed_sentence.text.strip('\ufeff')
        text = nltk.Text(nltk.word_tokenize(current_text))
        for i in range(0, len(text)):
            p = self.morpher.parse(text[i])[0]
            self.Memo[SyntaxVectorizer.COMPLEX_MORPHO_FEATURES]['Всего'] += 1
            if 'NOUN' == p.tag.POS:
                noun = p.normal_form.upper()
                if noun in self.items:
                    self.Memo[SyntaxVectorizer.COMPLEX_MORPHO_FEATURES]['Конкретные'] += 1
                elif noun in self.names:
                    self.Memo[SyntaxVectorizer.COMPLEX_MORPHO_FEATURES]['Конкретные'] += 1
                elif noun in self.prizn:
                    self.Memo[SyntaxVectorizer.COMPLEX_MORPHO_FEATURES]['Признак'] += 1
                elif noun in self.proc:
                    self.Memo[SyntaxVectorizer.COMPLEX_MORPHO_FEATURES]['Процесс-существительное'] += 1
                elif noun in self.sost:
                    self.Memo[SyntaxVectorizer.COMPLEX_MORPHO_FEATURES]['Состояние'] += 1
                elif noun in self.abstr:
                    self.Memo[SyntaxVectorizer.COMPLEX_MORPHO_FEATURES]['Абстрактные'] += 1
                else:
                    self.Memo[SyntaxVectorizer.COMPLEX_MORPHO_FEATURES]['Прочее'] += 1
            elif 'NPRO' == p.tag.POS:
                self.Memo[SyntaxVectorizer.COMPLEX_MORPHO_FEATURES]['Местоимение'] += 1
            elif 'ADJF' == p.tag.POS:
                self.Memo[SyntaxVectorizer.COMPLEX_MORPHO_FEATURES]['Признак предмета'] += 1
            elif 'ADJS' == p.tag.POS:
                self.Memo[SyntaxVectorizer.COMPLEX_MORPHO_FEATURES]['Признак предмета'] += 1
            elif 'COMP' == p.tag.POS:
                self.Memo[SyntaxVectorizer.COMPLEX_MORPHO_FEATURES]['Признак предмета'] += 1
            elif 'ADVB' == p.tag.POS:
                self.Memo[SyntaxVectorizer.COMPLEX_MORPHO_FEATURES]['Признак действия'] += 1
            elif 'VERB' == p.tag.POS:
                self.Memo[SyntaxVectorizer.COMPLEX_MORPHO_FEATURES]['Процесс'] += 1
                if p.tag.aspect == 'perf':
                    self.Memo[SyntaxVectorizer.COMPLEX_MORPHO_FEATURES]['Совершенный'] += 1
                else:
                    self.Memo[SyntaxVectorizer.COMPLEX_MORPHO_FEATURES]['Несовершенный'] += 1
                if p.tag.mood == 'indc':
                    self.Memo[SyntaxVectorizer.COMPLEX_MORPHO_FEATURES]['Изъявительное'] += 1
                else:
                    self.Memo[SyntaxVectorizer.COMPLEX_MORPHO_FEATURES]['Повелительное'] += 1
                if p.tag.tense == 'past':
                    self.Memo[SyntaxVectorizer.COMPLEX_MORPHO_FEATURES]['Прошедшее'] += 1
                elif p.tag.tense == 'pres':
                    self.Memo[SyntaxVectorizer.COMPLEX_MORPHO_FEATURES]['Настоящее'] += 1
                else:
                    self.Memo[SyntaxVectorizer.COMPLEX_MORPHO_FEATURES]['Будущее'] += 1
                if p.tag.voice == 'pssv':
                    self.Memo[SyntaxVectorizer.COMPLEX_MORPHO_FEATURES]['Страдательный'] += 1
                elif p.tag.voice == 'actv':
                    self.Memo[SyntaxVectorizer.COMPLEX_MORPHO_FEATURES]['Действительный'] += 1
            elif 'INFN' == p.tag.POS:
                self.Memo[SyntaxVectorizer.COMPLEX_MORPHO_FEATURES]['Процесс'] += 1
            elif 'PRTF' == p.tag.POS:
                self.Memo[SyntaxVectorizer.COMPLEX_MORPHO_FEATURES]['Процесс'] += 1
                if p.tag.voice == 'pssv':
                    self.Memo[SyntaxVectorizer.COMPLEX_MORPHO_FEATURES]['Страдательный'] += 1
                elif p.tag.voice == 'actv':
                    self.Memo[SyntaxVectorizer.COMPLEX_MORPHO_FEATURES]['Действительный'] += 1
            elif 'PRTS' == p.tag.POS:
                self.Memo[SyntaxVectorizer.COMPLEX_MORPHO_FEATURES]['Процесс'] += 1
                if p.tag.voice == 'pssv':
                    self.Memo[SyntaxVectorizer.COMPLEX_MORPHO_FEATURES]['Страдательный'] += 1
                elif p.tag.voice == 'actv':
                    self.Memo[SyntaxVectorizer.COMPLEX_MORPHO_FEATURES]['Действительный'] += 1
            elif 'GRND' == p.tag.POS:
                self.Memo[SyntaxVectorizer.COMPLEX_MORPHO_FEATURES]['Процесс'] += 1
            elif 'NUMR' == p.tag.POS:
                self.Memo[SyntaxVectorizer.COMPLEX_MORPHO_FEATURES]['Число'] += 1
            elif 'PRED' == p.tag.POS:
                self.Memo[SyntaxVectorizer.COMPLEX_MORPHO_FEATURES]['Категория состояния'] += 1
            elif 'PREP' == p.tag.POS:
                self.Memo[SyntaxVectorizer.COMPLEX_MORPHO_FEATURES]['Незнаменательные'] += 1
            elif 'CONJ' == p.tag.POS:
                self.Memo[SyntaxVectorizer.COMPLEX_MORPHO_FEATURES]['Незнаменательные'] += 1
            elif 'PRCL' == p.tag.POS:
                self.Memo[SyntaxVectorizer.COMPLEX_MORPHO_FEATURES]['Незнаменательные'] += 1
            elif 'INTJ' == p.tag.POS:
                self.Memo[SyntaxVectorizer.COMPLEX_MORPHO_FEATURES]['Незнаменательные'] += 1
            else:
                self.Memo[SyntaxVectorizer.COMPLEX_MORPHO_FEATURES]['Всего'] -= 1

    def normalize_complex_morpho(self):
        self.Attributes[SyntaxVectorizer.COMPLEX_MORPHO_FEATURES_EXTERNAL]['Абстрактность'] = \
            (self.Memo[SyntaxVectorizer.COMPLEX_MORPHO_FEATURES]['Абстрактные'] +
             self.Memo[SyntaxVectorizer.COMPLEX_MORPHO_FEATURES]['Признак'] +
             self.Memo[SyntaxVectorizer.COMPLEX_MORPHO_FEATURES]['Процесс-существительное'] +
             self.Memo[SyntaxVectorizer.COMPLEX_MORPHO_FEATURES]['Состояние']) / (self.Memo[SyntaxVectorizer.COMPLEX_MORPHO_FEATURES]['Конкретные'] +
                                                                                  self.Memo[SyntaxVectorizer.COMPLEX_MORPHO_FEATURES]['Местоимение'] +
                                                                                  self.Memo[SyntaxVectorizer.COMPLEX_MORPHO_FEATURES]['Абстрактные'] +
                                                                                  self.Memo[SyntaxVectorizer.COMPLEX_MORPHO_FEATURES]['Признак'] +
                                                                                  self.Memo[SyntaxVectorizer.COMPLEX_MORPHO_FEATURES]['Процесс-существительное'] +
                                                                                  self.Memo[SyntaxVectorizer.COMPLEX_MORPHO_FEATURES]['Состояние'])

        self.Attributes[SyntaxVectorizer.COMPLEX_MORPHO_FEATURES_EXTERNAL]['Предметность'] = \
            (self.Memo[SyntaxVectorizer.COMPLEX_MORPHO_FEATURES]['Конкретные'] + self.Memo[SyntaxVectorizer.COMPLEX_MORPHO_FEATURES]['Местоимение']) / \
            (self.Memo[SyntaxVectorizer.COMPLEX_MORPHO_FEATURES]['Всего'] - self.Memo[SyntaxVectorizer.COMPLEX_MORPHO_FEATURES]['Незнаменательные'])

        self.Attributes[SyntaxVectorizer.COMPLEX_MORPHO_FEATURES_EXTERNAL]['Местоименная замена'] = \
            self.Memo[SyntaxVectorizer.COMPLEX_MORPHO_FEATURES]['Местоимение'] / (
                    self.Memo[SyntaxVectorizer.COMPLEX_MORPHO_FEATURES]['Конкретные'] + self.Memo[SyntaxVectorizer.COMPLEX_MORPHO_FEATURES]['Местоимение'])

        self.Attributes[SyntaxVectorizer.COMPLEX_MORPHO_FEATURES_EXTERNAL]['Признак действия'] = \
            self.Memo[SyntaxVectorizer.COMPLEX_MORPHO_FEATURES]['Признак действия'] / (
                    self.Memo[SyntaxVectorizer.COMPLEX_MORPHO_FEATURES]['Признак'] +
                    self.Memo[SyntaxVectorizer.COMPLEX_MORPHO_FEATURES]['Признак предмета'] +
                    self.Memo[SyntaxVectorizer.COMPLEX_MORPHO_FEATURES]['Признак действия'])

        self.Attributes[SyntaxVectorizer.COMPLEX_MORPHO_FEATURES_EXTERNAL]['Обобщенность действия'] = \
            self.Memo[SyntaxVectorizer.COMPLEX_MORPHO_FEATURES]['Признак'] / (
                    self.Memo[SyntaxVectorizer.COMPLEX_MORPHO_FEATURES]['Признак предмета'] +
                    self.Memo[SyntaxVectorizer.COMPLEX_MORPHO_FEATURES]['Признак действия'] +
                    self.Memo[SyntaxVectorizer.COMPLEX_MORPHO_FEATURES]['Признак'])

        self.Attributes[SyntaxVectorizer.COMPLEX_MORPHO_FEATURES_EXTERNAL]['Описательность'] = \
            self.Memo[SyntaxVectorizer.COMPLEX_MORPHO_FEATURES]['Признак предмета'] / (
                    self.Memo[SyntaxVectorizer.COMPLEX_MORPHO_FEATURES]['Конкретные'] +
                    self.Memo[SyntaxVectorizer.COMPLEX_MORPHO_FEATURES]['Признак'] +
                    self.Memo[SyntaxVectorizer.COMPLEX_MORPHO_FEATURES]['Процесс-существительное'] +
                    self.Memo[SyntaxVectorizer.COMPLEX_MORPHO_FEATURES]['Состояние'] +
                    self.Memo[SyntaxVectorizer.COMPLEX_MORPHO_FEATURES]['Абстрактные'] +
                    self.Memo[SyntaxVectorizer.COMPLEX_MORPHO_FEATURES]['Прочее'])

        self.Attributes[SyntaxVectorizer.COMPLEX_MORPHO_FEATURES_EXTERNAL]['Описательность действия'] = \
            self.Memo[SyntaxVectorizer.COMPLEX_MORPHO_FEATURES]['Признак действия'] / \
            self.Memo[SyntaxVectorizer.COMPLEX_MORPHO_FEATURES]['Процесс']

        self.Attributes[SyntaxVectorizer.COMPLEX_MORPHO_FEATURES_EXTERNAL]['Численность'] = \
            self.Memo[SyntaxVectorizer.COMPLEX_MORPHO_FEATURES]['Число'] / (
                    self.Memo[SyntaxVectorizer.COMPLEX_MORPHO_FEATURES]['Конкретные'] +
                    self.Memo[SyntaxVectorizer.COMPLEX_MORPHO_FEATURES]['Признак'] +
                    self.Memo[SyntaxVectorizer.COMPLEX_MORPHO_FEATURES]['Процесс-существительное'] +
                    self.Memo[SyntaxVectorizer.COMPLEX_MORPHO_FEATURES]['Состояние'] +
                    self.Memo[SyntaxVectorizer.COMPLEX_MORPHO_FEATURES]['Абстрактные'] +
                    self.Memo[SyntaxVectorizer.COMPLEX_MORPHO_FEATURES]['Прочее'])

        self.Attributes[SyntaxVectorizer.COMPLEX_MORPHO_FEATURES_EXTERNAL]['Динамичность'] = (
                                                                                                     self.Memo[SyntaxVectorizer.COMPLEX_MORPHO_FEATURES]['Процесс-существительное'] +
                                                                                                     self.Memo[SyntaxVectorizer.COMPLEX_MORPHO_FEATURES]['Процесс']) / (
                                                                                                     self.Memo[SyntaxVectorizer.COMPLEX_MORPHO_FEATURES]['Всего'] -
                                                                                                     self.Memo[SyntaxVectorizer.COMPLEX_MORPHO_FEATURES]['Незнаменательные'])

        self.Attributes[SyntaxVectorizer.COMPLEX_MORPHO_FEATURES_EXTERNAL]['Состояние'] = (
                                                                                                  self.Memo[SyntaxVectorizer.COMPLEX_MORPHO_FEATURES]['Состояние'] +
                                                                                                  self.Memo[SyntaxVectorizer.COMPLEX_MORPHO_FEATURES]['Категория состояния']) / (
                                                                                                  self.Memo[SyntaxVectorizer.COMPLEX_MORPHO_FEATURES]['Всего'] -
                                                                                                  self.Memo[SyntaxVectorizer.COMPLEX_MORPHO_FEATURES]['Незнаменательные'])

        self.Attributes[SyntaxVectorizer.COMPLEX_MORPHO_FEATURES_EXTERNAL]['Реальная модальность'] = \
            self.Memo[SyntaxVectorizer.COMPLEX_MORPHO_FEATURES]['Изъявительное'] / (
                    self.Memo[SyntaxVectorizer.COMPLEX_MORPHO_FEATURES]['Повелительное'] +
                    self.Memo[SyntaxVectorizer.COMPLEX_MORPHO_FEATURES]['Изъявительное'])

        self.Attributes[SyntaxVectorizer.COMPLEX_MORPHO_FEATURES_EXTERNAL]['Пассив'] = \
            self.Memo[SyntaxVectorizer.COMPLEX_MORPHO_FEATURES]['Страдательный'] / (
                    self.Memo[SyntaxVectorizer.COMPLEX_MORPHO_FEATURES]['Действительный'] +
                    self.Memo[SyntaxVectorizer.COMPLEX_MORPHO_FEATURES]['Страдательный'])

        self.Attributes[SyntaxVectorizer.COMPLEX_MORPHO_FEATURES_EXTERNAL]['Настоящее'] = \
            self.Memo[SyntaxVectorizer.COMPLEX_MORPHO_FEATURES]['Настоящее'] / (
                    self.Memo[SyntaxVectorizer.COMPLEX_MORPHO_FEATURES]['Настоящее'] +
                    self.Memo[SyntaxVectorizer.COMPLEX_MORPHO_FEATURES]['Прошедшее'] +
                    self.Memo[SyntaxVectorizer.COMPLEX_MORPHO_FEATURES]['Будущее'])

        self.Attributes[SyntaxVectorizer.COMPLEX_MORPHO_FEATURES_EXTERNAL]['Прошедшее'] = \
            self.Memo[SyntaxVectorizer.COMPLEX_MORPHO_FEATURES]['Прошедшее'] / (
                    self.Memo[SyntaxVectorizer.COMPLEX_MORPHO_FEATURES]['Настоящее'] +
                    self.Memo[SyntaxVectorizer.COMPLEX_MORPHO_FEATURES]['Прошедшее'] +
                    self.Memo[SyntaxVectorizer.COMPLEX_MORPHO_FEATURES]['Будущее'])

        self.Attributes[SyntaxVectorizer.COMPLEX_MORPHO_FEATURES_EXTERNAL]['Будущее'] = \
            self.Memo[SyntaxVectorizer.COMPLEX_MORPHO_FEATURES]['Будущее'] / (
                    self.Memo[SyntaxVectorizer.COMPLEX_MORPHO_FEATURES]['Настоящее'] +
                    self.Memo[SyntaxVectorizer.COMPLEX_MORPHO_FEATURES]['Прошедшее'] +
                    self.Memo[SyntaxVectorizer.COMPLEX_MORPHO_FEATURES]['Будущее'])

        self.Attributes[SyntaxVectorizer.COMPLEX_MORPHO_FEATURES_EXTERNAL]['Совершенность действия'] = \
            self.Memo[SyntaxVectorizer.COMPLEX_MORPHO_FEATURES]['Совершенный'] / (
                    self.Memo[SyntaxVectorizer.COMPLEX_MORPHO_FEATURES]['Несовершенный'] +
                    self.Memo[SyntaxVectorizer.COMPLEX_MORPHO_FEATURES]['Совершенный'])


    def convert_to_attributes(self, list_of_parsed_trees):
        self.text_structures_initializer()
        self.setup_rules()
        for one_dependency_tree in tqdm(list_of_parsed_trees):
            for one_token_in_parsed_sentence in one_dependency_tree:
                # Теперь идём по каждому фрагменту дерева зависимостей
                # Берем каждый токен и применяем к нему token-based функции
                for _, token_processing_function, _ in self.token_based_functions:
                    token_processing_function(one_token_in_parsed_sentence)

            for _, sentence_processing_function, _ in self.sentence_based_functions:
                sentence_processing_function(one_dependency_tree)
                self.total_sentences += 1

        for name,_, normalize_function in self.token_based_functions:
            #print('Normalization procedure for {} features (token-based)'.format(name))
            normalize_function()
        for name,_, normalize_function in self.sentence_based_functions:
            #print('Normalization procedure for {} features (sentence-based)'.format(name))
            normalize_function()

    def get_res_attributes(self):
        return self.Attributes

    def normalize_complex_syntax(self):
        # Общее количество всех синтаксических зависимостей
        total_count_all_dependencies = sum([self.Memo[SyntaxVectorizer.SYNTAX_DEP_AUX_PHRASE_BASED][x] for x in self.Memo[SyntaxVectorizer.SYNTAX_DEP_AUX_PHRASE_BASED]])
        coordination_tags = ['nsubj', 'csubj', 'expl', 'csubj:pass', 'nsubj:pass']  # Координация
        agreement_tags = ['nmod', 'appos', 'acl', 'amod', 'det', 'nummod', 'compound']  # Согласование
        control_tags = ['obj', 'iobj', 'ccomp', 'obl']  # Управление
        contiguity_tags = ['advmod']  # Примыкание

        # Классификация по типу связи: Координация
        self.Attributes[SyntaxVectorizer.COMPLEX_SYNTAX_FEATURES]['CoordinationPB'] = sum(
            [self.Memo[SyntaxVectorizer.SYNTAX_DEP_AUX_PHRASE_BASED].get(x,0) for x in coordination_tags]) / total_count_all_dependencies
        # Классификация по типу связи: Согласование
        self.Attributes[SyntaxVectorizer.COMPLEX_SYNTAX_FEATURES]['AgreementPB'] = sum(
            [self.Memo[SyntaxVectorizer.SYNTAX_DEP_AUX_PHRASE_BASED].get(x, 0) for x in agreement_tags]) / total_count_all_dependencies
        # Классификация по типу связи: Управление
        self.Attributes[SyntaxVectorizer.COMPLEX_SYNTAX_FEATURES]['ControlPB'] = sum(
            [self.Memo[SyntaxVectorizer.SYNTAX_DEP_AUX_PHRASE_BASED].get(x, 0) for x in control_tags]) / total_count_all_dependencies
        # Классификация по типу связи: Примыкание
        self.Attributes[SyntaxVectorizer.COMPLEX_SYNTAX_FEATURES]['ContiguityPB'] = sum(
            [self.Memo[SyntaxVectorizer.SYNTAX_DEP_AUX_PHRASE_BASED].get(x, 0) for x in contiguity_tags]) / total_count_all_dependencies

        # Количественно-структурные типы
        self.Attributes[SyntaxVectorizer.COMPLEX_SYNTAX_FEATURES]['comphr'] = self.Memo[SyntaxVectorizer.SYNTAX_PHRASE_BASED_LEVEL]['comphr'] / total_count_all_dependencies
        self.Attributes[SyntaxVectorizer.COMPLEX_SYNTAX_FEATURES]['simlphr'] = self.Memo[SyntaxVectorizer.SYNTAX_PHRASE_BASED_LEVEL]['simlphr'] / total_count_all_dependencies

        # Синтаксически-несвободные словосочетания
        syntactically_non_free_phrases_tags = ['fixed', 'flat', 'compound']
        self.Attributes[SyntaxVectorizer.COMPLEX_SYNTAX_FEATURES]['syntaxNonFreePhrases'] = sum([
            self.Memo[SyntaxVectorizer.SYNTAX_DEP_AUX_PHRASE_BASED].get(x, 0) for x in syntactically_non_free_phrases_tags]) / total_count_all_dependencies
        # Лексико-грамматические типы
        self.Attributes[SyntaxVectorizer.COMPLEX_SYNTAX_FEATURES]['nomphr'] = self.Memo[SyntaxVectorizer.SYNTAX_PHRASE_BASED_LEVEL]['nomphr'] / total_count_all_dependencies  # именной тип
        self.Attributes[SyntaxVectorizer.COMPLEX_SYNTAX_FEATURES]['verbphr'] = self.Memo[SyntaxVectorizer.SYNTAX_PHRASE_BASED_LEVEL]['verbphr'] / total_count_all_dependencies  # глагольный тип
        self.Attributes[SyntaxVectorizer.COMPLEX_SYNTAX_FEATURES]['advphr'] = self.Memo[SyntaxVectorizer.SYNTAX_PHRASE_BASED_LEVEL]['advphr'] / total_count_all_dependencies  # наречный тип

        # Членимые и нечленимые предложения
        self.Attributes[SyntaxVectorizer.COMPLEX_SYNTAX_FEATURES]['NotSeparatedSentences'] = self.Memo[SyntaxVectorizer.SYNTAX_SENTENCE_BASED_LEVEL]['NotseparatedSentence'] / self.total_sentences
        self.Attributes[SyntaxVectorizer.COMPLEX_SYNTAX_FEATURES]['Vocative'] = self.Memo[SyntaxVectorizer.SYNTAX_SENTENCE_BASED_LEVEL]['Vocative'] / self.total_sentences
        self.Attributes[SyntaxVectorizer.COMPLEX_SYNTAX_FEATURES]['Genitive'] = self.Memo[SyntaxVectorizer.SYNTAX_SENTENCE_BASED_LEVEL]['Genitive'] / self.total_sentences

        # Односоставные и двусоставные слова
        # Односоставные
        self.Attributes[SyntaxVectorizer.COMPLEX_SYNTAX_FEATURES]['SingleCompose'] = self.Memo[SyntaxVectorizer.SYNTAX_SENTENCE_BASED_LEVEL]['Single'] / self.total_sentences
        # Определенно-личные
        self.Attributes[SyntaxVectorizer.COMPLEX_SYNTAX_FEATURES]['DefinetelyPersonal'] = self.Memo[SyntaxVectorizer.SYNTAX_SENTENCE_BASED_LEVEL]['DefinetelyPersonal'] / self.total_sentences
        # Неопределенно-личные
        self.Attributes[SyntaxVectorizer.COMPLEX_SYNTAX_FEATURES]['UndefinetelyPersonal'] = self.Memo[SyntaxVectorizer.SYNTAX_SENTENCE_BASED_LEVEL]['UndefinetelyPersonal'] / self.total_sentences

        # Инфинитивные
        self.Attributes[SyntaxVectorizer.COMPLEX_SYNTAX_FEATURES]['Infinitive'] = self.Memo[SyntaxVectorizer.SYNTAX_SENTENCE_BASED_LEVEL]['Infinitive'] / self.total_sentences
        # Безличные
        self.Attributes[SyntaxVectorizer.COMPLEX_SYNTAX_FEATURES]['UnPersonal'] = self.Memo[SyntaxVectorizer.SYNTAX_SENTENCE_BASED_LEVEL]['UnPersonal'] / self.total_sentences
        # Номинативные
        self.Attributes[SyntaxVectorizer.COMPLEX_SYNTAX_FEATURES]['Nominative'] = self.Memo[SyntaxVectorizer.SYNTAX_SENTENCE_BASED_LEVEL]['Nominative'] / self.total_sentences

        # Осложненные конструкции
        self.Attributes[SyntaxVectorizer.COMPLEX_SYNTAX_FEATURES]['IntroConstruction'] = self.Memo[SyntaxVectorizer.SYNTAX_SENTENCE_BASED_LEVEL]['IntroConstruction'] / self.total_sentences
        self.Attributes[SyntaxVectorizer.COMPLEX_SYNTAX_FEATURES]['InterjectionConstruction']= self.Memo[SyntaxVectorizer.SYNTAX_SENTENCE_BASED_LEVEL]['InterjectionConstruction'] / self.total_sentences
        self.Attributes[SyntaxVectorizer.COMPLEX_SYNTAX_FEATURES]['AppealConstruction'] = self.Memo[SyntaxVectorizer.SYNTAX_SENTENCE_BASED_LEVEL]['AppealConstruction'] / self.total_sentences
        self.Attributes[SyntaxVectorizer.COMPLEX_SYNTAX_FEATURES]['InsertConstruction'] = self.Memo[SyntaxVectorizer.SYNTAX_SENTENCE_BASED_LEVEL]['InsertConstruction'] / self.total_sentences

        self.Attributes[SyntaxVectorizer.COMPLEX_SYNTAX_FEATURES]['AdjTurnover'] = self.Memo[SyntaxVectorizer.SYNTAX_SENTENCE_BASED_LEVEL]['AdjTurnover'] / self.total_sentences
        self.Attributes[SyntaxVectorizer.COMPLEX_SYNTAX_FEATURES]['verbalAdverb'] = self.Memo[SyntaxVectorizer.SYNTAX_SENTENCE_BASED_LEVEL]['verbalAdverb'] / self.total_sentences
        self.Attributes[SyntaxVectorizer.COMPLEX_SYNTAX_FEATURES]['participle'] = self.Memo[SyntaxVectorizer.SYNTAX_SENTENCE_BASED_LEVEL]['participle'] / self.total_sentences
        self.Attributes[SyntaxVectorizer.COMPLEX_SYNTAX_FEATURES]['ComplicatedByHomogeneous'] = self.Memo[SyntaxVectorizer.SYNTAX_SENTENCE_BASED_LEVEL]['ComplicatedByHomogeneous'] / self.total_sentences
        self.Attributes[SyntaxVectorizer.COMPLEX_SYNTAX_FEATURES]['ComplicatedByApplication'] = self.Memo[SyntaxVectorizer.SYNTAX_SENTENCE_BASED_LEVEL]['ComplicatedByApplication'] / self.total_sentences




    def calculate_complex_syntax(self, parsed_sentence):
        complicationPOSCondition = 'PUNCT' in [token.pos_ for token in parsed_sentence[:-1]] # Кроме последнего знака
        complicationDEPCondition = 'parataxis' in [token.dep_ for token in parsed_sentence]

        if complicationPOSCondition:
            positionsOfCommas = [index for (index, token) in enumerate(parsed_sentence[:-1]) if token.pos_ == 'PUNCT'
                                 and token.text == ',']
            if positionsOfCommas:
                minPosition = min(positionsOfCommas)
                maxPosition = max(positionsOfCommas)

                for current_index in range(minPosition, maxPosition):
                    current_token = parsed_sentence[current_index]
                    if current_token.dep_ in ['amod']:
                        self.Memo[SyntaxVectorizer.SYNTAX_SENTENCE_BASED_LEVEL]['AdjTurnover'] += 1
                    elif current_token.dep_ in ['advcl']:
                        self.Memo[SyntaxVectorizer.SYNTAX_SENTENCE_BASED_LEVEL]['verbalAdverb'] += 1
                    elif current_token.dep_ in ['acl']:
                        self.Memo[SyntaxVectorizer.SYNTAX_SENTENCE_BASED_LEVEL]['participle'] += 1


        if 'conj' in [token.dep_ for token in parsed_sentence]:
            # - Если внутри предложения есть отношение conj (не обращаем внимания на знаки препинания) → осложнено однородными членами (complicated)
            self.Memo[SyntaxVectorizer.SYNTAX_SENTENCE_BASED_LEVEL]['ComplicatedByHomogeneous'] += 1

        if 'appos' in [token.dep_ for token in parsed_sentence]:
            #  Если есть отношение appos (не обращаем внимания на знаки препинания) → осложнено приложением
            self.Memo[SyntaxVectorizer.SYNTAX_SENTENCE_BASED_LEVEL]['ComplicatedByApplication'] += 1


        for current_token in parsed_sentence:
            if current_token.dep_ != '' and current_token.pos_ != 'SPACE':
                self.Memo[SyntaxVectorizer.SYNTAX_DEP_AUX_PHRASE_BASED][current_token.dep_] += 1
            else:
                continue

            # Количество потомков
            childCount = len(list(current_token.children))
            if childCount > 1:
                self.Memo[SyntaxVectorizer.SYNTAX_PHRASE_BASED_LEVEL]['comphr'] += 1
            else:
                self.Memo[SyntaxVectorizer.SYNTAX_PHRASE_BASED_LEVEL]['simlphr'] += 1

            if  current_token.dep_ == 'parataxis':
                list_of_childs = [x.pos_ for x in current_token.children]
                if 'ADV' in list_of_childs:           # Вводная конструкция
                    self.Memo[SyntaxVectorizer.SYNTAX_SENTENCE_BASED_LEVEL]['IntroConstruction'] += 1
                elif 'INTJ' in list_of_childs:        # междометие
                    self.Memo[SyntaxVectorizer.SYNTAX_SENTENCE_BASED_LEVEL]['InterjectionConstruction'] += 1
                elif 'PROPN' in list_of_childs:
                    self.Memo[SyntaxVectorizer.SYNTAX_SENTENCE_BASED_LEVEL]['AppealConstruction'] += 1
                else:
                    self.Memo[SyntaxVectorizer.SYNTAX_SENTENCE_BASED_LEVEL]['InsertConstruction'] += 1

            if childCount > 0:
                if current_token.pos_ in ['NOUN', 'ADJ', 'NUM', 'PRON']:
                    self.Memo[SyntaxVectorizer.SYNTAX_PHRASE_BASED_LEVEL]['nomphr'] += 1
                elif current_token.pos_ in ['VERB', 'причастие', 'деепричастие']:
                    self.Memo[SyntaxVectorizer.SYNTAX_PHRASE_BASED_LEVEL]['verbphr'] += 1
                elif current_token.pos_ in ['ADV']:
                    self.Memo[SyntaxVectorizer.SYNTAX_PHRASE_BASED_LEVEL]['advphr'] += 1
            if current_token.dep_ == 'ROOT':
                if current_token.pos_ in ['INTJ', 'PART']:
                    # Нечленимые предложения
                    self.Memo[SyntaxVectorizer.SYNTAX_SENTENCE_BASED_LEVEL]['NotseparatedSentence'] += 1
                elif current_token.pos_ in ['PROPN']:
                    # Вокативы
                    self.Memo[SyntaxVectorizer.SYNTAX_SENTENCE_BASED_LEVEL]['Vocative'] += 1
                elif current_token.pos_ in ['NOUN']:
                    # Генитивные
                    morphoDictRes = processTag(current_token.tag_)
                    if 'Case' in morphoDictRes:
                        if morphoDictRes['Case'] in ['Par', 'Gen']:
                            self.Memo[SyntaxVectorizer.SYNTAX_SENTENCE_BASED_LEVEL]['Genitive'] += 1
                if (('nsubj' not in list(map(lambda x: x.dep_, current_token.children))) or
                    ('csubj' not in list(map(lambda x: x.dep_, current_token.children)))):
                    # Односоставное
                    # В правилох непонятно, выделяется односоставное и среди них определенно-личные
                    # обобщенно-личные, неопределенно-личные, инфинитивные, бюезличные, номинативные и т.д.?
                    self.Memo[SyntaxVectorizer.SYNTAX_SENTENCE_BASED_LEVEL]['Single'] += 1
                #else:
                #   print('Non single: {}'.format(parsed_sentence))

                if current_token.pos_ == 'VERB':
                    # Определенно-личные (и обобщенно-личные)
                    morphoDictRes = processTag(current_token.tag_)
                    if ('Number' in morphoDictRes):
                        if ('Tense' in morphoDictRes):
                            if morphoDictRes['Number'] == 'Plur' and morphoDictRes['Tense'] == 'Past':
                                # Неопределенно-личные ROOT-VERB
                                self.Memo[SyntaxVectorizer.SYNTAX_SENTENCE_BASED_LEVEL]['UndefinetelyPersonal'] += 1
                        if ('Mood' in morphoDictRes):
                            if (morphoDictRes['Number'] in ['Plur', 'Sing'] and morphoDictRes['Mood'] == 'Imp'):
                                # Определенно личные (обобщенно-личные)
                                self.Memo[SyntaxVectorizer.SYNTAX_SENTENCE_BASED_LEVEL]['DefinetelyPersonal'] += 1
                        if (('Person' in morphoDictRes) and ('Tense' in morphoDictRes)):
                            if (morphoDictRes['Person'] == '3' and
                                morphoDictRes['Number'] == 'Plur' and
                                (morphoDictRes['Tense'] == 'Pres' or morphoDictRes['Tense'] == 'Fut')):
                                # Неопределенно личные
                                self.Memo[SyntaxVectorizer.SYNTAX_SENTENCE_BASED_LEVEL]['UndefinetelyPersonal'] += 1
                        if (('Mood' in morphoDictRes) and ('Person' in morphoDictRes) and ('Tense' in morphoDictRes)):
                            if ((morphoDictRes['Mood'] == 'Ind') and
                                    (morphoDictRes['Person'] in ['1', '2']) and
                                    (morphoDictRes['Number'] in ['Sing', 'Plur'])):
                                # Определенно личные (обобщенно-личные)
                                self.Memo[SyntaxVectorizer.SYNTAX_SENTENCE_BASED_LEVEL]['DefinetelyPersonal'] += 1
                    if ('VerbForm' in morphoDictRes):
                        if morphoDictRes['VerbForm'] == 'Inf':
                            # Инфинитивные
                            self.Memo[SyntaxVectorizer.SYNTAX_SENTENCE_BASED_LEVEL]['Infinitive'] += 1
                if current_token.pos_ == 'ADV':
                    # Безличные
                    self.Memo[SyntaxVectorizer.SYNTAX_SENTENCE_BASED_LEVEL]['UnPersonal'] += 1
                if current_token.pos_ == 'NOUN':
                    morphoDictRes = processTag(current_token.tag_)
                    # Номинативные
                    if 'Case' in morphoDictRes:
                        if morphoDictRes['Case'] in ['Nom']:
                            self.Memo[SyntaxVectorizer.SYNTAX_SENTENCE_BASED_LEVEL]['Nominative'] += 1


from preprocessor import load_pickled_file
from preprocessor import PATH_TO_RUS_LANG_MODEL

import spacy

if __name__ == "__main__":
    # Just checking
    # Please, pay attention, that this class imported

    t = load_pickled_file('.\\ProcessedTexts\\Горький_Старуха Изергиль.pkl')
    print('Pickle loaded')
    ru_nlp = spacy.load(PATH_TO_RUS_LANG_MODEL)  # Загрузим языковую модель
    print('Model loaded')
    hj = SyntaxVectorizer(ru_nlp)
    hj.convert_to_attributes(t['Trees'])

