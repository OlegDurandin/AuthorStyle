
import nltk
from collections import defaultdict, Counter
from itertools import combinations
from .processor import processTag
from tqdm import tqdm
import os

REAL_MODALITY = 'real modality'
PASS_VOICE_VALUE = 'pass voice'
ACTIVE_VOICE_VALUE = 'active voice'

PAST_TENSE_VALUE = 'past tense'
PRESENT_TENSE_VALUE = 'present tense'
#FUTURE_TENSE_VALUE = 'future tense'

class SyntaxVectorizerEN:

    MORPHO_TAGS = 'MorphoTags'
    SYNTAX_DEP_TAGS = 'SyntaxDependencies'
    BIGRAMS_TREELETS = 'BigramTreelet'
    TRIGRAMS_TREELETS = 'TrigramTreelet'

    COMPLEX_MORPHO_FEATURES_EXTERNAL = 'ExtendedMorpho'
    COMPLEX_SYNTAX_FEATURES = 'SyntaxComplex'
    SYNTAX_DEP_AUX_PHRASE_BASED = 'AuxPhraseBasedLevelSyntax'
    SYNTAX_PHRASE_BASED_LEVEL = 'PhraseBasedLevelSyntax'
    SYNTAX_SENTENCE_BASED_LEVEL = 'SentenceBasedLevelSyntax'

    COMPLEX_MORPHO_FEATURES = 'ComplexMorphoFeatures'
    PRONOMINAL_REPLACEMENT = 'pronominal_replacement'
    ACTION_FEATURE = 'action feature'
    DESCRIPTIVENESS = 'descriptiveness'  # Описательность
    ACTION_DESCRIPTIVENESS = 'action descriptiveness'
    NUMBERNESS = 'number'  # численность
    DYNAMISM = 'dynamism'

    def __init__(self, loaded_spacy_model):
        self.ru_nlp = loaded_spacy_model  # Загрузим языковую модель
        self.total_sentences = 0

    def setup_rules(self):
        self.token_based_functions = [
            (
                SyntaxVectorizerEN.MORPHO_TAGS,
                self.calculate_morpho_tags,
                self.normalize_morpho_tags
            ),
            (
                SyntaxVectorizerEN.SYNTAX_DEP_TAGS,
                self.calculate_syntax_tags,
                self.normalize_syntax_tags
            ),
            (
                SyntaxVectorizerEN.BIGRAMS_TREELETS,
                self.calculate_bigram_treelet,
                self.normalize_bigram_treelet
            ),
            # (
            #     SyntaxVectorizerEN.TRIGRAMS_TREELETS,
            #     self.calculate_trigram_treelet,
            #     self.normalize_trigram_treelet
            # ),
        ]
        self.sentence_based_functions = [
            (
                SyntaxVectorizerEN.COMPLEX_MORPHO_FEATURES_EXTERNAL,
                self.calculate_complex_Morpho,
                self.normalize_complex_morpho),
            (
                SyntaxVectorizerEN.COMPLEX_SYNTAX_FEATURES,
                self.calculate_complex_syntax,
                self.normalize_complex_syntax)
        ]


    def text_structures_initializer(self):
        self.Attributes = {}
        self.Attributes[SyntaxVectorizerEN.MORPHO_TAGS] = Counter()
        self.Attributes[SyntaxVectorizerEN.SYNTAX_DEP_TAGS] = Counter()
        self.Attributes[SyntaxVectorizerEN.BIGRAMS_TREELETS] = Counter()
        self.Attributes[SyntaxVectorizerEN.TRIGRAMS_TREELETS] = Counter()
        self.Attributes[SyntaxVectorizerEN.COMPLEX_MORPHO_FEATURES_EXTERNAL] = Counter()
        self.Attributes[SyntaxVectorizerEN.COMPLEX_SYNTAX_FEATURES] = Counter()

        self.Memo = {}
        self.Memo[SyntaxVectorizerEN.COMPLEX_MORPHO_FEATURES] = Counter()
        self.Memo[SyntaxVectorizerEN.SYNTAX_DEP_AUX_PHRASE_BASED] = Counter()
        self.Memo[SyntaxVectorizerEN.SYNTAX_PHRASE_BASED_LEVEL] = Counter()
        self.Memo[SyntaxVectorizerEN.SYNTAX_SENTENCE_BASED_LEVEL] = Counter()

        self.total_sentences = 0

    def calculate_morpho_tags(self, current_token):
        self.Attributes[SyntaxVectorizerEN.MORPHO_TAGS][current_token.pos_] += 1
    def normalize_morpho_tags(self):
        total = sum(self.Attributes[SyntaxVectorizerEN.MORPHO_TAGS].values(), 0.0)
        for key in self.Attributes[SyntaxVectorizerEN.MORPHO_TAGS]:
            self.Attributes[SyntaxVectorizerEN.MORPHO_TAGS][key] /= total

    def calculate_syntax_tags(self, current_token):
        self.Attributes[SyntaxVectorizerEN.SYNTAX_DEP_TAGS][current_token.dep_] += 1
    def normalize_syntax_tags(self):
        total = sum(self.Attributes[SyntaxVectorizerEN.SYNTAX_DEP_TAGS].values(), 0.0)
        for key in self.Attributes[SyntaxVectorizerEN.SYNTAX_DEP_TAGS]:
            self.Attributes[SyntaxVectorizerEN.SYNTAX_DEP_TAGS][key] /= total

    def calculate_bigram_treelet(self, current_token):
        if current_token.dep != 'punct':
            self.Attributes[SyntaxVectorizerEN.BIGRAMS_TREELETS][(current_token.head.pos_, current_token.dep_, current_token.pos_)] += 1
    def normalize_bigram_treelet(self):
        total = sum(self.Attributes[SyntaxVectorizerEN.BIGRAMS_TREELETS].values(), 0.0)
        for key in self.Attributes[SyntaxVectorizerEN.BIGRAMS_TREELETS]:
            self.Attributes[SyntaxVectorizerEN.BIGRAMS_TREELETS][key] /= total

    def calculate_trigram_treelet(self, current_token):
        for (left_child, right_child) in combinations(current_token.children, 2):
            if left_child.dep_ != 'punct' and right_child.dep_ != 'punct':
                self.Attributes[SyntaxVectorizerEN.TRIGRAMS_TREELETS][
                    (1, left_child.pos_, left_child.dep_, current_token.pos_, right_child.dep_, right_child.pos_)] += 1
        if current_token.dep_ != 'ROOT':
            for child in current_token.children:
                if child.dep_ == 'punct' or current_token.dep_ == 'punct':
                    continue
                self.Attributes[SyntaxVectorizerEN.TRIGRAMS_TREELETS][
                    (2, child.pos_, child.dep_, current_token.pos_, current_token.dep_, current_token.head.pos_)] += 1
    def normalize_trigram_treelet(self):
        # Вопрос по нормализации: должны ли они быть независимы друг от друга
        total = sum(self.Attributes[SyntaxVectorizerEN.TRIGRAMS_TREELETS].values(), 0.0)
        for key in self.Attributes[SyntaxVectorizerEN.TRIGRAMS_TREELETS]:
            self.Attributes[SyntaxVectorizerEN.TRIGRAMS_TREELETS][key] /= total

    def calculate_complex_Morpho(self, parsed_sentence):
        for current_token in parsed_sentence:
            morphoDictRes = processTag(current_token.tag_)

            # real modality = кол-во глаголов с категорией Mood=Ind / кол-во глаголов
            if current_token.pos_ == 'VERB':
                if ('Mood' in morphoDictRes) and ('Ind' == morphoDictRes['Mood']):
                    self.Memo[SyntaxVectorizerEN.COMPLEX_MORPHO_FEATURES][REAL_MODALITY] += 1
                if ('Voice' in morphoDictRes):
                    if 'Pass' == morphoDictRes['Voice']:
                        self.Memo[SyntaxVectorizerEN.COMPLEX_MORPHO_FEATURES][PASS_VOICE_VALUE] += 1
                    elif 'Act' == morphoDictRes['Voice']:
                        self.Memo[SyntaxVectorizerEN.COMPLEX_MORPHO_FEATURES][ACTIVE_VOICE_VALUE] += 1
            # passive = кол-во глаголов с категорией Voice=Pass
            # кол-во глаголов с категорией Voice=Pass + кол-во глаголов с категорией Voice=Act
                if ('Tense' in morphoDictRes):
                    if 'Pres' == morphoDictRes['Tense']:
                        self.Memo[SyntaxVectorizerEN.COMPLEX_MORPHO_FEATURES][PRESENT_TENSE_VALUE] += 1
                    elif 'Past' == morphoDictRes['Tense']:
                        self.Memo[SyntaxVectorizerEN.COMPLEX_MORPHO_FEATURES][PAST_TENSE_VALUE] += 1
                    # elif 'Future' == morphoDictRes['Tense']:
                    #     self.Memo[SyntaxVectorizerEN.COMPLEX_MORPHO_FEATURES][FUTURE_TENSE_VALUE] += 1
            # - present_tense = кол-во глаголов с категорией Tense=Pres / кол-во глаголов с категорией Tense=Pres + кол-во глаголов с категорией Tense=Past + кол-во глаголов с категорией Tense=Fut

        pass

    def normalize_complex_morpho(self):
        total = sum(self.Attributes[SyntaxVectorizerEN.MORPHO_TAGS].values(), 0.0)


        # местоименная замена
        self.Attributes[SyntaxVectorizerEN.COMPLEX_MORPHO_FEATURES_EXTERNAL][SyntaxVectorizerEN.PRONOMINAL_REPLACEMENT] = self.Attributes[SyntaxVectorizerEN.MORPHO_TAGS]['PRON'] / (self.Attributes[SyntaxVectorizerEN.MORPHO_TAGS]['PRON'] + self.Attributes[SyntaxVectorizerEN.MORPHO_TAGS]['NOUN'])
        # признак действия
        self.Attributes[SyntaxVectorizerEN.COMPLEX_MORPHO_FEATURES_EXTERNAL][SyntaxVectorizerEN.ACTION_FEATURE] = self.Attributes[SyntaxVectorizerEN.MORPHO_TAGS]['ADV'] / (
                            self.Attributes[SyntaxVectorizerEN.MORPHO_TAGS]['ADV'] +
                            self.Attributes[SyntaxVectorizerEN.MORPHO_TAGS]['ADJ'] +
                            self.Attributes[SyntaxVectorizerEN.MORPHO_TAGS]['DET']
        )

        # описательность
        self.Attributes[SyntaxVectorizerEN.COMPLEX_MORPHO_FEATURES_EXTERNAL][SyntaxVectorizerEN.DESCRIPTIVENESS] = self.Attributes[SyntaxVectorizerEN.MORPHO_TAGS]['ADJ'] / self.Attributes[SyntaxVectorizerEN.MORPHO_TAGS]['NOUN']

        # описательность действия
        self.Attributes[SyntaxVectorizerEN.COMPLEX_MORPHO_FEATURES_EXTERNAL][SyntaxVectorizerEN.ACTION_DESCRIPTIVENESS] = self.Attributes[SyntaxVectorizerEN.MORPHO_TAGS]['ADV'] / self.Attributes[SyntaxVectorizerEN.MORPHO_TAGS]['VERB']

        self.Attributes[SyntaxVectorizerEN.COMPLEX_MORPHO_FEATURES_EXTERNAL][SyntaxVectorizerEN.NUMBERNESS] = self.Attributes[SyntaxVectorizerEN.MORPHO_TAGS]['NUM'] / self.Attributes[SyntaxVectorizerEN.MORPHO_TAGS]['NOUN']  # численность

        self.Attributes[SyntaxVectorizerEN.COMPLEX_MORPHO_FEATURES_EXTERNAL][SyntaxVectorizerEN.DYNAMISM]  = self.Attributes[SyntaxVectorizerEN.MORPHO_TAGS]['VERB'] / (total - (
                    self.Attributes[SyntaxVectorizerEN.MORPHO_TAGS]['ADP'] +
                    self.Attributes[SyntaxVectorizerEN.MORPHO_TAGS]['CCONJ'] +
                    self.Attributes[SyntaxVectorizerEN.MORPHO_TAGS]['PART'] +
                    self.Attributes[SyntaxVectorizerEN.MORPHO_TAGS]['PUNCT'] +
                    self.Attributes[SyntaxVectorizerEN.MORPHO_TAGS]['SCONJ'] +
                    self.Attributes[SyntaxVectorizerEN.MORPHO_TAGS]['SYM']))  # динамичность

        self.Attributes[SyntaxVectorizerEN.COMPLEX_MORPHO_FEATURES_EXTERNAL][REAL_MODALITY] = self.Memo[SyntaxVectorizerEN.COMPLEX_MORPHO_FEATURES][REAL_MODALITY] / self.Attributes[SyntaxVectorizerEN.MORPHO_TAGS]['VERB']
        try:
            self.Attributes[SyntaxVectorizerEN.COMPLEX_MORPHO_FEATURES_EXTERNAL][PASS_VOICE_VALUE] = self.Memo[SyntaxVectorizerEN.COMPLEX_MORPHO_FEATURES][PASS_VOICE_VALUE] / self.Attributes[SyntaxVectorizerEN.MORPHO_TAGS]['VERB']
        except ZeroDivisionError:
            self.Attributes[SyntaxVectorizerEN.COMPLEX_MORPHO_FEATURES_EXTERNAL][PASS_VOICE_VALUE] = 0

        total_tense = self.Memo[SyntaxVectorizerEN.COMPLEX_MORPHO_FEATURES][PRESENT_TENSE_VALUE] + \
                      self.Memo[SyntaxVectorizerEN.COMPLEX_MORPHO_FEATURES][PAST_TENSE_VALUE]
#                     + self.Memo[SyntaxVectorizerEN.COMPLEX_MORPHO_FEATURES][FUTURE_TENSE_VALUE]
        if total_tense != 0:
            self.Attributes[SyntaxVectorizerEN.COMPLEX_MORPHO_FEATURES_EXTERNAL][PAST_TENSE_VALUE] = \
                self.Memo[SyntaxVectorizerEN.COMPLEX_MORPHO_FEATURES][PAST_TENSE_VALUE] / total_tense
            self.Attributes[SyntaxVectorizerEN.COMPLEX_MORPHO_FEATURES_EXTERNAL][PRESENT_TENSE_VALUE] = \
                self.Memo[SyntaxVectorizerEN.COMPLEX_MORPHO_FEATURES][PRESENT_TENSE_VALUE] / total_tense
        else:
            self.Attributes[SyntaxVectorizerEN.COMPLEX_MORPHO_FEATURES_EXTERNAL][PAST_TENSE_VALUE] = 0
            self.Attributes[SyntaxVectorizerEN.COMPLEX_MORPHO_FEATURES_EXTERNAL][PRESENT_TENSE_VALUE] = 0
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
        total_count_all_dependencies = sum([self.Memo[SyntaxVectorizerEN.SYNTAX_DEP_AUX_PHRASE_BASED][x] for x in self.Memo[SyntaxVectorizerEN.SYNTAX_DEP_AUX_PHRASE_BASED]])
        coordination_tags = ['nsubj', 'csubj', 'expl', 'csubj:pass', 'nsubj:pass']  # Координация
        agreement_tags = ['nmod', 'appos', 'acl', 'amod', 'det', 'nummod', 'compound']  # Согласование
        control_tags = ['obj', 'iobj', 'ccomp', 'obl']  # Управление
        contiguity_tags = ['advmod']  # Примыкание

        # Классификация по типу связи: Координация
        self.Attributes[SyntaxVectorizerEN.COMPLEX_SYNTAX_FEATURES]['CoordinationPB'] = sum(
            [self.Memo[SyntaxVectorizerEN.SYNTAX_DEP_AUX_PHRASE_BASED].get(x, 0) for x in coordination_tags]) / total_count_all_dependencies
        # Классификация по типу связи: Согласование
        self.Attributes[SyntaxVectorizerEN.COMPLEX_SYNTAX_FEATURES]['AgreementPB'] = sum(
            [self.Memo[SyntaxVectorizerEN.SYNTAX_DEP_AUX_PHRASE_BASED].get(x, 0) for x in agreement_tags]) / total_count_all_dependencies
        # Классификация по типу связи: Управление
        self.Attributes[SyntaxVectorizerEN.COMPLEX_SYNTAX_FEATURES]['ControlPB'] = sum(
            [self.Memo[SyntaxVectorizerEN.SYNTAX_DEP_AUX_PHRASE_BASED].get(x, 0) for x in control_tags]) / total_count_all_dependencies
        # Классификация по типу связи: Примыкание
        self.Attributes[SyntaxVectorizerEN.COMPLEX_SYNTAX_FEATURES]['ContiguityPB'] = sum(
            [self.Memo[SyntaxVectorizerEN.SYNTAX_DEP_AUX_PHRASE_BASED].get(x, 0) for x in contiguity_tags]) / total_count_all_dependencies

        # Количественно-структурные типы
        self.Attributes[SyntaxVectorizerEN.COMPLEX_SYNTAX_FEATURES]['comphr'] = self.Memo[SyntaxVectorizerEN.SYNTAX_PHRASE_BASED_LEVEL]['comphr'] / total_count_all_dependencies
        self.Attributes[SyntaxVectorizerEN.COMPLEX_SYNTAX_FEATURES]['simlphr'] = self.Memo[SyntaxVectorizerEN.SYNTAX_PHRASE_BASED_LEVEL]['simlphr'] / total_count_all_dependencies

        # Синтаксически-несвободные словосочетания
        syntactically_non_free_phrases_tags = ['fixed', 'flat', 'compound']
        self.Attributes[SyntaxVectorizerEN.COMPLEX_SYNTAX_FEATURES]['syntaxNonFreePhrases'] = sum([
            self.Memo[SyntaxVectorizerEN.SYNTAX_DEP_AUX_PHRASE_BASED].get(x, 0) for x in syntactically_non_free_phrases_tags]) / total_count_all_dependencies
        # Лексико-грамматические типы
        self.Attributes[SyntaxVectorizerEN.COMPLEX_SYNTAX_FEATURES]['nomphr'] = self.Memo[SyntaxVectorizerEN.SYNTAX_PHRASE_BASED_LEVEL]['nomphr'] / total_count_all_dependencies  # именной тип
        self.Attributes[SyntaxVectorizerEN.COMPLEX_SYNTAX_FEATURES]['verbphr'] = self.Memo[SyntaxVectorizerEN.SYNTAX_PHRASE_BASED_LEVEL]['verbphr'] / total_count_all_dependencies  # глагольный тип
        self.Attributes[SyntaxVectorizerEN.COMPLEX_SYNTAX_FEATURES]['advphr'] = self.Memo[SyntaxVectorizerEN.SYNTAX_PHRASE_BASED_LEVEL]['advphr'] / total_count_all_dependencies  # наречный тип

        # Членимые и нечленимые предложения
        self.Attributes[SyntaxVectorizerEN.COMPLEX_SYNTAX_FEATURES]['NotSeparatedSentences'] = self.Memo[SyntaxVectorizerEN.SYNTAX_SENTENCE_BASED_LEVEL]['NotseparatedSentence'] / self.total_sentences
        self.Attributes[SyntaxVectorizerEN.COMPLEX_SYNTAX_FEATURES]['Vocative'] = self.Memo[SyntaxVectorizerEN.SYNTAX_SENTENCE_BASED_LEVEL]['Vocative'] / self.total_sentences
        #self.Attributes[SyntaxVectorizerEN.COMPLEX_SYNTAX_FEATURES]['Genitive'] = self.Memo[SyntaxVectorizerEN.SYNTAX_SENTENCE_BASED_LEVEL]['Genitive'] / self.total_sentences

        # Односоставные и двусоставные слова
        # Односоставные
        #self.Attributes[SyntaxVectorizerEN.COMPLEX_SYNTAX_FEATURES]['SingleCompose'] = self.Memo[SyntaxVectorizerEN.SYNTAX_SENTENCE_BASED_LEVEL]['Single'] / self.total_sentences
        # Определенно-личные
        #self.Attributes[SyntaxVectorizerEN.COMPLEX_SYNTAX_FEATURES]['DefinetelyPersonal'] = self.Memo[SyntaxVectorizerEN.SYNTAX_SENTENCE_BASED_LEVEL]['DefinetelyPersonal'] / self.total_sentences
        # Неопределенно-личные
        #self.Attributes[SyntaxVectorizerEN.COMPLEX_SYNTAX_FEATURES]['UndefinetelyPersonal'] = self.Memo[SyntaxVectorizerEN.SYNTAX_SENTENCE_BASED_LEVEL]['UndefinetelyPersonal'] / self.total_sentences

        # Инфинитивные
        # self.Attributes[SyntaxVectorizerEN.COMPLEX_SYNTAX_FEATURES]['Infinitive'] = self.Memo[SyntaxVectorizerEN.SYNTAX_SENTENCE_BASED_LEVEL]['Infinitive'] / self.total_sentences
        # Безличные
        #self.Attributes[SyntaxVectorizerEN.COMPLEX_SYNTAX_FEATURES]['UnPersonal'] = self.Memo[SyntaxVectorizerEN.SYNTAX_SENTENCE_BASED_LEVEL]['UnPersonal'] / self.total_sentences
        # Номинативные
        self.Attributes[SyntaxVectorizerEN.COMPLEX_SYNTAX_FEATURES]['Nominative'] = self.Memo[SyntaxVectorizerEN.SYNTAX_SENTENCE_BASED_LEVEL]['Nominative'] / self.total_sentences

        # Осложненные конструкции
        self.Attributes[SyntaxVectorizerEN.COMPLEX_SYNTAX_FEATURES]['IntroConstruction'] = self.Memo[SyntaxVectorizerEN.SYNTAX_SENTENCE_BASED_LEVEL]['IntroConstruction'] / self.total_sentences
        self.Attributes[SyntaxVectorizerEN.COMPLEX_SYNTAX_FEATURES]['InterjectionConstruction']= self.Memo[SyntaxVectorizerEN.SYNTAX_SENTENCE_BASED_LEVEL]['InterjectionConstruction'] / self.total_sentences
        self.Attributes[SyntaxVectorizerEN.COMPLEX_SYNTAX_FEATURES]['AppealConstruction'] = self.Memo[SyntaxVectorizerEN.SYNTAX_SENTENCE_BASED_LEVEL]['AppealConstruction'] / self.total_sentences
        self.Attributes[SyntaxVectorizerEN.COMPLEX_SYNTAX_FEATURES]['InsertConstruction'] = self.Memo[SyntaxVectorizerEN.SYNTAX_SENTENCE_BASED_LEVEL]['InsertConstruction'] / self.total_sentences

        self.Attributes[SyntaxVectorizerEN.COMPLEX_SYNTAX_FEATURES]['AdjTurnover'] = self.Memo[SyntaxVectorizerEN.SYNTAX_SENTENCE_BASED_LEVEL]['AdjTurnover'] / self.total_sentences
        self.Attributes[SyntaxVectorizerEN.COMPLEX_SYNTAX_FEATURES]['verbalAdverb'] = self.Memo[SyntaxVectorizerEN.SYNTAX_SENTENCE_BASED_LEVEL]['verbalAdverb'] / self.total_sentences
        self.Attributes[SyntaxVectorizerEN.COMPLEX_SYNTAX_FEATURES]['participle'] = self.Memo[SyntaxVectorizerEN.SYNTAX_SENTENCE_BASED_LEVEL]['participle'] / self.total_sentences
        self.Attributes[SyntaxVectorizerEN.COMPLEX_SYNTAX_FEATURES]['ComplicatedByHomogeneous'] = self.Memo[SyntaxVectorizerEN.SYNTAX_SENTENCE_BASED_LEVEL]['ComplicatedByHomogeneous'] / self.total_sentences
        self.Attributes[SyntaxVectorizerEN.COMPLEX_SYNTAX_FEATURES]['ComplicatedByApplication'] = self.Memo[SyntaxVectorizerEN.SYNTAX_SENTENCE_BASED_LEVEL]['ComplicatedByApplication'] / self.total_sentences




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
                        self.Memo[SyntaxVectorizerEN.SYNTAX_SENTENCE_BASED_LEVEL]['AdjTurnover'] += 1
                    elif current_token.dep_ in ['advcl']:
                        self.Memo[SyntaxVectorizerEN.SYNTAX_SENTENCE_BASED_LEVEL]['verbalAdverb'] += 1
                    elif current_token.dep_ in ['acl']:
                        self.Memo[SyntaxVectorizerEN.SYNTAX_SENTENCE_BASED_LEVEL]['participle'] += 1


        if 'conj' in [token.dep_ for token in parsed_sentence]:
            # - Если внутри предложения есть отношение conj (не обращаем внимания на знаки препинания) → осложнено однородными членами (complicated)
            self.Memo[SyntaxVectorizerEN.SYNTAX_SENTENCE_BASED_LEVEL]['ComplicatedByHomogeneous'] += 1

        if 'appos' in [token.dep_ for token in parsed_sentence]:
            #  Если есть отношение appos (не обращаем внимания на знаки препинания) → осложнено приложением
            self.Memo[SyntaxVectorizerEN.SYNTAX_SENTENCE_BASED_LEVEL]['ComplicatedByApplication'] += 1


        for current_token in parsed_sentence:
            if current_token.dep_ != '' and current_token.pos_ != 'SPACE':
                self.Memo[SyntaxVectorizerEN.SYNTAX_DEP_AUX_PHRASE_BASED][current_token.dep_] += 1
            else:
                continue

            # Количество потомков
            childCount = len(list(current_token.children))
            if childCount > 1:
                self.Memo[SyntaxVectorizerEN.SYNTAX_PHRASE_BASED_LEVEL]['comphr'] += 1
            else:
                self.Memo[SyntaxVectorizerEN.SYNTAX_PHRASE_BASED_LEVEL]['simlphr'] += 1

            if  current_token.dep_ == 'parataxis':
                list_of_childs = [x.pos_ for x in current_token.children]
                if 'ADV' in list_of_childs:           # Вводная конструкция
                    self.Memo[SyntaxVectorizerEN.SYNTAX_SENTENCE_BASED_LEVEL]['IntroConstruction'] += 1
                elif 'INTJ' in list_of_childs:        # междометие
                    self.Memo[SyntaxVectorizerEN.SYNTAX_SENTENCE_BASED_LEVEL]['InterjectionConstruction'] += 1
                elif 'PROPN' in list_of_childs:
                    self.Memo[SyntaxVectorizerEN.SYNTAX_SENTENCE_BASED_LEVEL]['AppealConstruction'] += 1
                else:
                    self.Memo[SyntaxVectorizerEN.SYNTAX_SENTENCE_BASED_LEVEL]['InsertConstruction'] += 1

            if childCount > 0:
                if current_token.pos_ in ['NOUN', 'ADJ', 'NUM', 'PRON']:
                    self.Memo[SyntaxVectorizerEN.SYNTAX_PHRASE_BASED_LEVEL]['nomphr'] += 1
                elif current_token.pos_ in ['VERB', 'причастие', 'деепричастие']:
                    self.Memo[SyntaxVectorizerEN.SYNTAX_PHRASE_BASED_LEVEL]['verbphr'] += 1
                elif current_token.pos_ in ['ADV']:
                    self.Memo[SyntaxVectorizerEN.SYNTAX_PHRASE_BASED_LEVEL]['advphr'] += 1
            if current_token.dep_ == 'ROOT':
                if current_token.pos_ in ['INTJ', 'PART']:
                    # Нечленимые предложения
                    self.Memo[SyntaxVectorizerEN.SYNTAX_SENTENCE_BASED_LEVEL]['NotseparatedSentence'] += 1
                elif current_token.pos_ in ['PROPN']:
                    # Вокативы
                    self.Memo[SyntaxVectorizerEN.SYNTAX_SENTENCE_BASED_LEVEL]['Vocative'] += 1
                elif current_token.pos_ in ['NOUN']:
                    # Генитивные
                    morphoDictRes = processTag(current_token.tag_)
                    # if 'Case' in morphoDictRes:
                    #     if morphoDictRes['Case'] in ['Par', 'Gen']:
                    #         self.Memo[SyntaxVectorizerEN.SYNTAX_SENTENCE_BASED_LEVEL]['Genitive'] += 1
                if (('nsubj' not in list(map(lambda x: x.dep_, current_token.children))) or
                    ('csubj' not in list(map(lambda x: x.dep_, current_token.children)))):
                    # Односоставное
                    # В правилох непонятно, выделяется односоставное и среди них определенно-личные
                    # обобщенно-личные, неопределенно-личные, инфинитивные, бюезличные, номинативные и т.д.?
                    self.Memo[SyntaxVectorizerEN.SYNTAX_SENTENCE_BASED_LEVEL]['Single'] += 1
                #else:
                #   print('Non single: {}'.format(parsed_sentence))

                if current_token.pos_ == 'VERB':
                    # Определенно-личные (и обобщенно-личные)
                    morphoDictRes = processTag(current_token.tag_)
                    # if ('Number' in morphoDictRes):
                    #     # if ('Tense' in morphoDictRes):
                    #     #     if morphoDictRes['Number'] == 'Plur' and morphoDictRes['Tense'] == 'Past':
                    #     #         # Неопределенно-личные ROOT-VERB
                    #     #         self.Memo[SyntaxVectorizerEN.SYNTAX_SENTENCE_BASED_LEVEL]['UndefinetelyPersonal'] += 1
                    #     #if ('Mood' in morphoDictRes):
                    #         #if (morphoDictRes['Number'] in ['Plur', 'Sing'] and morphoDictRes['Mood'] == 'Imp'):
                    #             # Определенно личные (обобщенно-личные)
                    #             #self.Memo[SyntaxVectorizerEN.SYNTAX_SENTENCE_BASED_LEVEL]['DefinetelyPersonal'] += 1
                    #     # if (('Person' in morphoDictRes) and ('Tense' in morphoDictRes)):
                    #     #     if (morphoDictRes['Person'] == '3' and
                    #     #         morphoDictRes['Number'] == 'Plur' and
                    #     #         (morphoDictRes['Tense'] == 'Pres' or morphoDictRes['Tense'] == 'Fut')):
                    #     #         # Неопределенно личные
                    #     #         self.Memo[SyntaxVectorizerEN.SYNTAX_SENTENCE_BASED_LEVEL]['UndefinetelyPersonal'] += 1
                    #     if (('Mood' in morphoDictRes) and ('Person' in morphoDictRes) and ('Tense' in morphoDictRes)):
                    #         if ((morphoDictRes['Mood'] == 'Ind') and
                    #                 (morphoDictRes['Person'] in ['1', '2']) and
                    #                 (morphoDictRes['Number'] in ['Sing', 'Plur'])):
                    #             # Определенно личные (обобщенно-личные)
                    #             self.Memo[SyntaxVectorizerEN.SYNTAX_SENTENCE_BASED_LEVEL]['DefinetelyPersonal'] += 1
                    # if ('VerbForm' in morphoDictRes):
                    #     if morphoDictRes['VerbForm'] == 'Inf':
                    #         # Инфинитивные
                    #         self.Memo[SyntaxVectorizerEN.SYNTAX_SENTENCE_BASED_LEVEL]['Infinitive'] += 1
                # if current_token.pos_ == 'ADV':
                #     # Безличные
                #     self.Memo[SyntaxVectorizerEN.SYNTAX_SENTENCE_BASED_LEVEL]['UnPersonal'] += 1
                if current_token.pos_ == 'NOUN':
                    morphoDictRes = processTag(current_token.tag_)
                    # Номинативные
                    if 'Case' in morphoDictRes:
                        if morphoDictRes['Case'] in ['Nom']:
                            self.Memo[SyntaxVectorizerEN.SYNTAX_SENTENCE_BASED_LEVEL]['Nominative'] += 1
