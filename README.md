# Authorship Attribution in Russian with New High-Performing and Fully Interpretable Morpho-Syntactic Features

This project contain implementation the system, proposed in the "Authorship Attribution in Russian with New High-Performing and Fully Interpretable Morpho-Syntactic Features" (E. Pimonova, O. Durandin, A. Malafeev), and presented in the AIST-2019 Conference

Structure of project:
* Texts should be placed into "Texts" (we didn't shared this folder due to licence restrictions).
* Result of processing dependency trees with rules will be placed in "OutData" folder (represent in repository).

Initially you should place target texts into Text folder (name format: "Author_Title.txt", i.e. Author must be separated from title with subscript _ ).
Script _preprocessor.py_ processed all text files from "Text" folder, through SpaCy analyzer. Please, pay attention that we use Model of Russian Language from https://github.com/buriy/spacy-ru

Dependency trees will be saved into pickle format and will be saved in _ProcessedTexts_ folder. 
NB: Change value of variable **PATH_TO_RUS_LANG_MODEL** with path to spaCy model of russian language.

**datasetBuilder.py** read pickle files from _ProcessedTexts_ and proceed it through **processor.py** (that implement morpho and syntactic rules). **datasetBuilder.py** build a few csv files, that contain different text representations (simple/complex morphology, simple/complex syntax, see the paper for additional info).

In the first step you must to run **preprocessor.py** script to build pickle files. Another runs of this script will check if file already processed. Only new files will be processed.
