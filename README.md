# europeanamt-translate-mt
This repository contains a collection of tools for automatic translation. The tools are intended to be used as part of a pipeline for processing text, translate sentences and evaluate the quality of the translations.

The scripts **batcher.py**, **sentencepiece.py** and **onmt_model.py** are related to the training of the machine translation models.

The scripts **europeana_preprocess.py**, **contains_test_val.py**, **english_formalizer_norm.py**, **equals_sentences_val.py** and **spanish__plural_abreviations** show how to normalize and validate the input and output text of the model.

The **get_quality_score.ipynb** script shows how a polynomial regression model can be trained to obtain the quality of translations.

