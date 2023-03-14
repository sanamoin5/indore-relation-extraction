# IndoRE: Relation Extraction for Low Resource Indian Languages
 
This repository contains the scripts and notebooks for the IndoRE-datathon-2021 challenge hosted on Kaggle (https://www.kaggle.com/competitions/indore-datathon-2021/overview). 
Relation extraction is an important and well explored problem in NLP. Given a sentence and a pair of entities from the sentence, it requires to predict the relation between the two.
Despite recent interest in deep generative models, for Indian languages, relation classification is still under explored due to the unavailability of tagged data.

In this challenge we predicted the relation between two entities in a sentence for three Indian languages (Bengali, Telugu and Hindi) from a limited tagged training data. There were 25 relations for each language.

The data folder contains the dataset which was provided along with the one we obtained by distant supervision. The test actual outputs are stored in test_results.csv file.

The results folder contains the outputs, predictions and what we submitted for the competition.

The source_code has notebooks for both mBERT and RoBERTa model for different variants, along with a notebook that does the comparison and analysis of the results. The scripts folder contains additional scripts that were utilized for data processing and model building.
