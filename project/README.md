# Project Overview

Sentiment analysis is a popular topic in Natural Language Processing (NLP). It measures the attitude of customers towards products or services. Tremendous comments and reviews are generated every day and it is impossible to analyze them manually. Automated sentiment analysis with high accuracy help companies with decision making the product improvement. In this work, we will compare different approaches of sentiment analysis on Yelp Review Polarity [1] data set. Methods compared include logistic regression (TF-IDF + LR), Bag of words (BOW), Recurrent Neural Network (LSTM, bi-LSTM), Attention model (LSTM + attention) and Transformer (BERT).

# Description of Submitted Files
There are six folders containing the implemented models:

* __logisitic\_regression__: Run _log\_reg.py_ to perform logistic regression  
* __BERT__: A finished copy is named as _BERT.ipynb_ and exported as pdf with name _BERT.pdf_  
* __bow, lstm, bi\_lstm, att\_lstm__: Each contains a *\_model.py, \_sentiment.py and run\_lstm.pbs. *\_model.py contain the model structure and \_sentiment.py contain the training and testing process. 


 Following files are related to data preprocessing:


* _preprocess.py_ would prepare the data, transform the tokens to ids and save for later use.
* _save\_data.py_ would pad the data prepared by _preprocess.py_ according to the desired sequence_length and save them to numpy array.

# Execution Instructions

*  _log\_reg.py_ could be run directly on Blue Waters.
* _BERT.ipynb_ could be run on Google Colab.
*  To run the rest models, One may login to computer node on Blue Waters and enter module load python/2.0.1. Then, one need to run _preprocess.py_ and _save\_data.py_ to prepare the data. Data will be saved to the folder __preprocessed\_data__. Finally, one can either submit jobs using run\_lstm.pbs or directly run *\_sentiment.py after you cd to the folder of different models.






