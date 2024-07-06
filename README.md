# SMS Spam Detector

## Overview

The SMS Spam Detector is a machine learning project designed to classify SMS messages as spam or ham (not spam). This project leverages natural language processing (NLP) 
techniques and various machine learning algorithms to effectively identify and filter out spam messages.

## Features

*	**Text Preprocessing:** Tokenization, stop word removal, stemming, and other preprocessing techniques to clean and prepare the text data.
*	**Model Training:** Utilizes multiple machine learning models such as Naive Bayes, SVM, and Random Forest for training.
*	**Evaluation:** Performance evaluation using metrics like accuracy, precision, recall, and F1-score.
*	**Prediction:** Allows for the classification of new SMS messages.
  
## Dataset

The dataset used for training and evaluation is included in the data/ directory. Ensure it is correctly formatted before running the scripts. It contains a collection
of 5,280 SMS messages, labeled as spam or ham.

The dataset file (sms.csv) contains two columns:
*	**target:** Indicates whether the message is spam (1) or ham (0).
*	**Message:** The actual text content of the SMS message.

## Requirements

To run the project, you need the following dependencies:
*	Python 3.x
*	pandas
*	numpy
*	scikit-learn
*	nltk (Natural Language Toolkit)
*	matplotlib
  
## Usage

1.	Clone the repository or download the project files.
2.	Place the **sms.csv** file in the project directory.
3.	Run the **spam_detecter.ipynb** script to train and evaluate the spam detection model.
4.	The script will load the dataset, preprocess the text data, and train a machine learning model using the TF-IDF (Term Frequency-Inverse Document Frequency) technique.
5.	After training, the model will be evaluated on a holdout set and the performance metrics (such as accuracy, precision, recall, and F1-score) will be displayed.
6.	Finally, you can use the trained model to predict the label (spam/ham) of new SMS messages by modifying the predict function in the script.

## Results

The trained model achieved an accuracy of 98 % and Precision is 100 % on the test set and performed well in terms of precision, recall, and F1-score.
   
## Contact

For any questions or feedback, please reach out to Diksha at dikshamaurya72@gmail.com.




