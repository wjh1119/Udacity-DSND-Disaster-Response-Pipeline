# import libraries
import sys
import pandas as pd
import numpy as np
from sqlalchemy import create_engine
import re

import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet') # download for lemmatization

from nltk.tokenize import word_tokenize,sent_tokenize
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer,TfidfTransformer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

import pickle

stop_words = stopwords.words("english")
lemmatizer = WordNetLemmatizer()

def load_data(database_filepath):
    '''load data from database

    Parameters
    ---------------------
    database_filepath: str
        the url of database

    Returns
    ---------------------
    X: pd.DataFrame
        df which including messages
    Y: pd.DataFrame
        df which including categories
    category_names: list
        names of categories
    '''
    # load data from database
    engine = create_engine('sqlite:///'+database_filename)
    df = pd.read_sql(sql='select * from DisasterResponse',con=engine)
    X = df.message.values
    Y = df.iloc[:,3:]
    category_names = Y.columns
    return X,Y,category_names


def tokenize(text):
    ''' tokenize text

    Parameters
    -----------------------
    text: str
        test which needs to be tokenized

    Returns
    -----------------------
    str: tokenized text
    '''
    # normalize case and remove punctuation
    text = re.sub(r"[^a-zA-Z0-9]", " ", text.lower())
    
    # tokenize text
    tokens = word_tokenize(text)
    
    # lemmatize andremove stop words
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]

    return tokens


def build_model():
    ''' build model

    Returns
    -----------------------
    model: model for prediction 
    '''
    
    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize, max_df=0.75)),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(estimator=RandomForestClassifier(),n_jobs=-1))
    ])
    
    parameters = {
        'vect__max_df': (0.5,0.75),
        'clf__estimator__n_estimators': [25,50],
    }

    cv = GridSearchCV(pipeline,param_grid=parameters,n_jobs=-1,verbose=2)
    return cv


def evaluate_model(model, X_test, Y_test, category_names):
    '''evaluate model

    Parameters
    ------------------
    model: model which needs to be evaluated
    X_test: pd.DataFrame
        df which including messages
    Y_test: pd.DataFrame
        df which including categories
    category_names: list
        names of categories

    '''
    
    y_pred = model.predict(X_test)
    
    # change data's type to DataFrame
    y_pred_df = pd.DataFrame(y_pred,columns=category_names,index=range(y_pred.shape[0]))
    y_test_df = pd.DataFrame(Y_test,columns=category_names,index=range(y_pred.shape[0]))
    y_test_df.fillna(0,inplace=True)
    y_test_df = y_test_df.applymap(lambda x:int(x))

    # for each category
    for column in category_names:
        
        # show f1_score,persicion,recall
        curr_f1_report = classification_report(y_test_df[[column]],y_pred_df[[column]])
        print("*"*80)
        print("f1 score table for '%s' column:\n" %column)
        print("The average f1 score is %.2f" % float(curr_f1_report.splitlines()[-1].split()[5]))
        print(curr_f1_report)


def save_model(model, model_filepath):
    '''save model at model_filepath
    Parameters
    -------------------------
    model: model 
    model_filepath: str
        the url of the file which saves model
    '''

    model_pickle_file = open(model_filepath,'wb')
    pickle.dump(model,model_pickle_file)


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()