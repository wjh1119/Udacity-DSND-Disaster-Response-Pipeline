# import libraries
import sys
import pandas as pd
from sqlalchemy import create_engine


def load_data(messages_filepath, categories_filepath):
    ''' load data 

    Parameters
    ----------------------
    messages_filepath,categories_filepath: str, pathlib.Path, py._path.local.LocalPath or any \
    object with a read() method (such as a file handle or StringIO)
        The string could be a URL. Valid URL schemes include http, ftp, s3, and
        file. For file URLs, a host is expected. For instance, a local file could
        be file://localhost/path/to/table.csv

    Returns
    -----------------
    result : pd.DataFrame
        
    '''
    # load messages dataset
    messages = pd.read_csv(messages_filepath,index_col='id')

    # load categories dataset
    categories = pd.read_csv(categories_filepath,index_col='id')

    # merge datasets
    df = pd.merge(messages,categories,on='id')
    
    return df


def clean_data(df):
    '''clean data

    Parameters
    -------------------
    df:pd.DataFrame
        data that needs to be cleaned

    Returns
    -------------------
    result : pd.DataFrame
    '''

    # create a dataframe of the 36 individual category columns
    categories = df['categories'].str.split(pat=';',expand=True)

    # select the first row of the categories dataframe
    row = categories.iloc[0,:]

    # use this row to extract a list of new column names for categories.
    # one way is to apply a lambda function that takes everything 
    # up to the second to last character of each string with slicing
    category_colnames = [i.split('-')[0] for i in row]

    # rename the columns of `categories`
    categories.columns = category_colnames

    for column in categories:
        # set each value to be the last character of the string
        categories[column] = categories[column].map(lambda x:x[-1:])
        
        # convert column from string to numeric
        categories[column] = categories[column].map(lambda x:int(x))

    # drop the original categories column from `df`
    df = df.drop(columns=['categories'])

    # concatenate the original dataframe with the new `categories` dataframe
    df = pd.merge(df,categories,on='id')

    # drop duplicates
    df = df.drop_duplicates()

    return df


def save_data(df, database_filename):
    '''save data

    Parameters
    --------------------
    df:pd.DataFrame
        data that needs to be saved
    database_filename: str
        the url of database

    Returns
    -------------------
    None
    '''
    
    engine = create_engine(database_filename)
    df.to_sql('DisasterResponse', engine, index=False, if_exists='replace')


def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)
        
        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)
        
        print('Cleaned data saved to database!')
    
    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()