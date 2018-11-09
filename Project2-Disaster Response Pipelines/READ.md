# Disaster Response Pipelines

--------------------------------------
1. [Installation](#installation)
2. [Project Motivation](#motivation)
3. [File Descriptions](#files)
4. [Results](#results)

## 1. Installation <a name="installation"></a>  
- nltk(3.3)
- sklearn(0.19.2)
- The code should run with no issues using Python versions 3.*.

## 2. Project Motivation <a name="motivation"></a>  

Create a machine learning pipeline to categorize these events so that users can send the messages to an appropriate disaster relief agency.

## 3. File Descriptions <a name="files"></a>   

> * **data/disaster_messages.csv data/disaster_categories.csv :** original data
> * **data/process_data.py:** to run ETL pipeline that cleans data and stores in database
> * **data/DisasterResponse.db:** database that stores cleaned data 
> * **models/train_classifier.py:** to run ML pipeline that trains classifier and saves
> * **models/classifier.pkl:** a pickle file which saves model
> * **data/:** a Flask framework for presenting data

## 4. Results <a name="results"></a>  
- You can see visualizations about data in the web app.
- The web app can use the trained model to input text and return classification results  
#### **How to open the web app?**
Run the following command in the app's directory to run your web app.  
    `python run.py`
Then go to http://0.0.0.0:3001/

