# Disaster Response Pipelines

--------------------------------------
1. [Installation](#installation)
2. [Project Motivation](#motivation)
3. [File Descriptions](#files)
4. [Results](#results)

## 1. Installation <a name="installation"></a>  
- nltk(3.3)
- scikit-learn(0.19.1)
- plotly(3.3.0)
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
- The web app shows visualizations about data.
- The web app can use the trained model to input text and return classification results  
#### **How to run the web app?**
Before run the web app:
> 1. Run the following commands in the project's root directory to set up your database and model.
> 
>     - To run ETL pipeline that cleans data and stores in database
>         `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
>     - To run ML pipeline that trains classifier and saves
>         `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

Run the following command in the app's directory to run the web app.  
>     `python run.py`
Then go to http://0.0.0.0:3001/

