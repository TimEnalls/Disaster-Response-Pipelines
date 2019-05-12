# Disaster-Response-Pipelines


## Summary of Project
In this project, I analyzed real disaster data provided by Figure Eight to build a model for an API that classifies disaster messages. I created a machine learning pipeline to categorize disaster events so that a users can have their messages sent to an appropriate disaster relief agency.


## How to run the Python scripts and web app
Run the following command in the app's directory to run your web app.
    `python run.py`

Next, go to http://0.0.0.0:3001/

## The Files in the Repository

### App Folder
•	**Run.py** – This Python file contains the following steps: 

  &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;•	Initializes a flask app
  
  &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;•	Tokenizes and normalizes text
  
  &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;•	Loads data from a database
  
  &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;•	Loads a model from “Train_classifer”
  
  &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;•	Returns a website that displays model results
  
•	Template Folder – contains the html files

  •	**Go.html** – contains html code for master.html
  
  •	**Master.html** – allows users to enter messages that are then automatically classified

### Data Folder

•	**Process_data** – This Python file contains the following steps: 

  •	Loads csv data containing category and messages data
  
  •	Cleans the data by splitting the category field and dropping duplicates
  
  •	Saves that data into a database.
  
•	**DisasterResponse.db** – this database fie contains the clean data processed in “process_data.py”

•	**Disaster_categories.csv** – this text file contains a column with concatenates category names

•	**Disaster_messages.csv** – this text file contains messages typed by disaster victims

### Models Folder
•	**Train_classifer** – This Python file contains the following steps: 

•	Tokenizes disaster message text, normalizes that text to lower case, and removes stop words

•	Builds an Adaboost model that uses grid search to optimize it’s hyperparameters

•	Evaluates the model and predicts the categories of messages

•	The trained model is saved as “pickle” into the run.py file

