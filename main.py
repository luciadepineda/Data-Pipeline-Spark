"""
main.py

The Main code chains the three pipelines (Data Management, Data Analysis and
RunTime Classifier), performing all the steps from the beginning to the end.
It extracts the data, trains the model, and then is able to classify records.
For more specific information about a particular pipeline, refer to its Python script.

Authors: Lucía De Pineda & Héctor Fortuño

"""

import os
import sys
import pyspark
from pyspark import SparkConf
from pyspark.sql import SparkSession
import pyspark.sql.functions as f
from pyspark.sql.types import StructType, StructField, StringType, DoubleType, IntegerType
from time import time
from datetime import datetime
from pyspark.ml import Pipeline, PipelineModel
from pyspark.ml.classification import DecisionTreeClassifier
from pyspark.ml.feature import StringIndexer, VectorIndexer, VectorAssembler
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from Data_Management import data_management
from Data_Analysis import data_analysis
from RunTime_Classifier import runtime_classifier

def main():
    ##### Data Management pipeline 
    # Create data matrix
    matrix = data_management()

    ##### Data Analysis pipeline 
    # Create trained model
    model = data_analysis(matrix)

    ##### RunTime classifier pipeline
    # Read arguments and make a prediction
    parser = argparse.ArgumentParser()
    parser.add_argument('--aircraft', required=True, default=None, type=str, help='Identifier of the aircraft with format XX-XXX')
    parser.add_argument('--date', required=True, default=None, type=str, help='Date with format dd-mm-yy')

    args = parser.parse_args()

    # Create the corresponding variables
    aircraft = args.aircraft
    date = args.date

    # Check that the date is in the correct format
    try:
    	timeid = datetime.strptime(date, '%d-%m-%y').date()
    except:
    	raise Exception('Format of date is incorrect')

    # Obtain the prediction
    prediction = runtime_classifier(aircraft, date, model)

main()
