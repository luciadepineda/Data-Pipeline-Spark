"""
Data_Analysis.py

The Data Analysis Pipeline uses the data matrix created in the previous pipeline
and trains a classifier model. Specifically, we use the MLlib library to train a
decision tree. 
The steps in the pipeline are:
1- Change of data types to the corresponding ones
2- Use a VectorAssembler to combine all feature data and separate 'label' data
3- Use a Label indexer to index labels (0 or 1)
4- Split of the data into training and validation
5- Train a DecisionTree model with our labels and features
6- Chain indexer and tree in a Pipeline
7- Train model after running the indexer
8- Make predictions
9- Compute metrics of accuracy, recall and precision to validate the model
10- Return the final trained model
The steps are explained in more detail in the corresponding part of the code.

The Data Analysis pipeline is implemented in a function called 'data_analysis',
which takes as argument the data matrix and returns the trained model. This
function is called in the main, creating and training the model and storing
it in the current working directory.

Authors: Lucía De Pineda & Héctor Fortuño

"""

import os
import sys
import pyspark
from pyspark import SparkConf
from pyspark.sql import SparkSession
import pyspark.sql.functions as f
from time import time
from datetime import datetime
from pyspark.ml import Pipeline
from pyspark.ml.classification import DecisionTreeClassifier
from pyspark.ml.feature import StringIndexer, VectorIndexer, VectorAssembler
from pyspark.ml.evaluation import MulticlassClassificationEvaluator

# Configure environment variables
HADOOP_HOME = "./resources/hadoop_home"
JDBC_JAR = "./resources/postgresql-42.2.8.jar"
PYSPARK_PYTHON = "python3.6"
PYSPARK_DRIVER_PYTHON = "python3.6"

os.environ["HADOOP_HOME"] = HADOOP_HOME
sys.path.append(HADOOP_HOME + "\\bin")
os.environ["PYSPARK_PYTHON"] = PYSPARK_PYTHON
os.environ["PYSPARK_DRIVER_PYTHON"] = PYSPARK_DRIVER_PYTHON

# Create the spark configuration
conf = SparkConf()
conf.set("spark.jars", JDBC_JAR)

# Build the spark session
spark = SparkSession.builder \
        .config(conf=conf) \
        .master("local") \
        .appName("Training") \
        .getOrCreate()

sc = pyspark.SparkContext.getOrCreate()

"""
Function that implements the data analysis pipeline, taking as argument the
matrix with the data and returning the validated model.
"""
def data_analysis(data):

    ##### 1- Change of data types to the corresponding ones

    data = data.withColumn("FH",data["FH"].cast("double"))
    data = data.withColumn("FC",data["FC"].cast("int"))
    data = data.withColumn("DM",data["DM"].cast("int"))
    data = data.withColumn("SensorAVG",data["SensorAVG"].cast("double"))

    ##### 2- Use a VectorAssembler to combine all feature data and separate 'label' data

    features = ["FH","FC","DM","SensorAVG"]
    va = VectorAssembler(inputCols = features, outputCol="features")
    data = va.transform(data)
    data = data.select(["features", "label"])

    ##### 3- Use a Label indexer to index labels (0 or 1)

    # Index labels, adding metadata to the label column.
    # Fit on whole dataset to include all labels in index.
    labelIndexer = StringIndexer(inputCol="label", outputCol="indexedLabel").fit(data)

    ##### 4- Split of the data into training and validation

    (trainingData, validationData) = data.randomSplit([0.7, 0.3])

    ##### 5- Train a DecisionTree model with our labels and features

    dt = DecisionTreeClassifier(labelCol="indexedLabel", featuresCol="features")

    ##### 6- Chain indexer and tree in a Pipeline

    pipeline = Pipeline(stages=[labelIndexer, dt])

    ##### 7- Train model after running the indexer

    model = pipeline.fit(trainingData)

    ##### 8- Make predictions

    predictions = model.transform(validationData)

    # Select example rows to display
    print("PREDICTIONS:")
    predictions.select("prediction", "indexedLabel", "features").show(5)

    ##### 9- Compute metrics of accuracy, recall and precision to validate the model

    print("METRICS:")
    evaluator = MulticlassClassificationEvaluator(
        labelCol="indexedLabel", predictionCol="prediction", metricName="accuracy")
    accuracy = evaluator.evaluate(predictions)
    print("Accuracy: ",accuracy)
    evaluator2 = MulticlassClassificationEvaluator(
        labelCol="indexedLabel", predictionCol="prediction", metricName="weightedRecall")
    recall = evaluator2.evaluate(predictions)
    print("Weighted recall: ",recall)
    evaluator3 = MulticlassClassificationEvaluator(
        labelCol="indexedLabel", predictionCol="prediction", metricName="weightedPrecision")
    precision = evaluator3.evaluate(predictions)
    print("Weighted precision: ",precision,"\n")

    # Print the tree model characteristics
    treeModel = model.stages[1]
    print("MODEL CHARACTERISTICS: ",treeModel,"\n")

    ##### 10- Return the final trained model

    return model


def main():
    # First we read the matrix previously created
    data = spark.read.csv("data_matrix.csv", header=True)

    # With this data, we create and train a model
    model = data_analysis(data)

    # Store model to our current working directory
    cwd = os.getcwd()
    model_name = "model"
    model.write().overwrite().save(cwd + "/" + model_name)
    print("The model was saved as '" + model_name + "' in " + cwd)

main()
