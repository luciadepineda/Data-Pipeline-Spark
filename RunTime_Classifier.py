"""
RunTime_Classifier.py

The RunTime Classifier Pipeline predicts if, given an aircraft and a day, this
aircraft is going to have an unscheduled maintenance in the following seven days. 
It prepares these tuples to be inputted into the previously trained model and then
classifies the record as Maintenance (1) or NonMaintenance (0).
The followed steps are:
1- Replicate the Data Management Pipeline (extracting the KPIs from the DW and
obtaining the average measure of the sensor for that day and aircraft from the csv files)
2- Prepare tuple to be inputted into the model
3- Use the model to make the prediction
The steps are explained in more detail in the corresponding part of the code.

The Runtime Classifier pipeline is implemented in a function called 'runtime_classifier',
which takes as argument the aircraft, day and a trained model and returns the prediction.
This function is called in the main, where we read the three arguments given
by the user and then compute the prediction.

Authors: Lucía De Pineda & Héctor Fortuño

"""

import os
import sys
import pyspark
import argparse
from pyspark import SparkConf
from pyspark.sql import SparkSession
from pyspark.sql.types import StructType, StructField, StringType, DoubleType, IntegerType
import pyspark.sql.functions as f
from time import time
from datetime import datetime
from pyspark.ml import Pipeline, PipelineModel
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
Function that given an aircraft, a day and a model, replicates the data
management pipeline and uses the model to classify the record. It returns
the prediction of the aircraft having or not unscheduled maintenance in the following 
7 days from the given day (1 if there is maintenance predicted and 0 if not)
"""
def runtime_classifier(aircraft, date, model):

    ##### 1- Replicate the Data Management Pipeline

    # Change date type from String to Date
    timeid = datetime.strptime(date, '%d-%m-%y').date()

    ## Read DW(aircraftutilization) table

    aircraftutilization = (spark.read \
    	.format("jdbc") \
    	.option("driver","org.postgresql.Driver") \
    	.option("url", "jdbc:postgresql://postgresfib.fib.upc.edu:6433/DW?sslmode=require") \
    	.option("dbtable", "public.aircraftutilization") \
    	.option("user", "lucia.de.pineda") \
    	.option("password", "DB021202") \
    	.load())

    # Get the information for that aircraft and day
    df1 = aircraftutilization.select("flighthours","flightcycles","delayedminutes") \
    	.where("(aircraftid = '{}') AND (timeid = '{}')".format(aircraft, timeid))

    # If there is no information of that aircraft and day, prediction can't be made
    if df1.count() <= 0:
    	raise Exception('No flight info from this aircraft on this date found in the DW')

    # Join the information from several flights, adding the metrics
    # Format: (1,(FH,FC,DM))
    dw = df1.rdd
    dw_info = dw.map(lambda a: (1, (float(a[0]), int(a[1]), int(a[2])))) \
    				.reduceByKey(lambda s,t: (s[0]+t[0],s[1]+t[1],s[2]+t[2])).collect()

    ## Read csv files
    tuples = []

    path = "./trainingData"
    list_files = os.listdir(path)
    time_string = date[0:2] + date[3:5] + date[6:8]
    for file in list_files:
    	# Extract information for that aircraft and day
    	if file[20:26] == aircraft and file[0:6] == time_string:
    		data = spark.read.csv(path+'/'+file, sep=";", header=True)

    		# Average sensors
    		avg = data.select(f.mean("value")).collect()[0][0]

    		# Add flight info to list of tuples
    		tuples.append((1,(avg,1)))

    # Check that there is information in some csv for that aircraft and day
    if len(tuples) <= 0:
    	raise Exception('No sensor info from this aircraft on this date found in the csv directory')

    # Average values for different flights' sensor information
    tuples = sc.parallelize(tuples)
    csv_info = tuples.reduceByKey(lambda s,t: (s[0]+t[0],s[1]+t[1])) \
    	.mapValues(lambda v: v[0]/v[1]).collect()

    ##### 2- Prepare tuple to be inputted into the model

    # Get info together & put it in the right format
    # Format: [FH,FC,DM,SensorAVG,label]
    data = [[dw_info[0][1][0], dw_info[0][1][1], dw_info[0][1][2], csv_info[0][1], None]]

    # Convert data to Spark DataFrame with correct data types
    schema = StructType([StructField("FH", DoubleType(), False), StructField("FC", IntegerType(), False), StructField("DM", IntegerType(), False), StructField("SensorAVG", DoubleType(), False), StructField("label", DoubleType(), True)])
    data = spark.createDataFrame(data, schema = schema)

    # Use a VectorAssembler to combine all feature data and separate 'label' data
    features = ["FH","FC","DM","SensorAVG"]
    va = VectorAssembler(inputCols = features, outputCol="features")
    data = va.transform(data)
    data = data.select(["features", "label"])

    ##### 3- Use the model to make the prediction

    prediction = model.transform(data)

    if prediction.select("prediction") == 1:
    	print("Maintenance: An unscheduled maintenance event is predicted in the following 7 days.")
    else:
    	print("NonMaintenance: An unscheduled maintenance event is NOT predicted in the following 7 days.")

    # Return the entire prediction given by the model 
    # (DataFrame, label predicted is in column 'prediction')
    return prediction

def main():
    # Read arguments of aircraft, day and model filename
    parser = argparse.ArgumentParser()
    parser.add_argument('--aircraft', required=True, default=None, type=str, help='Identifier of the aircraft with format XX-XXX')
    parser.add_argument('--date', required=True, default=None, type=str, help='Date with format dd-mm-yy')
    parser.add_argument('--model', required=False, default='model', type=str, help='Name of the model file saved in your directory (e.g. myModel). Default: model')

    args = parser.parse_args()

    # Create the corresponding variables
    aircraft = args.aircraft
    date = args.date
    model_name = args.model

    # Load the stored validated model
    # Check that the model exists
    try:
        cwd = os.getcwd()
        model = PipelineModel.load(cwd + "/" + model_name)
    except:
    	raise Exception('Unable to find the model in the current working directory')

    # Check that the date is in the correct format
    try:
    	timeid = datetime.strptime(date, '%d-%m-%y').date()
    except:
    	raise Exception('Format of date is incorrect')

    # Obtain the prediction information
    prediction = runtime_classifier(aircraft, date, model)

main()
