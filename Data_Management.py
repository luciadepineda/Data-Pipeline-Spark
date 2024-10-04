"""
Data_Management.py

The Data Management Pipeline generates a matrix where the rows denote the
information of an aircraft per day, and the columns refer to the FH, FC and DY
KPIs, and the average measurement of the 3453 sensor values. 
The followed steps are:
1- Read desired information from the 'aircraftutilization' table from Data Warehouse
2- Read desired information from the 'maintenanceevents' table from AMOS.oldinstance
3- Read CSV files with sensor measurements (which must be in a folder named 'trainingData'
   located in the current working directory)
4- Join DW and CSV info
5- Join the info obtained and AMOS info to create labels
6- Add labels (Maintenance/NonMaintenance) for each aircraft and day
7- Convert to matrix gathering all the data
The steps are explained in more detail in the corresponding part of the code.

The Data Management pipeline is implemented in a function called 'data_management',
which doesn't take any argument and returns the matrix. This function is called
in the main, creating the matrix and saving it to a csv file in the current working
directory.

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

# Configure environment variables
HADOOP_HOME = "./resources/hadoop_home"
JDBC_JAR = "./resources/postgresql-42.2.8.jar"
PYSPARK_PYTHON = "python"
PYSPARK_DRIVER_PYTHON = "python"

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
Function that implements the Data Management pipeline and returns the matrix
with the gathered data.
"""
def data_management():

	##### 1- Read desired information from the 'aircraftutilization' table from Data Warehouse

	aircraftutilization = (spark.read \
		.format("jdbc") \
		.option("driver","org.postgresql.Driver") \
		.option("url", "jdbc:postgresql://postgresfib.fib.upc.edu:6433/DW?sslmode=require") \
		.option("dbtable", "public.aircraftutilization") \
		.option("user", "lucia.de.pineda") \
		.option("password", "DB021202") \
		.load())

	df1 = aircraftutilization.select("aircraftid","timeid","flighthours", "flightcycles","delayedminutes")

	##### 2- Read desired information from the 'maintenanceevents' table from AMOS.oldinstance

	maintenanceevents = (spark.read \
		.format("jdbc") \
		.option("driver","org.postgresql.Driver") \
		.option("url","jdbc:postgresql://postgresfib.fib.upc.edu:6433/AMOS?sslmode=require") \
		.option("dbtable", "oldinstance.maintenanceevents") \
		.option("user", "lucia.de.pineda") \
		.option("password", "DB021202") \
		.load())

	# We only want the unscheduled maintenances caused by the subsystem 3453
	df2 = maintenanceevents.select("aircraftregistration","starttime") \
		.where("(subsystem = '3453') AND (kind IN ('AircraftOnGround','Delay','Safety'))")

	##### 3- Read CSV files with sensor measurements

	tuples = []

	path = "./trainingData"
	list_files = os.listdir(path)
	for file in list_files:
		# Read CSV
		data = spark.read.csv(path+'/'+file, sep=";", header=True)

		# Get Aircraft name from filename
		aircraftid = file[20:26]

		# Get Date as Date format
		day = datetime.strptime(data.select("date") \
			.collect()[0][0][0:19], '%Y-%m-%d %H:%M:%S').date()

		# Average sensors
		avg = data.select(f.mean("value")).collect()[0][0]

		# Add flight info to list of tuples with format for Spark
		key = (aircraftid, day)
		tuples.append((key,avg))

	# Average sensor values for different flights with same day and aircraft
	tuples = sc.parallelize(tuples)
	csv_info = tuples.mapValues(lambda v: (v,1)) \
		.reduceByKey(lambda s,t: (s[0]+t[0],s[1]+t[1])) \
		.mapValues(lambda v: v[0]/v[1])

	##### 4- Join DW and CSV info

	# Create a key,value structure for the DW info
	dw = df1.rdd

	# Join the elements with the same aircraft and timeid, adding the metrics
	# Format: ((aircraftid,timeid),(FH,FC,DM))
	dw_info = dw.map(lambda a: ((a[0],a[1]), (float(a[2]), \
					int(a[3]), int(a[4])))) \
				.reduceByKey(lambda s,t: (s[0]+t[0],s[1]+t[1],s[2]+t[2]))

	# Do the join, keeping only the (aircraftid, timeid) rows that contain info
	# in the csv (inner join)
	# Format: ((aircraftid,timeid),((FH,FC,DM),SensorAVG))
	join1 = dw_info.join(csv_info)

	##### 5- Join the info obtained and AMOS info to create labels

	# Create a key,value structure for the AMOS info
	# Format: (aircraftid, timeid)
	amos = df2.rdd
	amos_info = amos.map(lambda a: (a[0], a[1].date()))

	# Change format of join1 to (aircraftid, (timeid, metrics))
	# Format: (aircraftid, (timeid,FH,FC,DM,SensorAVG))
	aircrafts = join1.map(lambda a: (a[0][0], (a[0][1], a[1][0][0], a[1][0][1], a[1][0][2] \
						, a[1][1])))

	# Perform the join by aircraftid, keeping all rows from join1 ('aircrafts')
	# (left outer join) and adding the timeids for maintenances
	# Format: (aircraftid, ((timeid,FH,FC,DM,SensorAVG),timeidAMOS))
	join2 = aircrafts.leftOuterJoin(amos_info)
	
	##### 6- Add labels (Maintenance/NonMaintenance) for each aircraft and day

	# Map to check if the difference of days between an aircraft's timeid and a maintenance
	# event for that aircraft is smaller or equal to 7, which means that for that aircraft
	# and day there is a maintenance in the next 7 days (label 1)
	# Also, change the tuples to format ((aircraftid, timeid), metrics, label)
	# Format: ((aircraftid, timeid),(FH,FC,DM,SensorAVG,label))
	days7 = join2.map(lambda a: ((a[0], a[1][0][0]), (a[1][0][1], a[1][0][2], a[1][0][3] \
					, a[1][0][4], 1)) if a[1][1] >= a[1][0][0] and (a[1][1] - a[1][0][0]).days < 8 \
					 else ((a[0], a[1][0][0]), (a[1][0][1], a[1][0][2], a[1][0][3], a[1][0][4], 0)))

	# Reduce by key, taking the max label (if it has a 1 for that aircraft and
	# timeid keep it, as at least one maintenance event happens in the next 7 days) 
	# and eliminate keys as we don't need them anymore
	# Also, change the labels to Maintenance/NonMaintenance
	result = days7.reduceByKey(lambda s,t: (s[0], s[1], s[2], s[3], max(s[4], t[4]))) \
			  .map(lambda a: (a[1][0],a[1][1],a[1][2],a[1][3],"Maintenance") \
					if a[1][4]==1 else (a[1][0],a[1][1],a[1][2],a[1][3],"NonMaintenance"))

	##### 7- Convert to matrix gathering all the data

	cols = ["FH", "FC", "DM", "SensorAVG", "label"]
	matrix = result.toDF(cols)
	matrix.show(n=10)
	print("(Aircrafts,days) with maintenance in the next 7 days: ")
	print(matrix.where(matrix.label == 'Maintenance').count())
	print("(Aircrafts,days) without maintenance in the next 7 days: ")
	print(matrix.where(matrix.label == 'NonMaintenance').count())

	return matrix

def main():
	# Create matrix
	matrix = data_management()

	# Save matrix to CSV
	matrix.write.mode('overwrite').csv("data_matrix.csv", header=True)

main()
