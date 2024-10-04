# ACME-Flying Predictive Analytics using Big Data Technologies

This project applies predictive analytics techniques to the ACME-Flying use case, focusing on predicting unscheduled aircraft maintenance in the next 7 days. The project uses Big Data technologies and is implemented using PySpark. The solution handles data from two main sources: a Data Warehouse (created in a previous project) and sensor data from the ACMS system, generating pipelines for data management, analysis, and real-time classification.

## Project Structure

- **Data_Management.py**: This script creates a data processing pipeline to build a feature matrix from the available data. It reads sensor data (CSV files) and aggregates it at the day level, then enriches the data with KPIs from the Data Warehouse. Finally, it labels each row for supervised learning, predicting maintenance or no maintenance.
  
- **Data_Analysis.py**: This script creates a data analysis pipeline to train a decision tree model using MLlib. It splits the data into training and validation sets, trains the classifier, and evaluates the model using accuracy and recall metrics.

- **RunTime_Classifier.py**: This script implements a runtime classifier pipeline. It uses a pre-trained model to predict if an aircraft will require unscheduled maintenance. It replicates the data processing steps for a new record and outputs the classification result.

- **main.py**: The main script to execute the complete workflow, from data processing to prediction.

## Assumptions & Notes
- The project assumes simplified input sources and features (FH, FC, DM, and sensor data).
- All sensor data is stored locally as CSVs, but in real-world cases, this data would be stored in a data lake.
- The Data Warehouse used in this project was created in a previous project, consolidating data from the AMOS and AIMS databases.
