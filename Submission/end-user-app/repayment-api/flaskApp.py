import numpy as np
import pandas as pd
from pyspark import SparkContext
from pyspark.sql import SQLContext
from pyspark import SparkConf
from pyspark.sql import SparkSession
from pyspark.sql.types import *
from pyspark.ml import PipelineModel
from flask import Flask, request, jsonify, render_template

sc = SparkContext('local')
sqlContext = SQLContext(sc)
app = Flask(__name__)
model = PipelineModel.load('final_model')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST', 'GET'])
def predict():
    schema = StructType([ StructField("CONTROL", IntegerType(), False)\
                       ,StructField("ADM_RATE", DoubleType(), True)\
                       ,StructField("ADM_RATE_ALL", DoubleType(), True)\
                       ,StructField("SAT_AVG_ALL", DoubleType(), True)\
                       ,StructField("SATMTMID", DoubleType(), True)\
                       ,StructField("UGDS", DoubleType(), True)\
                       ,StructField("HIGHDEG", IntegerType(), False)\
                       ,StructField("TUITFTE", DoubleType(), True)\
                       ,StructField("COSTT4_A", DoubleType(), True)\
                       ,StructField("PCTFLOAN", DoubleType(), True)\
                       ,StructField("COMP_ORIG_YR2_RT", DoubleType(), True)\
                       ,StructField("UGDS_WHITE", DoubleType(), True)\
                       ,StructField("UGDS_BLACK", DoubleType(), True)\
                       ,StructField("UGDS_HISP", DoubleType(), True)\
                       ,StructField("UGDS_ASIAN", DoubleType(), True)\
                       ,StructField("UGDS_AIAN", DoubleType(), True)\
                       ,StructField("UGDS_NHPI", DoubleType(), True)\
                       ,StructField("UGDS_2MOR", DoubleType(), True)\
                       ,StructField("UGDS_NRA", DoubleType(), True)\
                       ,StructField("UGDS_UNKN", DoubleType(), True)\
                       ,StructField("PPTUG_EF", DoubleType(), True)\
                       ,StructField("COSTT4_P", DoubleType(), True)\
                       ,StructField("TUITIONFEE_IN", DoubleType(), True)\
                       ,StructField("TUITIONFEE_OUT", DoubleType(), True)\
                       ,StructField("TUITIONFEE_PROG", DoubleType(), True)
                       ,StructField("INEXPFTE", DoubleType(), True)\
                       ,StructField("PCTPELL", DoubleType(), True)\
                       ,StructField("COMP_ORIG_YR3_RT", DoubleType(), True)\
                       ,StructField("LOAN_COMP_ORIG_YR3_RT", DoubleType(), True)\
                       ,StructField("DEATH_YR4_RT", DoubleType(), True)\
                       ,StructField("COMP_ORIG_YR4_RT", DoubleType(), True)\
                       ,StructField("AGE_ENTRY", DoubleType(), True)\
                       ,StructField("COUNT_NWNE_P10", DoubleType(), True)\
                       ,StructField("COUNT_WNE_P10", DoubleType(), True)\
                       ,StructField("MN_EARN_WNE_P10", DoubleType(), True)\
                       ,StructField("MD_EARN_WNE_P10", DoubleType(), True)])
                       
    input_features = ["CONTROL","ADM_RATE","ADM_RATE_ALL","SAT_AVG_ALL","SATMTMID","UGDS","HIGHDEG", "TUITFTE", "COSTT4_A","PCTFLOAN","COMP_ORIG_YR2_RT", "UGDS_WHITE","UGDS_BLACK","UGDS_HISP","UGDS_ASIAN","UGDS_AIAN","UGDS_NHPI","UGDS_2MOR","UGDS_NRA","UGDS_UNKN","PPTUG_EF","COSTT4_P","TUITIONFEE_IN","TUITIONFEE_OUT","TUITIONFEE_PROG","INEXPFTE","PCTPELL","COMP_ORIG_YR3_RT","LOAN_COMP_ORIG_YR3_RT","DEATH_YR4_RT","COMP_ORIG_YR4_RT","AGE_ENTRY","COUNT_NWNE_P10","COUNT_WNE_P10","MN_EARN_WNE_P10","MD_EARN_WNE_P10"]
    int_features = ["CONTROL", "HIGHDEG"]
    model_features = []    
    
    for feat in input_features:
        if feat in int_features:
            formInput = int(request.form[feat]) if (len(request.form[feat]) > 0) else np.nan
        else:
            formInput = float(request.form[feat]) if (len(request.form[feat]) > 0) else np.nan
        model_features.append(formInput)
	
    pipeline_inputs = sqlContext.createDataFrame([model_features],schema=schema)
    prediction = model.transform(pipeline_inputs).select("prediction").collect()[0][0]
    return render_template('index.html', prediction_text='prediction is:  {}'.format(prediction))

if __name__ == "__main__":
    ext_files=['./final_model/metadata/_SUCCESS','./final_model/metadata/part-00000']
    app.run(debug=True, port=5000,extra_files =ext_files)
