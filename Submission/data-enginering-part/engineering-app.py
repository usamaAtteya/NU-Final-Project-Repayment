import numpy as np
import findspark
findspark.init()
from pyspark.sql import SparkSession
from pyspark import SparkConf
from pyspark import SparkContext
from pyspark.sql import functions as F
from pyspark.ml.feature import StringIndexer,OneHotEncoderEstimator,VectorAssembler
from pyspark.ml import Pipeline
from pyspark.sql.types import DoubleType
from pyspark.ml.tuning import ParamGridBuilder, CrossValidator
from pyspark.ml.regression import GBTRegressor
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.sql.types import *


# this preprocessing step is important only in training, because in run time we will not receive any value as privacySuppressed, but it will be null
def cleanPrivacySuppressed(dataFrame):
    # define udf to clean PrivacySuppressed for dataframe  column
    clean_privacy_suppressed =F.udf(lambda value: np.nan if value == "PrivacySuppressed" else value)
    # apply the function on every column in dataframe
    dataFrame = dataFrame.select([ clean_privacy_suppressed(c).cast("double" if c not in categ_features else "int")  for c in dataFrame.columns if c in features])
    # restore old names of columns (previously applied function changed col names)
    for index in range(len(dataFrame.columns)):
        dataFrame = dataFrame.withColumnRenamed(dataFrame.columns[index], dataFrame.columns[index].replace("CAST(<lambda>(","").replace(") AS DOUBLE)","").replace(") AS INT)",""))     
    return dataFrame

if __name__ == "__main__":
    
    sconf = SparkConf().setAppName("Repayment-Project")
    # sc = SparkContext(conf=sconf)
    spark = SparkSession.builder \
    .config('spark.master', 'local[2]') \
    .appName('Final-Project') \
    .getOrCreate()
    sc = spark.sparkContext
    web_app_model_path = "/Users/apple/Desktop/flask_projects/repayment-api/final_model"
    data_path = "/Users/apple/Desktop/BD2/bd2/data/CollegeScorecard_Raw_Data/MERGED2012_13_PP.csv"
    scorecard_data =  spark.read.format("csv").option("header", "true").option("inferSchema", "true").load(data_path)

    features =["CONTROL","ADM_RATE","ADM_RATE_ALL","SAT_AVG_ALL","SATMTMID","UGDS","HIGHDEG",  "TUITFTE", 
       "COSTT4_A", "PCTFLOAN","COMP_ORIG_YR2_RT", "UGDS_WHITE","UGDS_BLACK","UGDS_HISP","UGDS_ASIAN","UGDS_AIAN","UGDS_NHPI","UGDS_2MOR","UGDS_NRA","UGDS_UNKN","PPTUG_EF","COSTT4_P","TUITIONFEE_IN","TUITIONFEE_OUT","TUITIONFEE_PROG","INEXPFTE","PCTPELL","COMP_ORIG_YR3_RT","LOAN_COMP_ORIG_YR3_RT","DEATH_YR4_RT","COMP_ORIG_YR4_RT","AGE_ENTRY","COUNT_NWNE_P10","COUNT_WNE_P10","MN_EARN_WNE_P10","MD_EARN_WNE_P10","COMPL_RPY_1YR_RT"]
    categ_features = ["CONTROL","HIGHDEG"]
    target = "COMPL_RPY_1YR_RT"

    # this preprocessing step is important only in training, so it will not be included in the pipeline.
    scorecard_data_cleaned = cleanPrivacySuppressed(scorecard_data)
    # select not null target rows
    scorecard_data_cleaned =scorecard_data_cleaned.where(F.col(target).isNotNull()).where(F.col(target)!= np.nan)

    # hot encoder preprocessing
    hot_encoder = OneHotEncoderEstimator(inputCols=categ_features, outputCols=["{0}_ENCODED".format(colName) for colName in categ_features] )

    #vector assembler
    model_input_features = [f for f in features if f not in categ_features and f is not target ]
    model_input_features.extend(["{0}_ENCODED".format(colName) for colName in categ_features])
    vec_assembler = VectorAssembler(inputCols=model_input_features,outputCol="features",handleInvalid="keep")

    #preprocessing pipleline
    preprocessing_pipeline = Pipeline(stages=[hot_encoder,vec_assembler])
    preprocessed_data = preprocessing_pipeline.fit(scorecard_data_cleaned).transform(scorecard_data_cleaned)
    # cache this preprocessing step, this is performance optimization step for model-tunning process
    preprocessed_data.cache()


    # split data to train/test 80/20
    train_preprocessed_data = preprocessed_data.randomSplit([.8,.2])[0]
    train_preprocessed_data.cache()

    #model
    gbmodel = GBTRegressor(featuresCol="features",labelCol=target)

    # model tuning process
    evaluator =RegressionEvaluator(labelCol=target)
    paramGrid = (ParamGridBuilder()
             .addGrid(gbmodel.maxDepth, [2, 4, 6])
             .addGrid(gbmodel.maxBins, [20, 60])
             .addGrid(gbmodel.maxIter, [10, 20])
             .addGrid(gbmodel.minInfoGain, [0.0, 0.05])
             .build())
    cv = CrossValidator(estimator=gbmodel, estimatorParamMaps=paramGrid, evaluator=evaluator, numFolds=5)

    pipeline_cv = cv.fit(train_preprocessed_data)


    # final pipeline to deploy is the preprocessing steps + best model (best hyperparameters)
    final_pipeline = Pipeline(stages=[*preprocessing_pipeline.getStages(), pipeline_cv.bestModel])

    #train on all data and save model to disk
    final_model = final_pipeline.fit(scorecard_data_cleaned)
    final_model.write().overwrite().save(web_app_model_path)

    sc.stop()