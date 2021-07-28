import os, argparse

from pyspark.sql import SparkSession
from pyspark.sql.functions import col
from pyspark.ml.linalg import Vectors
from pyspark.ml.functions import vector_to_array
from pyspark.sql.functions import concat_ws
from pyspark.ml.tuning import CrossValidatorModel
from pyspark.ml.evaluation import MulticlassClassificationEvaluator

if __name__ == "__main__":
    spark = SparkSession \
        .builder \
        .appName("pa2_predict") \
        .getOrCreate()
        
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', required=True, \
        help="Path to testing dataset file")
    parser.add_argument('-model', default= os.getcwd() + "/model", \
        help="Path to training model directory")
    parser.add_argument('-o', default= os.getcwd() + "/output", \
        help="Output path")
    
    args = parser.parse_args()    

    ifile = args.i
    modelDir = args.model
    ofile = args.o
        
    model = CrossValidatorModel.load( modelDir )
    
    test = spark \
        .read \
        .load(ifile, format="csv", sep=";", inferSchema="true", header="true")
    
    #Clean up column names
    new_column_name_list= list(map(lambda x: x.replace("\"", ""), test.columns))
    test = ( test.toDF(*new_column_name_list) )
    
    prediction = model.transform(test)
    
    prediction = prediction.withColumn("difference", col("quality") - col("prediction") )
    
    prediction \
        .withColumn("probability", concat_ws( ",", vector_to_array("probability") ) ) \
        .select("fixed acidity", \
                "volatile acidity", \
                "citric acid", \
                "residual sugar", \
                "chlorides", \
                "free sulfur dioxide", \
                "total sulfur dioxide", \
                "density", \
                "pH", \
                "sulphates", \
                "alcohol", \
                "quality", \
                "prediction", \
                "difference", \
                "probability").write.mode("overwrite").option("header", "true").csv( ofile )
    
    selected = prediction.select( "quality", "prediction", "difference", "probability")
    for row in selected.collect():
        quality, prediction, difference, prob = row  # type: ignore
        print(
            "actual quality=%f --> predicted quality=%f, dif=%s, prob=%s" % (
                quality, prediction, difference, str(prob)   # type: ignore
            )
        )
    
    spark.stop()