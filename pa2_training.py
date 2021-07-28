import os, argparse
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.classification import RandomForestClassifier
from pyspark.sql import SparkSession
from pyspark.ml import Pipeline
from pyspark.ml.linalg import Vectors
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.feature import StandardScaler
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
from pyspark.ml.evaluation import MulticlassClassificationEvaluator

if __name__ == "__main__":
    spark = SparkSession \
        .builder \
        .appName("pa2_training") \
        .getOrCreate()
        
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', required=True, \
        help="Path to training dataset file")
    parser.add_argument('-o', default= os.getcwd() + "/model", \
        help="Model output path")
    
    args = parser.parse_args()    

    ifile = args.i
    ofile = args.o
    


    # Load training data
    training = spark \
        .read \
        .load(ifile, format="csv", sep=";", inferSchema="true", header="true")
    
    #Clean up column names
    new_column_name_list= list(map(lambda x: x.replace("\"", ""), training.columns))
    training = ( training.toDF(*new_column_name_list) ).withColumnRenamed("quality", "label")

    #Configure ML pipeline
    #lr = LogisticRegression(maxIter=10)
    rf = RandomForestClassifier()
    assembler = VectorAssembler( \
        inputCols=["fixed acidity", \
                   "volatile acidity", \
                   "citric acid", \
                   "residual sugar", \
                   "chlorides", \
                   "free sulfur dioxide", \
                   "total sulfur dioxide", \
                   "density", \
                   "pH", \
                   "sulphates", \
                   "alcohol"], \
        outputCol="num_features")
    scaler = StandardScaler(inputCol="num_features", outputCol="features", withStd=True)
    pipeline = Pipeline(stages=[assembler, scaler, rf])

    #Configure cross validator
    paramGrid = ParamGridBuilder() \
        .addGrid(rf.numTrees, [10, 100, 500]) \
        .build()
        
    crossval = CrossValidator(estimator=pipeline,
                          estimatorParamMaps=paramGrid,
                          evaluator=MulticlassClassificationEvaluator(metricName='f1'),
                          numFolds=3)
    
    # Fit the model
    model = crossval.fit(training)

    #training.show()
    
    model.write().overwrite().save( ofile )
    
    # Print the coefficients and intercept for multinomial logistic regression
    #print("Coefficients: \n" + str(model.bestModel.stages[-1].coefficientMatrix))
    #print("Intercept: " + str(model.bestModel.stages[-1].interceptVector))

    trainingSummary = model.bestModel.stages[-1].summary

    # Obtain the objective per iteration
    objectiveHistory = trainingSummary.objectiveHistory
    print("objectiveHistory:")
    for objective in objectiveHistory:
        print(objective)

    # for multiclass, we can inspect metrics on a per-label basis
    print("False positive rate by label:")
    for i, rate in enumerate(trainingSummary.falsePositiveRateByLabel):
        print("label %d: %s" % (i, rate))

    print("True positive rate by label:")
    for i, rate in enumerate(trainingSummary.truePositiveRateByLabel):
        print("label %d: %s" % (i, rate))

    print("Precision by label:")
    for i, prec in enumerate(trainingSummary.precisionByLabel):
        print("label %d: %s" % (i, prec))

    print("Recall by label:")
    for i, rec in enumerate(trainingSummary.recallByLabel):
        print("label %d: %s" % (i, rec))

    print("F-measure by label:")
    for i, f in enumerate(trainingSummary.fMeasureByLabel()):
        print("label %d: %s" % (i, f))

    accuracy = trainingSummary.accuracy
    falsePositiveRate = trainingSummary.weightedFalsePositiveRate
    truePositiveRate = trainingSummary.weightedTruePositiveRate
    fMeasure = trainingSummary.weightedFMeasure()
    precision = trainingSummary.weightedPrecision
    recall = trainingSummary.weightedRecall
    print("Accuracy: %s\nFPR: %s\nTPR: %s\nF-measure: %s\nPrecision: %s\nRecall: %s"
          % (accuracy, falsePositiveRate, truePositiveRate, fMeasure, precision, recall))
    
    spark.stop()
