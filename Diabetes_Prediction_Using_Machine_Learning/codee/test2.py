from pyspark.ml.tuning import ParamGridBuilder, CrossValidator
from pyspark.sql import SparkSession
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
import matplotlib.pyplot as plt
import numpy as np


spark = SparkSession.builder.appName("LogisticRegressionBollywood").getOrCreate()


data = spark.read.csv("diabetes_prediction_dataset.csv", header=True, inferSchema=True)

target_col = "diabetes"
data = data.drop("gender")
data = data.drop("smoking_history")
feature_cols = [col for col in data.columns if col != target_col]
vector_assembler = VectorAssembler(inputCols=feature_cols, outputCol="features")
data = vector_assembler.transform(data)

(train, test) = data.randomSplit([0.7, 0.3], seed = 1)

cv = CrossValidator(estimator=lr, estimatorParamMaps=paramGrid, evaluator=evaluator, numFolds=5)
# Run cross validations
cvModel = cv.fit(train)
predict_train=cvModel.transform(train)
predict_test=cvModel.transform(test)
print("Cross-validation areaUnderROC for train set is {}".format(evaluator.evaluate(predict_train)))
print("Cross-validation areaUnderROC for test set is {}".format(evaluator.evaluate(predict_test)))
