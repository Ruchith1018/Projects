from pyspark.sql import SparkSession
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.classification import RandomForestClassifier
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
rf = RandomForestClassifier(labelCol=target_col, featuresCol="features")
## train the model
rfModel = rf.fit(train)


trainSet = rfModel.summary
roc = trainSet.roc.toPandas()
plt.plot(roc['FPR'],roc['TPR'])
plt.ylabel('False Positive Rate')
plt.xlabel('True Positive Rate')
plt.title('ROC Curve')
plt.show()
print('TrainSet areaUnderROC: ' + str(trainSet.areaUnderROC))


## make predictions
predictions = rfModel.transform(test)
predictions.show(10)
spark.stop()
