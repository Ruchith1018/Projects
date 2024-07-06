from pyspark.sql import SparkSession
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.evaluation import MulticlassClassificationEvaluator

spark = SparkSession.builder.appName("LogisticRegressionBollywood").getOrCreate()


data = spark.read.csv("diabetes_prediction_dataset.csv", header=True, inferSchema=True)

target_col = "diabetes"
data = data.drop("gender")
data = data.drop("smoking_history")
feature_cols = [col for col in data.columns if col != target_col]

vector_assembler = VectorAssembler(inputCols=feature_cols, outputCol="features")

data = vector_assembler.transform(data)

train_data, test_data = data.randomSplit([0.7, 0.3], seed=123)

nb = NaiveBayes(labelCol="label", featuresCol="features")
model = nb.fit(train_data)

predictions = model.transform(test_data)
print(predictions)

evaluator = MulticlassClassificationEvaluator(
    labelCol="label", predictionCol="prediction", metricName="accuracy"
)
accuracy = evaluator.evaluate(predictions)

print(f"Accuracy: {accuracy*100}")

spark.stop()
