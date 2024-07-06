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

train_data, test_data = data.randomSplit([0.9, 0.1], seed=123)

lr = LogisticRegression(labelCol=target_col, featuresCol="features")
model = lr.fit(train_data)

predictions = model.transform(test_data)

evaluator = MulticlassClassificationEvaluator(
    labelCol=target_col, predictionCol="prediction", metricName="accuracy"
)


s=[]
for i in range(1,10):
    
    train_data, test_data = data.randomSplit([i/10, 1-(i/10)], seed=123)

    lr = LogisticRegression(labelCol=target_col, featuresCol="features")
    model = lr.fit(train_data)
    predictions = model.transform(test_data)

    evaluator = MulticlassClassificationEvaluator(
        labelCol=target_col, predictionCol="prediction", metricName="accuracy"
    )
    accuracy = evaluator.evaluate(predictions)
    s.append(accuracy)

print(s)
accuracy = evaluator.evaluate(predictions)
print(f"Accuracy: {accuracy*100}")

tp = predictions.filter("diabetes = 1 AND prediction = 1").count()
fp = predictions.filter("diabetes = 0 AND prediction = 1").count()
tn = predictions.filter("diabetes = 0 AND prediction = 0").count()
fn = predictions.filter("diabetes = 1 AND prediction = 0").count()

precision = tp / (tp + fp)
recall = tp / (tp + fn)
f1_score = 2 * (precision * recall) / (precision + recall)

print(f"Precision: {precision}")
print(f"Recall: {recall}")
print(f"F1 Score: {f1_score}")



spark.stop()
