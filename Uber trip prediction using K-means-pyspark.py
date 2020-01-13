from pyspark.sql import SparkSession,Row
spark=SparkSession.builder.appName("dates").getOrCreate()
#from pyspark.sql import Row,SQLContext
#from pyspark import SparkConf,SparkContext
from pyspark.sql.functions import format_number,dayofmonth,hour,dayofyear,month,year,weekofyear,date_format,to_timestamp
from pyspark.sql.types import LongType, StringType, IntegerType, FloatType
from pyspark.sql.types import TimestampType, StructType, StructField,DateType
from pyspark.sql.functions import udf,unix_timestamp,from_unixtime
from datetime import datetime

cust_schema=StructType([StructField("Date",StringType()),
                   StructField("Lati",FloatType()),
                   StructField("Longi",FloatType()),
                   StructField("Base_1",StringType())])



uber_april_2=(spark.read.schema(cust_schema).option("header","true").csv("uber-data-apr14.csv"))
           
uber_april_2.show()


uber_april_3=uber_april_2.withColumn('Date_Time',from_unixtime(unix_timestamp(uber_april_2['Date'],"M/d/yyyy h:mm:ss"),"yyyy-MM-dd hh:mm:ss"))

uber_april_3.show()
from pyspark.sql.functions import isnan,when,count,Column
import numpy
from numpy import array
uber_april_3.select([count(when(isnan(c),c)).alias(c) for c in uber_april_3.columns]).show()
#uber_april_3.write.csv("uber_april_intermideiate.csv")

uber_apr_4=uber_april_3.na.drop()
#uber_apr_4.show()

#uber_apr_5=uber_apr_4.select([dayofmonth(uber_apr_4['Date_Time']),hour(uber_apr_4['Date_Time']),uber_apr_4['Date_time'],uber_apr_4['Lati'],uber_apr_4['Longi'],uber_apr_4['Base_1']])
#uber_apr_5.show()

from pyspark.ml.feature import VectorAssembler
vecAssembler=VectorAssembler(inputCols=['Lati',"Longi"],outputCol="features")
uber_6=vecAssembler.transform(uber_apr_4)
uber_6.show()

from pyspark.ml.clustering import KMeans
#from pyspark.mllib.evaluation import ClusteringEvaluator

kmeans=KMeans().setK(5).setSeed(2)
model=kmeans.fit(uber_6)

predictions=model.transform(uber_6)
predictions.show()

centers=model.clusterCenters()
print("Cluster Centers: ")
for center in centers:
    print(center)




