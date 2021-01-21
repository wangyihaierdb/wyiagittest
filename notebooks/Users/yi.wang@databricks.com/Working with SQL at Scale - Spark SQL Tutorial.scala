// Databricks notebook source
// MAGIC %md
// MAGIC ## SQL at Scale with Spark SQL and DataFrames
// MAGIC 
// MAGIC Spark SQL brings native support for SQL to Spark and streamlines the process of querying data stored both in RDDs (Spark’s distributed datasets) and in external sources. Spark SQL conveniently blurs the lines between RDDs and relational tables. Unifying these powerful abstractions makes it easy for developers to intermix SQL commands querying external data with complex analytics, all within in a single application. Concretely, Spark SQL will allow developers to:
// MAGIC 
// MAGIC - Import relational data from Parquet files and Hive tables
// MAGIC - Run SQL queries over imported data and existing RDDs
// MAGIC - Easily write RDDs out to Hive tables or Parquet files
// MAGIC 
// MAGIC Spark SQL also includes a cost-based optimizer, columnar storage, and code generation to make queries fast. At the same time, it scales to thousands of nodes and multi-hour queries using the Spark engine, which provides full mid-query fault tolerance, without having to worry about using a different engine for historical data.
// MAGIC 
// MAGIC _For getting a deeper perspective into the background, concepts, architecture of Spark SQL and DataFrames you can check out the original article, __['SQL at Scale with Apache Spark SQL and DataFrames - Concepts, Architecture and Examples'](https://medium.com/p/c567853a702f)___
// MAGIC 
// MAGIC This tutorial will familiarize you with essential Spark capabilities to deal with structured data typically often obtained from databases or flat files. We will explore typical ways of querying and aggregating relational data by leveraging concepts of DataFrames and SQL using Spark. We will work on an interesting dataset from the [KDD Cup 1999](http://kdd.ics.uci.edu/databases/kddcup99/kddcup99.html) and try to query the data using high level abstrations like the dataframe which has already been a hit in popular data analysis tools like R and Python. We will also look at how easy it is to build data queries using the SQL language which you have learnt and retrieve insightful information from our data. This also happens at scale without us having to do a lot more since Spark distributes these data structures efficiently in the backend which makes our queries scalable and as efficient as possible.

// COMMAND ----------

// MAGIC %python
// MAGIC import pandas as pd
// MAGIC import matplotlib.pyplot as plt
// MAGIC plt.style.use('fivethirtyeight')

// COMMAND ----------

// MAGIC %md 
// MAGIC ## Data Retrieval
// MAGIC 
// MAGIC We will use data from the [KDD Cup 1999](http://kdd.ics.uci.edu/databases/kddcup99/kddcup99.html), which is the data set used for The Third International Knowledge Discovery and Data Mining Tools Competition, which was held in conjunction with KDD-99 The Fifth International Conference on Knowledge Discovery and Data Mining. The competition task was to build a network intrusion detector, a predictive model capable of distinguishing between "bad" connections, called intrusions or attacks, and "good" normal connections. This database contains a standard set of data to be audited, which includes a wide variety of intrusions simulated in a military network environment. 
// MAGIC 
// MAGIC We will be using the reduced dataset `kddcup.data_10_percent.gz` containing nearly half million nework interactions since we would be downloading this Gzip file from the web locally and then work on the same. If you have a good, stable internet connection, feel free to download and work with the full dataset available as `kddcup.data.gz`

// COMMAND ----------

// MAGIC %md
// MAGIC #### Working with data from the web
// MAGIC 
// MAGIC Dealing with datasets retrieved from the web can be a bit tricky in Databricks. Fortunately, we have some excellent utility packages like `dbutils` which help in making our job easier. Let's take a quick look at some essential functions for this module.

// COMMAND ----------

// MAGIC %python
// MAGIC dbutils.help()

// COMMAND ----------

// MAGIC %md
// MAGIC #### Retrieve and store data in Databricks
// MAGIC 
// MAGIC We will now leverage the python `urllib` library to extract the KDD Cup 99 data from their web repository, store it in a temporary location and then move it to the Databricks filesystem which can enable easy access to this data for analysis
// MAGIC 
// MAGIC __Note:__ If you skip this step and download the data directly, you may end up getting a `InvalidInputException: Input path does not exist` error

// COMMAND ----------

// MAGIC %python
// MAGIC import urllib
// MAGIC urllib.urlretrieve("http://kdd.ics.uci.edu/databases/kddcup99/kddcup.data_10_percent.gz", "/tmp/kddcup_data.gz")
// MAGIC dbutils.fs.mv("file:/tmp/kddcup_data.gz", "dbfs:/kdd/kddcup_data.gz")
// MAGIC display(dbutils.fs.ls("dbfs:/kdd"))

// COMMAND ----------

// MAGIC %md
// MAGIC ## Building the KDD Dataset
// MAGIC 
// MAGIC Now that we have our data stored in the Databricks filesystem. Let's load up our data from the disk into Spark's traditional abstracted data structure, the [Resilient Distributed Dataset (RDD)](https://spark.apache.org/docs/latest/rdd-programming-guide.html#resilient-distributed-datasets-rdds)

// COMMAND ----------

// MAGIC %python
// MAGIC data_file = "dbfs:/kdd/kddcup_data.gz"
// MAGIC raw_rdd = sc.textFile(data_file).cache()
// MAGIC raw_rdd.take(5)

// COMMAND ----------

// MAGIC %python
// MAGIC type(raw_rdd)

// COMMAND ----------

// MAGIC %md
// MAGIC ## Building a Spark DataFrame on our Data
// MAGIC 
// MAGIC A Spark DataFrame is an interesting data structure representing a distributed collecion of data. A DataFrame is a Dataset organized into named columns. It is conceptually equivalent to a table in a relational database or a dataframe in R/Python, but with richer optimizations under the hood. DataFrames can be constructed from a wide array of sources such as: structured data files, tables in Hive, external databases, or existing RDDs in our case.
// MAGIC 
// MAGIC Typically the entry point into all SQL functionality in Spark is the `SQLContext` class. To create a basic instance of this call, all we need is a `SparkContext` reference. In Databricks this global context object is available as `sc` for this purpose.

// COMMAND ----------

// MAGIC %python
// MAGIC from pyspark.sql import SQLContext
// MAGIC sqlContext = SQLContext(sc)
// MAGIC sqlContext 

// COMMAND ----------

// MAGIC %md
// MAGIC #### Splitting the CSV data
// MAGIC Each entry in our RDD is a comma-separated line of data which we first need to split before we can parse and build our dataframe

// COMMAND ----------

// MAGIC %python
// MAGIC csv_rdd = raw_rdd.map(lambda row: row.split(","))
// MAGIC print(csv_rdd.take(2))
// MAGIC print(type(csv_rdd))

// COMMAND ----------

// MAGIC %md
// MAGIC #### Check the total number of features (columns)
// MAGIC We can use the following code to check the total number of potential columns in our dataset

// COMMAND ----------

// MAGIC %python
// MAGIC len(csv_rdd.take(1)[0])

// COMMAND ----------

// MAGIC %md
// MAGIC #### Data Understanding and Parsing
// MAGIC 
// MAGIC The KDD 99 Cup data consists of different attributes captured from connection data. The full list of attributes in the data can be obtained [__here__](http://kdd.ics.uci.edu/databases/kddcup99/kddcup.names) and further details pertaining to the description for each attribute\column can be found [__here__](http://kdd.ics.uci.edu/databases/kddcup99/task.html). We will just be using some specific columns from the dataset, the details of which are specified below.
// MAGIC 
// MAGIC 
// MAGIC | feature num | feature name       | description                                                  | type       |
// MAGIC |-------------|--------------------|--------------------------------------------------------------|------------|
// MAGIC | 1           | duration           | length (number of seconds) of the connection                 | continuous |
// MAGIC | 2           | protocol_type      | type of the protocol, e.g. tcp, udp, etc.                    | discrete   |
// MAGIC | 3           | service            | network service on the destination, e.g., http, telnet, etc. | discrete   |
// MAGIC | 4           | src_bytes          | number of data bytes from source to destination              | continuous |
// MAGIC | 5           | dst_bytes          | number of data bytes from destination to source              | continuous |
// MAGIC | 6           | flag               | normal or error status of the connection                     | discrete   |
// MAGIC | 7           | wrong_fragment     | number of ``wrong'' fragments                                | continuous |
// MAGIC | 8           | urgent             | number of urgent packets                                     | continuous |
// MAGIC | 9           | hot                | number of ``hot'' indicators                                 | continuous |
// MAGIC | 10          | num_failed_logins  | number of failed login attempts                              | continuous |
// MAGIC | 11          | num_compromised    | number of ``compromised'' conditions                         | continuous |
// MAGIC | 12          | su_attempted       | 1 if ``su root'' command attempted; 0 otherwise              | discrete   |
// MAGIC | 13          | num_root           | number of ``root'' accesses                                  | continuous |
// MAGIC | 14          | num_file_creations | number of file creation operations                           | continuous |
// MAGIC 
// MAGIC We will be extracting the following columns based on their positions in each datapoint (row) and build a new RDD as follows

// COMMAND ----------

// MAGIC %python
// MAGIC from pyspark.sql import Row
// MAGIC 
// MAGIC parsed_rdd = csv_rdd.map(lambda r: Row(
// MAGIC     duration=int(r[0]), 
// MAGIC     protocol_type=r[1],
// MAGIC     service=r[2],
// MAGIC     flag=r[3],
// MAGIC     src_bytes=int(r[4]),
// MAGIC     dst_bytes=int(r[5]),
// MAGIC     wrong_fragment=int(r[7]),
// MAGIC     urgent=int(r[8]),
// MAGIC     hot=int(r[9]),
// MAGIC     num_failed_logins=int(r[10]),
// MAGIC     num_compromised=int(r[12]),
// MAGIC     su_attempted=r[14],
// MAGIC     num_root=int(r[15]),
// MAGIC     num_file_creations=int(r[16]),
// MAGIC     label=r[-1]
// MAGIC     )
// MAGIC )
// MAGIC parsed_rdd.take(5)

// COMMAND ----------

// MAGIC %md
// MAGIC #### Constructing the DataFrame
// MAGIC Now that our data is neatly parsed and formatted, let's build our DataFrame!

// COMMAND ----------

// MAGIC %python
// MAGIC df = sqlContext.createDataFrame(parsed_rdd)
// MAGIC display(df.head(10))

// COMMAND ----------

// MAGIC %md
// MAGIC Now, we can easily have a look at our dataframe's schema using tne `printSchema(...)` function.

// COMMAND ----------

// MAGIC %python
// MAGIC df.printSchema()

// COMMAND ----------

// MAGIC %md
// MAGIC ## Building a temporary table 
// MAGIC 
// MAGIC We can leverage the `registerTempTable()` function to build a temporaty table to run SQL commands on our DataFrame at scale! A point to remember is that the lifetime of this temp table is tied to the session. It creates an in-memory table that is scoped to the cluster in which it was created. The data is stored using Hive's highly-optimized, in-memory columnar format. 
// MAGIC 
// MAGIC You can also check out `saveAsTable()` which creates a permanent, physical table stored in S3 using the Parquet format. This table is accessible to all clusters. The table metadata including the location of the file(s) is stored within the Hive metastore.`

// COMMAND ----------

// MAGIC %python
// MAGIC help(df.registerTempTable)

// COMMAND ----------

// MAGIC %python
// MAGIC df.registerTempTable("connections")

// COMMAND ----------

// MAGIC %md
// MAGIC ## Executing SQL at Scale
// MAGIC Let's look at a few examples of how we can run SQL queries on our table based off our dataframe. We will start with some simple queries and then look at aggregations, filters, sorting, subqueries and pivots

// COMMAND ----------

// MAGIC %md
// MAGIC ### Connections based on the protocol type
// MAGIC 
// MAGIC Let's look at how we can get the total number of connections based on the type of connectivity protocol. First we will get this information using normal DataFrame DSL syntax to perform aggregations.

// COMMAND ----------

// MAGIC %python
// MAGIC display(df.groupBy('protocol_type')
// MAGIC           .count()
// MAGIC           .orderBy('count', ascending=False))

// COMMAND ----------

// MAGIC %md
// MAGIC Can we also use SQL to perform the same aggregation? Yes we can leverage the table we built earlier for this!

// COMMAND ----------

// MAGIC %python
// MAGIC protocols = sqlContext.sql("""
// MAGIC                            SELECT protocol_type, count(*) as freq
// MAGIC                            FROM connections
// MAGIC                            GROUP BY protocol_type
// MAGIC                            ORDER BY 2 DESC
// MAGIC                            """)
// MAGIC display(protocols)

// COMMAND ----------

// MAGIC %md
// MAGIC You can clearly see, that you get the same results and you do not need to worry about your background infrastructure or how the code is executed. Just write simple SQL!

// COMMAND ----------

// MAGIC %md
// MAGIC ### Connections based on good or bad (attack types) signatures
// MAGIC 
// MAGIC We will now run a simple aggregation to check the total number of connections based on good (normal) or bad (intrusion attacks) types.

// COMMAND ----------

// MAGIC %python
// MAGIC labels = sqlContext.sql("""
// MAGIC                            SELECT label, count(*) as freq
// MAGIC                            FROM connections
// MAGIC                            GROUP BY label
// MAGIC                            ORDER BY 2 DESC
// MAGIC                            """)
// MAGIC display(labels)

// COMMAND ----------

// MAGIC %md
// MAGIC We have a lot of different attack types. We can visualize this in the form of a bar chart. The simplest way is to use the excellent interface options in the Databricks notebook itself as depicted in the following figure!
// MAGIC 
// MAGIC ![](https://cdn-images-1.medium.com/max/800/1*MWtgLR6H4siUB1Ugc8sqog.png)
// MAGIC 
// MAGIC This gives us the following nice looking bar chart! Which you can customize further by clicking on __`Plot Options`__ as needed.

// COMMAND ----------

// MAGIC %python
// MAGIC labels = sqlContext.sql("""
// MAGIC                            SELECT label, count(*) as freq
// MAGIC                            FROM connections
// MAGIC                            GROUP BY label
// MAGIC                            ORDER BY 2 DESC
// MAGIC                            """)
// MAGIC display(labels)

// COMMAND ----------

// MAGIC %md
// MAGIC Another way is to write the code yourself to do it. You can extract the aggregated data as a `pandas` DataFrame and then plot it as a regular bar chart.

// COMMAND ----------

// MAGIC %python
// MAGIC labels_df = pd.DataFrame(labels.toPandas())
// MAGIC labels_df.set_index("label", drop=True,inplace=True)
// MAGIC labels_fig = labels_df.plot(kind='barh')
// MAGIC 
// MAGIC plt.rcParams["figure.figsize"] = (7, 5)
// MAGIC plt.rcParams.update({'font.size': 10})
// MAGIC plt.tight_layout()
// MAGIC display(labels_fig.figure)

// COMMAND ----------

// MAGIC %md
// MAGIC ### Connections based on protocols and attacks
// MAGIC 
// MAGIC Let's look at which protocols are most vulnerable to attacks now based on the following SQL query.

// COMMAND ----------

// MAGIC %python
// MAGIC attack_protocol = sqlContext.sql("""
// MAGIC                            SELECT 
// MAGIC                              protocol_type, 
// MAGIC                              CASE label
// MAGIC                                WHEN 'normal.' THEN 'no attack'
// MAGIC                                ELSE 'attack'
// MAGIC                              END AS state,
// MAGIC                              COUNT(*) as freq
// MAGIC                            FROM connections
// MAGIC                            GROUP BY protocol_type, state
// MAGIC                            ORDER BY 3 DESC
// MAGIC                            """)
// MAGIC display(attack_protocol)

// COMMAND ----------

// MAGIC %md
// MAGIC Well, looks like ICMP connections followed by TCP connections have had the maximum attacks!

// COMMAND ----------

// MAGIC %md
// MAGIC ### Connection stats based on protocols and attacks
// MAGIC 
// MAGIC Let's take a look at some statistical measures pertaining to these protocols and attacks for our connection requests.

// COMMAND ----------

// MAGIC %python
// MAGIC attack_stats = sqlContext.sql("""
// MAGIC                            SELECT 
// MAGIC                              protocol_type, 
// MAGIC                              CASE label
// MAGIC                                WHEN 'normal.' THEN 'no attack'
// MAGIC                                ELSE 'attack'
// MAGIC                              END AS state,
// MAGIC                              COUNT(*) as total_freq,
// MAGIC                              ROUND(AVG(src_bytes), 2) as mean_src_bytes,
// MAGIC                              ROUND(AVG(dst_bytes), 2) as mean_dst_bytes,
// MAGIC                              ROUND(AVG(duration), 2) as mean_duration,
// MAGIC                              SUM(num_failed_logins) as total_failed_logins,
// MAGIC                              SUM(num_compromised) as total_compromised,
// MAGIC                              SUM(num_file_creations) as total_file_creations,
// MAGIC                              SUM(su_attempted) as total_root_attempts,
// MAGIC                              SUM(num_root) as total_root_acceses
// MAGIC                            FROM connections
// MAGIC                            GROUP BY protocol_type, state
// MAGIC                            ORDER BY 3 DESC
// MAGIC                            """)
// MAGIC display(attack_stats)

// COMMAND ----------

// MAGIC %md
// MAGIC Looks like average amount of data being transmitted in TCP requests are much higher which is not surprising. Interestingly attacks have a much higher average payload of data being transmitted from the source to the destination.

// COMMAND ----------

// MAGIC %md
// MAGIC #### Filtering connection stats based on the TCP protocol by service and attack type
// MAGIC 
// MAGIC Let's take a closer look at TCP attacks given that we have more relevant data and statistics for the same. We will now aggregate different types of TCP attacks based on service, attack type and observe different metrics.

// COMMAND ----------

// MAGIC %python
// MAGIC tcp_attack_stats = sqlContext.sql("""
// MAGIC                                    SELECT 
// MAGIC                                      service,
// MAGIC                                      label as attack_type,
// MAGIC                                      COUNT(*) as total_freq,
// MAGIC                                      ROUND(AVG(duration), 2) as mean_duration,
// MAGIC                                      SUM(num_failed_logins) as total_failed_logins,
// MAGIC                                      SUM(num_file_creations) as total_file_creations,
// MAGIC                                      SUM(su_attempted) as total_root_attempts,
// MAGIC                                      SUM(num_root) as total_root_acceses
// MAGIC                                    FROM connections
// MAGIC                                    WHERE protocol_type = 'tcp'
// MAGIC                                    AND label != 'normal.'
// MAGIC                                    GROUP BY service, attack_type
// MAGIC                                    ORDER BY total_freq DESC
// MAGIC                                    """)
// MAGIC display(tcp_attack_stats)

// COMMAND ----------

// MAGIC %md
// MAGIC There are a lot of attack types and the preceding output shows a specific section of the same.

// COMMAND ----------

// MAGIC %md
// MAGIC ### Filtering connection stats based on the TCP protocol by service and attack type
// MAGIC 
// MAGIC We will now filter some of these attack types by imposing some constraints based on duration, file creations, root accesses in our query.

// COMMAND ----------

// MAGIC %python
// MAGIC tcp_attack_stats = sqlContext.sql("""
// MAGIC                                    SELECT 
// MAGIC                                      service,
// MAGIC                                      label as attack_type,
// MAGIC                                      COUNT(*) as total_freq,
// MAGIC                                      ROUND(AVG(duration), 2) as mean_duration,
// MAGIC                                      SUM(num_failed_logins) as total_failed_logins,
// MAGIC                                      SUM(num_file_creations) as total_file_creations,
// MAGIC                                      SUM(su_attempted) as total_root_attempts,
// MAGIC                                      SUM(num_root) as total_root_acceses
// MAGIC                                    FROM connections
// MAGIC                                    WHERE (protocol_type = 'tcp'
// MAGIC                                           AND label != 'normal.')
// MAGIC                                    GROUP BY service, attack_type
// MAGIC                                    HAVING (mean_duration >= 50
// MAGIC                                            AND total_file_creations >= 5
// MAGIC                                            AND total_root_acceses >= 1)
// MAGIC                                    ORDER BY total_freq DESC
// MAGIC                                    """)
// MAGIC display(tcp_attack_stats)

// COMMAND ----------

// MAGIC %md
// MAGIC Interesting to see multihop attacks being able to get root accesses to the destination hosts!

// COMMAND ----------

// MAGIC %md
// MAGIC ### Subqueries to filter TCP attack types based on service
// MAGIC 
// MAGIC Let's try to get all the TCP attacks based on service and attack type such that the overall mean duration of these attacks is greater than zero (`> 0`). For this we can do an inner query with all aggregation statistics and then extract the relevant queries and apply a mean duration filter in the outer query as shown below.

// COMMAND ----------

// MAGIC %python
// MAGIC tcp_attack_stats = sqlContext.sql("""
// MAGIC                                    SELECT 
// MAGIC                                      t.service,
// MAGIC                                      t.attack_type,
// MAGIC                                      t.total_freq
// MAGIC                                    FROM
// MAGIC                                    (SELECT 
// MAGIC                                      service,
// MAGIC                                      label as attack_type,
// MAGIC                                      COUNT(*) as total_freq,
// MAGIC                                      ROUND(AVG(duration), 2) as mean_duration,
// MAGIC                                      SUM(num_failed_logins) as total_failed_logins,
// MAGIC                                      SUM(num_file_creations) as total_file_creations,
// MAGIC                                      SUM(su_attempted) as total_root_attempts,
// MAGIC                                      SUM(num_root) as total_root_acceses
// MAGIC                                    FROM connections
// MAGIC                                    WHERE protocol_type = 'tcp'
// MAGIC                                    AND label != 'normal.'
// MAGIC                                    GROUP BY service, attack_type
// MAGIC                                    ORDER BY total_freq DESC) as t
// MAGIC                                      WHERE t.mean_duration > 0 
// MAGIC                                    """)
// MAGIC display(tcp_attack_stats)

// COMMAND ----------

// MAGIC %md
// MAGIC This is nice! Now an interesting way to also view this data is to use a pivot table where one attribute represents rows and another one represents columns. Let's see if we can leverage Spark DataFrames to do this!

// COMMAND ----------

// MAGIC %md
// MAGIC ### Building a Pivot Table from Aggregated Data
// MAGIC 
// MAGIC Here, we will build upon the previous DataFrame object we obtained where we aggregated attacks based on type and service. For this, we can leverage the power of Spark DataFrames and the DataFrame DSL.

// COMMAND ----------

// MAGIC %python
// MAGIC display((tcp_attack_stats.groupby('service')
// MAGIC                          .pivot('attack_type')
// MAGIC                          .agg({'total_freq':'max'})
// MAGIC                          .na.fill(0))
// MAGIC )

// COMMAND ----------

// MAGIC %md
// MAGIC We get a nice neat pivot table showing all the occurences based on service and attack type!
// MAGIC 
// MAGIC There are plenty of articles\tutorials available online so I would recommend you to check them out. Some useful resources for you to check out include, [the complete guide to Spark SQL from Databricks](https://docs.databricks.com/spark/latest/spark-sql/index.html).