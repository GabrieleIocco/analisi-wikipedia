# Databricks notebook source
# MAGIC %md
# MAGIC # **Wikipedia Analysis**

# COMMAND ----------

# MAGIC %md
# MAGIC #### _Due to the limited computational resources available in the free version of Databricks, I had to reduce the dataset by 80% and apply the CountVectorizer directly on the raw data. Otherwise, the server would crash or the cluster would be shut down due to timeout_

# COMMAND ----------

# MAGIC %md
# MAGIC ## Advanced data analysis and machine learning project.</br>The main goal is to better understand the vast wealth of information content offered by Wikipedia and to develop an automatic classification system that allows to effectively categorize new future articles

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ## Exploratory Data Analysis (EDA)
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC ### Understanding the characteristics of Wikipedia content divided into different thematic categories

# COMMAND ----------

from colorama import Fore, Back, Style

# Custom function to color text
def print_colored(text, color="white", bg_color=None, end="\n"):
    # Text Color Dictionary
    color_dict = {
        'red': Fore.RED,
        'blue': Fore.BLUE,
        'white': '\033[97m',  # Pure White (ANSI)
        'black': Fore.BLACK
    }

    # Background Color Dictionary
    bg_color_dict = {
        'black': Back.BLACK,
        'blue': Back.BLUE,
        'white': Back.WHITE
    }

    color_code = color_dict.get(color.lower(), '\033[97m') 
    bg_color_code = bg_color_dict.get(bg_color.lower(), '') if bg_color else ''
    
    print(f"{color_code}{bg_color_code}{text}{Style.RESET_ALL}", end=end)

# COMMAND ----------

spark.sql("DROP TABLE IF EXISTS default.wikipedia")

# COMMAND ----------

"""
Import dataset and sample 20% to reduce computational load

"""


!wget https://proai-datasets.s3.eu-west-3.amazonaws.com/wikipedia.csv -O /tmp/wikipedia.csv


import pandas as pd

dataset = pd.read_csv('/tmp/wikipedia.csv')
spark_df = spark.createDataFrame(dataset)
spark_df = spark_df.drop("Unnamed: 0")
spark_df.write.mode("overwrite").saveAsTable("wikipedia")

sampled_spark_df = spark_df.sample(withReplacement=False, fraction=0.2, seed=42)

# COMMAND ----------

print_colored(f"Numero di righe campionate: ", "blue", end="")
print(sampled_spark_df.count())

# COMMAND ----------

# Dataset Schema
sampled_spark_df.printSchema()

# COMMAND ----------

sampled_spark_df.isLocal()

# COMMAND ----------

display(sampled_spark_df.limit(20))

# COMMAND ----------

# MAGIC %md
# MAGIC ### Detailed Column Analysis</br>Statistics on Distinct Values, Cardinality and Missing Data

# COMMAND ----------

from pyspark.sql.functions import countDistinct, count

def analyze_column_uniqueness(sampled_spark_df):
    """
    Analyzes the uniqueness of values ​​in all columns of the DataFrame,
    providing both the count of distinct values ​​and the cardinality

    """
    # Total number of rows
    total_rows = sampled_spark_df.count()
    
    # Expressions for countDistinct for each column
    agg_expressions = [countDistinct(col).alias(f"{col}_distinct") 
                      for col in sampled_spark_df.columns]
    
    # Counting operations
    distinct_counts = sampled_spark_df.agg(*agg_expressions)
    
    # Convert to a dictionary for easier access to results
    result = distinct_counts.toPandas().to_dict('records')[0]
    
    print_colored("\nDetailed Column Analysis:", "blue")
    print("-" * 40)

    print(f"\n\nTotal number of rows: , {total_rows}")
    
    for col, count in result.items():
        # Removing the "_distinct" suffix from the column name
        column_name = col.replace('_distinct', '')
        # Cardinality
        cardinality = (count / total_rows) * 100
        
        print_colored(f"\nColumn: ", "blue", end="")
        print(column_name)
        print_colored(f"  • Distinct Values: ", "blue", end="")
        print(count)
        print_colored(f"  • Cardinality: ", "blue", end="")
        print(f"{cardinality:.2f}%")
        
analyze_column_uniqueness(sampled_spark_df)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Total Number of Rows</br>Null and Empty Fields

# COMMAND ----------

from pyspark.sql.functions import col, isnan, when, count

def analyze_missing_values(sampled_spark_df):
    """
    Detailed analysis of null and empty fields for each column of the DataFrame
    
    """
    # Total number of rows to calculate percentages
    total_rows = sampled_spark_df.count()
    
    print(f"\nMissing Values ​​Analysis on {total_rows:,} total lines:")
    print("-" * 60)
    
    for column in sampled_spark_df.columns:
        # Filtering for null and empty fields
        null_count = sampled_spark_df.filter(col(column).isNull()).count()
        empty_count = sampled_spark_df.filter(col(column) == "").count()
        
        # Percentages
        null_percentage = (null_count / total_rows) * 100
        empty_percentage = (empty_count / total_rows) * 100
        
        # Total of unsuitable values
        total_missing = null_count + empty_count
        total_percentage = (total_missing / total_rows) * 100
        
        print_colored(f"\nColumn: ", "blue", end="")
        print(column)
        print_colored(f"  • NULL Fields: ", "blue", end="")
        print(f"{null_count}, {null_percentage:.2f}%")
        print_colored(f"  • Empty Fields: ", "blue", end="")
        print(f"{empty_count:,} {empty_percentage:.2f}%")
            
        print_colored(f"  → Total of unsuitable values: ", "blue", end="")
        print(f" {total_missing} ({total_percentage:.2f}%")
    
analyze_missing_values(sampled_spark_df)

# COMMAND ----------

"""
Dictionary containing key-value pairs with the addition of the 'TOTAL_ROWS' column

"""
import pandas as pd

sampled_data = [
    {"col": "title",      "distinct_vals": 22233,   "cardinality_pct": 72.59,   "nulls": 0,    "empties": 0},
    {"col": "summary",    "distinct_vals": 22000,   "cardinality_pct": 71.83,   "nulls": 172,  "empties": 0},
    {"col": "documents",  "distinct_vals": 22031,   "cardinality_pct": 71.93,   "nulls": 172,  "empties": 0},
    {"col": "categoria",  "distinct_vals": 15,   "cardinality_pct": 0.05,   "nulls": 0,    "empties": 0},
    {"col": "TOTAL_ROWS",  "distinct_vals": 30629, "cardinality_pct": None,  "nulls": None, "empties": None},
]

sampled_data_df = pd.DataFrame(sampled_data)

# COMMAND ----------

display(sampled_data_df)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Graphical Data Analysis

# COMMAND ----------

import matplotlib.pyplot as plt
import math 
from matplotlib.ticker import MaxNLocator

plt.figure(figsize=(12, 6))
bars = plt.bar(sampled_data_df["col"], sampled_data_df["distinct_vals"], color="skyblue")
for bar in bars:
    height = bar.get_height()
    plt.text(
        bar.get_x() + bar.get_width()/2,  # center of the bar
        height,                         
        f'{int(height)}',               # text to display
        ha='center',                    
        va='bottom'                     
    )
plt.ylabel("Distinct Value")
plt.title("Distinct Value For Column (With Total Of Rows)")
plt.show()

plt.figure(figsize=(12, 6))
bars = plt.bar(sampled_data_df["col"], sampled_data_df["cardinality_pct"], color="salmon")
for bar in bars:
    height = bar.get_height()
    # If height is a valid number
    if math.isnan(height):
        # If it is NaN a default value (0) is written
        continue  
    plt.text(
        bar.get_x() + bar.get_width()/2, 
        height,                         
        f'{int(height)}',             
        ha='center',                 
        va='bottom'                  
    )
plt.ylabel("Distinct Value")
plt.ylabel("Cardinality Percentage")
plt.title("Cardinality Percentage For Column")
plt.show()

plt.figure(figsize=(12, 6))
bars = plt.bar(sampled_data_df["col"], sampled_data_df["nulls"], color="skyblue")
for bar in bars:
    height = bar.get_height()
   
    if math.isnan(height):
   
        continue  
    plt.text(
        bar.get_x() + bar.get_width()/2, 
        height,                         
        f'{int(height)}',               
        ha='center',                   
        va='bottom'                    
    )
plt.ylabel("NULL Fields")
plt.title("NULL Fields For Column")
plt.show()

plt.figure(figsize=(12, 6))
bars = plt.bar(sampled_data_df["col"], sampled_data_df["empties"], color="salmon")
for bar in bars:
    height = bar.get_height()
 
    if math.isnan(height):
       
        continue  
    plt.text(
        bar.get_x() + bar.get_width()/2, 
        height,                         
        f'{int(height)}',               
        ha='center',                   
        va='bottom'                     
    )
plt.ylabel("Empties Fields")
plt.title("Empties Fields")
plt.show()


# COMMAND ----------

# MAGIC %md
# MAGIC ### Count of articles present for each category

# COMMAND ----------

from pyspark.sql.functions import count, desc
from matplotlib.ticker import MaxNLocator

# Grouping by category and counting the number of items
category_counts = sampled_spark_df.groupBy("categoria").agg(count("*").alias("num_articles")).orderBy(desc("num_articles"))

category_counts_pd = category_counts.toPandas()


plt.figure(figsize=(12, 6))
bars = plt.bar(category_counts_pd["categoria"], category_counts_pd["num_articles"], color="skyblue")

# Annotation over the bars
for bar in bars:
    height = bar.get_height()
    plt.text(
        bar.get_x() + bar.get_width()/2, 
        height, 
        f'{int(height)}', 
        ha='center', 
        va='bottom'
    )

max_val = category_counts_pd["num_articles"].max()

# Disable default labels on x-axis
plt.xticks([])

# Annotation below bars: horizontal labels with alternating heights
for i, bar in enumerate(bars):
    x = bar.get_x() + bar.get_width()/2
   
    offset = 0.05 * max_val if i % 2 == 0 else 0.1 * max_val
    plt.text(
        x,
        -offset, 
        category_counts_pd["categoria"].iloc[i],
        ha='center',
        va='top',
        rotation=0  # Horizontal text
    )

# Expanding the lower limit to accommodate labels
plt.ylim(-0.15 * max_val, None)
plt.xlabel("Thematic categories")
plt.ylabel("Number of articles")
plt.title("Numbers of Articles for Categories")
plt.gca().yaxis.set_major_locator(MaxNLocator(integer=True))
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC #### _The number of articles present among the various categories appears to be quite balanced_

# COMMAND ----------

# MAGIC %md
# MAGIC ### Average word count per article</br>The length of the longest and shortest article for each category

# COMMAND ----------

from pyspark.sql.functions import split, size, avg, min, max, col, desc

sampled_spark_filled_df = sampled_spark_df.na.fill({"documents": ""})

# Number of words for each article in the "summary" column
documents_length_df = sampled_spark_filled_df.withColumn("word_count", size(split(col("documents"), "\\s+")))

# Grouping by category and calculating statistics
stats_per_category = documents_length_df.groupBy("categoria").agg(
    count("*").alias("num_articles"),
    avg("word_count").alias("avg_word_count_article"),
    min("word_count").alias("min_word_count_article"),
    max("word_count").alias("max_word_count_article")
).orderBy(desc("num_articles"))

display(stats_per_category)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Graphical Data Analysis

# COMMAND ----------

from pyspark.sql.functions import desc

stats_per_category_ordered = stats_per_category.orderBy(desc("max_word_count_article"))
stats_per_category_pd = stats_per_category_ordered.toPandas()


# Maximum value and the related category
max_val = stats_per_category_pd["max_word_count_article"].max()
max_category = stats_per_category_pd[stats_per_category_pd["max_word_count_article"] == max_val]["categoria"].iloc[0]

plt.figure(figsize=(12, 6))
bars = plt.bar(stats_per_category_pd["categoria"], stats_per_category_pd["max_word_count_article"], color="skyblue")

# Annotation over the bars
for bar in bars:
    height = bar.get_height()
    if height == max_val:
        plt.text(
            bar.get_x() + bar.get_width()/2, 
            height, 
            f'{int(height)}', 
            ha='center', 
            va='bottom'
        )
    else:
        plt.text(
            bar.get_x() + bar.get_width()/2, 
            height, 
            f'{int(height)}', 
            ha='center', 
            va='bottom'
        )

plt.ylabel("Number of Words")
plt.title("Max Words Count for Category")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# COMMAND ----------

for index, row in stats_per_category_pd.iterrows():
    print(f"Categoria: {row['categoria']}, Max Word Count: {row['max_word_count_article']}")

# COMMAND ----------

# MAGIC %md
# MAGIC #### _The count of the maximum number of words between the various categories appears to be very unbalanced_

# COMMAND ----------


stats_per_category_ordered = stats_per_category.orderBy(desc("min_word_count_article"))
stats_per_category_pd = stats_per_category_ordered.toPandas()

# Trova il valore massimo e la relativa categoria
max_val = stats_per_category_pd["min_word_count_article"].max()
max_category = stats_per_category_pd[stats_per_category_pd["min_word_count_article"] == max_val]["categoria"].iloc[0]

# Crea il bar plot
plt.figure(figsize=(12, 6))
bars = plt.bar(stats_per_category_pd["categoria"], stats_per_category_pd["min_word_count_article"], color="skyblue")

# Annotazione sopra le barre: evidenzia in rosso la barra con il valore massimo
for bar in bars:
    height = bar.get_height()
    if height == max_val:
        plt.text(
            bar.get_x() + bar.get_width()/2, 
            height, 
            f'{int(height)}', 
            ha='center', 
            va='bottom'
        )
    else:
        plt.text(
            bar.get_x() + bar.get_width()/2, 
            height, 
            f'{int(height)}', 
            ha='center', 
            va='bottom'
        )

plt.ylabel("Number of Words")
plt.title("Min Words Count for Category")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# COMMAND ----------

# Stampa i valori per ogni categoria
for index, row in stats_per_category_pd.iterrows():
    print(f"Categoria: {row['categoria']}, Min Word Count: {row['min_word_count_article']}")


# COMMAND ----------

# MAGIC %md
# MAGIC %md
# MAGIC #### _The counting of the minimum number of words between the various categories appears to be quite unbalanced_

# COMMAND ----------

# MAGIC %md
# MAGIC ### Since all categories contain at least one article composed of few words and therefore little information, I filter the dataset by eliminating articles that have less than 25 words

# COMMAND ----------

from pyspark.sql.functions import size, split

# Removal of articles with less than 25 words
filtered_spark_df = sampled_spark_filled_df.filter(size(split(col("documents"), " ")) >= 25)

print("Numero di articoli dopo il filtro:", filtered_spark_df.count())

# COMMAND ----------

# MAGIC %md
# MAGIC ### Calculating statistics after applying the filter

# COMMAND ----------

from pyspark.sql.functions import count, desc
from matplotlib.ticker import MaxNLocator

category_counts = filtered_spark_df.groupBy("categoria").agg(count("*").alias("num_articles")).orderBy(desc("num_articles"))

category_counts_pd = category_counts.toPandas()


plt.figure(figsize=(12, 6))
bars = plt.bar(category_counts_pd["categoria"], category_counts_pd["num_articles"], color="orange")

for bar in bars:
    height = bar.get_height()
    plt.text(
        bar.get_x() + bar.get_width()/2, 
        height, 
        f'{int(height)}', 
        ha='center', 
        va='bottom'
    )

max_val = category_counts_pd["num_articles"].max()

plt.xticks([])

for i, bar in enumerate(bars):
    x = bar.get_x() + bar.get_width()/2

    offset = 0.05 * max_val if i % 2 == 0 else 0.1 * max_val
    plt.text(
        x,
        -offset, 
        category_counts_pd["categoria"].iloc[i],
        ha='center',
        va='top',
        rotation=0  
    )

plt.ylim(-0.15 * max_val, None)
plt.xlabel("Thematic categories")
plt.ylabel("Number of articles")
plt.title("Numbers of Articles for Categories")
plt.gca().yaxis.set_major_locator(MaxNLocator(integer=True))
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC #### _The number of articles present among the various categories appears to be quite balanced_

# COMMAND ----------

from pyspark.sql.functions import split, size, avg, min, max, col, desc

filtered_sampled_spark_filled_df = filtered_spark_df.na.fill({"documents": ""})

filtered_documents_length_df = filtered_sampled_spark_filled_df.withColumn("word_count", size(split(col("documents"), "\\s+")))

filtered_stats_per_category = filtered_documents_length_df.groupBy("categoria").agg(
    count("*").alias("num_articles"),
    avg("word_count").alias("avg_word_count_article"),
    min("word_count").alias("min_word_count_article"),
    max("word_count").alias("max_word_count_article")
).orderBy(desc("num_articles"))

display(filtered_stats_per_category)

# COMMAND ----------

from pyspark.sql.functions import desc

# Sorting the dataset in descending order
filtered_stats_per_category_ordered = filtered_stats_per_category.orderBy(desc("max_word_count_article"))

# Convert to Pandas
filtered_stats_per_category_pd = filtered_stats_per_category_ordered.toPandas()

# Maximum value and the related category
max_val = filtered_stats_per_category_pd["max_word_count_article"].max()
max_category = filtered_stats_per_category_pd[filtered_stats_per_category_pd["max_word_count_article"] == max_val]["categoria"].iloc[0]

plt.figure(figsize=(12, 6))
bars = plt.bar(filtered_stats_per_category_pd["categoria"], filtered_stats_per_category_pd["max_word_count_article"], color="orange")

for bar in bars:
    height = bar.get_height()
    if height == max_val:
        plt.text(bar.get_x() + bar.get_width()/2, height, f'{int(height)}', ha='center', va='bottom')
    else:
        plt.text(bar.get_x() + bar.get_width()/2, height, f'{int(height)}', ha='center', va='bottom')

plt.ylabel("Number of Words")
plt.title("Max Words Count for Category")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()


# COMMAND ----------

# MAGIC %md
# MAGIC #### _The maximum number of words remained unchanged_

# COMMAND ----------


filtered_stats_per_category_ordered = filtered_stats_per_category.orderBy(desc("min_word_count_article"))
filtered_stats_per_category_pd = filtered_stats_per_category.toPandas()

# Trova il valore massimo e la relativa categoria
max_val = filtered_stats_per_category_pd["min_word_count_article"].max()
max_category = filtered_stats_per_category_pd[filtered_stats_per_category_pd["min_word_count_article"] == max_val]["categoria"].iloc[0]

# Crea il bar plot
plt.figure(figsize=(12, 6))
bars = plt.bar(filtered_stats_per_category_pd["categoria"], filtered_stats_per_category_pd["min_word_count_article"], color="orange")

# Annotazione sopra le barre: evidenzia in rosso la barra con il valore massimo
for bar in bars:
    height = bar.get_height()
    if height == max_val:
        plt.text(
            bar.get_x() + bar.get_width()/2, 
            height, 
            f'{int(height)}', 
            ha='center', 
            va='bottom'
        )
    else:
        plt.text(
            bar.get_x() + bar.get_width()/2, 
            height, 
            f'{int(height)}', 
            ha='center', 
            va='bottom'
        )

plt.ylabel("Number of Words")
plt.title("Min Words Count for Category")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC #### _The minimum number of words per category is less unbalanced after applying the filter on the minimum number of words.</br> The category 'politics' still has longer articles than the others and this could influence the automatic classifier_

# COMMAND ----------

# MAGIC %md
# MAGIC ### Calculate the number of articles in the 'summary' field that are also present in the 'documents' field

# COMMAND ----------

from pyspark.sql.functions import col

identical_num = filtered_spark_df.filter(col("summary") == col("documents")).count()
total_num = filtered_spark_df.count()

print(f"Number of articles in which in cui summary == documents: {identical_num} su {total_num}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Creation of representative Word Clouds for each category, to identify the most frequent and relevant terms

# COMMAND ----------

from pyspark.sql.functions import concat_ws, collect_list

# Filter by category
filtered_cat_documents_df = filtered_spark_df.groupBy("categoria").agg(
    concat_ws(" ", collect_list("documents")).alias("all_documents")
)

# COMMAND ----------

filtered_cat_documents_pd = filtered_cat_documents_df.toPandas()

# COMMAND ----------

import matplotlib.pyplot as plt
from wordcloud import WordCloud

for idx, row in filtered_cat_documents_pd.iterrows():
    cat = row["categoria"]
    text = row["all_documents"]
    
    if text.strip() == "":
        continue
    
    wc = WordCloud(width=1200, height=600, background_color="white").generate(text)
    
    plt.figure(figsize=(10, 5))
    plt.imshow(wc, interpolation="bilinear")
    plt.axis("off")
    plt.title( f"\n\nWord Cloud per Categoria: {cat}\n", fontsize=18)
    plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Data Preprocessing 

# COMMAND ----------

# MAGIC %md
# MAGIC #### _At this point in the project I identified the language of the articles for the summary and documents fields. The analysis showed that the predominant language in the text corpus is English, although other languages ​​are also present with a lower frequency.</br></br>After further exploration, I observed that the initial parts of some documents contain fragments in languages ​​other than English, typically to indicate birth and death dates, names of structures or other introductory information. However, the rest of the content remains in English.</br></br>To handle this feature, I implemented the Sliding Window technique, dividing each document into two sections:</br>Header, containing the initial part, where different languages ​​could be present.</br>Body, consisting of the main text in English.</br></br>The goal was to separate the languages ​​and train the classifier using only the English body, avoiding noise from multilingual elements. However, due to the limited computational resources available, I had to abandon this strategy and provide the classifier with the articles in their entirety.</br></br>I'll leave the code anyway_

# COMMAND ----------

# MAGIC %md
# MAGIC # Start - Optimizations Not Applied Due to Computational Limits

# COMMAND ----------

# MAGIC %md
# MAGIC ## Text Minining

# COMMAND ----------

# MAGIC %md
# MAGIC #### When applying NLP techniques with Text Mining, you can normalize texts (tokenization, stopword removal, lemmatization/stemming), use language detection techniques to identify the language of each text, filter and separate texts by language, and build models and pipelines for specific languages.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Automatic Document Language Detection with SparkNLP

# COMMAND ----------

# MAGIC %md
# MAGIC ### Language Analysis on column 'Summary'

# COMMAND ----------

import sparknlp
from sparknlp.base import DocumentAssembler
from sparknlp.annotator import LanguageDetectorDL
from pyspark.ml import Pipeline

"""
langdetect on the column 'summary' on the entire dataset
    
"""
    
# Document Assembler turns each line of text into a “document” type annotation
document_assembler = DocumentAssembler() \
    .setInputCol("summary") \
    .setOutputCol("summary_document")

# LanguageDetectorDL pre-trained
language_detector = LanguageDetectorDL.pretrained("ld_wiki_tatoeba_cnn_43", "xx") \
    .setInputCols(["summary_document"]) \
    .setOutputCol("language_summary")

# Pipeline
pipeline = Pipeline(stages=[
    document_assembler,
    language_detector
])

# COMMAND ----------

# Running the pipeline on the column 'summary' of the dataframe
model = pipeline.fit(sampled_spark_df)
sampled_spark_df_lang_summary = model.transform(sampled_spark_df)

sampled_spark_df_lang_summary.select("summary", "language_summary.result").show(truncate=False, n=10)


# COMMAND ----------

"""
Grouping of texts based on the language found
    
"""

from pyspark.sql.functions import explode, col

sampled_spark_df_lang_summary.select(explode(col("language_summary.result")).alias("lang")) \
    .groupBy("lang") \
    .count() \
    .orderBy(col("count").desc()) \
    .show()


# COMMAND ----------

# MAGIC %md
# MAGIC #### *The dataset appears to be multilingual even if it is mainly composed of documents written in English*

# COMMAND ----------

# MAGIC %md
# MAGIC #### Minority Language Analysis on column 'summary'

# COMMAND ----------

from pyspark.sql.functions import explode, col, row_number, rand
from pyspark.sql.window import Window

# Exploding column 'lingual.result' to have (lang, summary) on separate lines
sampled_exploded_summary_df = (sampled_spark_df_lang_summary
    .select(
        col("summary"),
        explode(col("language_summary.result")).alias("lang")
    )
)

distinct_langs_summary_count = sampled_exploded_summary_df.select("lang").distinct().count()
print_colored("Numero totale di lingue:", "blue", end="")
print(distinct_langs_summary_count)

# Let's filter out the English language
no_en_summary_df = sampled_exploded_summary_df.filter(col("lang") != "en")

# Partitioning via 'Window' with 'lang'
windowSpec = Window.partitionBy("lang").orderBy(rand())

# Creating column 'row_num' 
sampl_summary_df = no_en_summary_df.withColumn("row_num", row_number().over(windowSpec)) \
                     .filter(col("row_num") <= 10)

sampl_summary_df.select("lang", "summary", "row_num").orderBy("lang", "row_num").show(30, truncate=False)

# COMMAND ----------

# MAGIC %md
# MAGIC #### *The analysis of minority languages ​​on the 'summary' column revealed that only in the first part of a document is there a language other than English, for example to indicate the date of birth and death, or the name of a structure, the rest of the document is in English.*

# COMMAND ----------

# MAGIC %md
# MAGIC ## Language Analysis on 'documents' column

# COMMAND ----------

import sparknlp
from sparknlp.base import DocumentAssembler
from sparknlp.annotator import LanguageDetectorDL
from pyspark.ml import Pipeline

"""
langdetect on the column 'documents' on the entire dataset
    
"""
    
document_assembler_documents = DocumentAssembler() \
    .setInputCol("documents") \
    .setOutputCol("documents_document")

language_detector_documents = LanguageDetectorDL.pretrained("ld_wiki_tatoeba_cnn_43", "xx") \
    .setInputCols(["documents_document"]) \
    .setOutputCol("language_documents")

pipeline = Pipeline(stages=[
    document_assembler_documents,
    language_detector_documents
])

# COMMAND ----------

# Running the pipeline on the 'documents' column of the dataframe
model = pipeline.fit(sampled_spark_df)
sampled_spark_df_lang_documents = model.transform(sampled_spark_df)

sampled_spark_df_lang_documents.select("documents", "language_documents.result").show(truncate=False, n=10)

# COMMAND ----------

"""
Grouping of texts based on the language found
    
"""

from pyspark.sql.functions import explode, col

sampled_spark_df_lang_documents.select(explode(col("language_documents.result")).alias("lang")) \
    .groupBy("lang") \
    .count() \
    .orderBy(col("count").desc()) \
    .show()


# COMMAND ----------

# MAGIC %md
# MAGIC #### *The 'documents' column also has a multilingual dataset even though it is mostly composed of documents written in English*

# COMMAND ----------

# MAGIC %md
# MAGIC %md
# MAGIC ## Minority Language Analysis on column 'documents'

# COMMAND ----------

from pyspark.sql.functions import explode, col, row_number, rand
from pyspark.sql.window import Window

# 1) Esplodiamo la colonna 'language.result' per avere (lang, summary) su righe separate
sampled_exploded_documents_df = (sampled_spark_df_lang_documents
    .select(
        col("documents"),
        explode(col("language_documents.result")).alias("lang")
    )
)

distinct_langs_documents_count = sampled_exploded_documents_df.select("lang").distinct().count()
print_colored("Numero totale di lingue: ", "blue", end="")
print(distinct_langs_documents_count)

# 2) Filtriamo via l’inglese
no_en_documents_df = sampled_exploded_documents_df.filter(col("lang") != "en")

# 3) Creiamo una finestra: partizioniamo per 'lang'
#    e ordiniamo a caso (rand()) o con un'altra colonna se vuoi stabilire un criterio
windowSpec = Window.partitionBy("lang").orderBy(rand())

# 4) Assegniamo row_number e prendiamo solo le prime 10 righe per lingua
sampled_documents_df = no_en_documents_df.withColumn("row_num", row_number().over(windowSpec)) \
                     .filter(col("row_num") <= 10)

# 5) Visualizziamo
sampled_documents_df.select("lang", "documents", "row_num").orderBy("lang", "row_num").show(30, truncate=False)


# COMMAND ----------

# MAGIC %md
# MAGIC #### *The analysis of minority languages ​​on the 'documents' column also revealed that only in the first part of a document is there a language other than English, for example to indicate the date of birth and death, or the name of a structure, the rest of the document is in English.*

# COMMAND ----------

# MAGIC %md
# MAGIC ## Sliding Window

# COMMAND ----------

# MAGIC %md
# MAGIC #### Application of the sliding window with a division of the text into segments of fixed length and application of language detection to each segment. The idea is to detect the point where the language switches from the head language (non-English) to the body language (English) so as to focus the work only on the English text corpus.
# MAGIC

# COMMAND ----------

from langdetect import detect, DetectorFactory
DetectorFactory.seed = 42 

def split_by_language_change(text, window_size=25, step=5):
    """
    It splits the text into two parts: header and body.
    The text is scanned with a sliding window.
    It is assumed that the 'header' is in the first positions and in a language other than 'en',
    and that the 'body' is mostly in English.
    It also handles null values.

    window_size represents the minimum amount of
    text needed to perform reliable language detection.

    """
    # If text is None returns tuples of empty strings
    if text is None:
        return ("", "")
    if len(text) < window_size: 
        return ("", text)
    
    header_lang = None
    split_index = None
    
    # Sliding Window
    for i in range(0, len(text) - window_size + 1, step):
        segment = text[i: i + window_size]
        try:
            lang = detect(segment)
        except Exception:
            lang = "unknown"
        # Header language from first segment
        if header_lang is None:
            header_lang = lang
        else:
            # If the current language is 'en' and the header was not, this dot is used as a separator.
            if lang == "en" and header_lang != "en":
                split_index = i
                break
    if split_index is None:
        return ("", text)
    else:
        return (text[:split_index].strip(), text[split_index:].strip())

# COMMAND ----------

# MAGIC %md
# MAGIC #### Sample Text

# COMMAND ----------

# Example
sample_text = "gennady korotkevich (belarusian: генадзь караткевіч, hienadź karatkievič, russian: геннадий короткевич; born 25 september 1994) is a belarusian competitive programmer who has won major international competitions since the age of 11, as well as numerous national competitions. his top accomplishments include six consecutive gold medals in the international olympiad in informatics as well as the world championship in the 2013 and 2015 international collegiate programming contest world finals. as of december 2022, gennady is the highest-rated programmer on codechef, topcoder, atcoder and hackerrank. in january 2022, he achieved a historic rating of 3979 on codeforces, becoming the first to break the 3900 barrier.       "
header, body = split_by_language_change(sample_text)
print("Header:", header)
print("Body:", body)


# COMMAND ----------

# MAGIC %md
# MAGIC #### *After several tests with some example comments taken from the dataset I set window_size=25 and step=5*

# COMMAND ----------

# MAGIC %md
# MAGIC ### Applying the Sliding Window on the 'summary' column of the entire Dataset

# COMMAND ----------

from pyspark.sql.functions import udf
from pyspark.sql.types import StructType, StructField, StringType

schema = StructType([
    StructField("header", StringType(), True),
    StructField("body", StringType(), True)
])

split_udf_summary = udf(split_by_language_change, schema)

# Applying UDF on the "summary" column
split_df_summary = sampled_spark_df.withColumn("split_summary_text", split_udf_summary("summary"))

split_df_summary.select("split_summary_text.header", "split_summary_text.body").show(truncate=False)

# COMMAND ----------

# MAGIC %md
# MAGIC %md
# MAGIC ### Applying Sliding Window on the 'documents' column of the entire Dataset

# COMMAND ----------

from pyspark.sql.functions import udf
from pyspark.sql.types import StructType, StructField, StringType

schema = StructType([
    StructField("header", StringType(), True),
    StructField("body", StringType(), True)
])

split_udf_documents = udf(split_by_language_change, schema)

split_df_documents = sampled_spark_df.withColumn("split_documents_text", split_udf_documents("documents"))

split_df_documents.select("split_documents_text.header", "split_documents_text.body").show(truncate=False)

# COMMAND ----------

from pyspark.sql.functions import monotonically_increasing_id, concat_ws, col

"""
Merge and create combined columne

"""

# Adding an ID
body_summary_df = split_df_summary.select("body_summary").withColumn("id", monotonically_increasing_id())
body_documents_df = split_df_documents.select("body_documents").withColumn("id", monotonically_increasing_id())

# Join on DataFrames based on ID
combined_df = body_summary_df.join(body_documents_df, on="id", how="inner")

# Column 'combined_text' concatenating body_summary e body_documents
combined_df = combined_df.withColumn("combined_text", concat_ws(" ", col("body_summary"), col("body_documents")))

combined_df.select("id", "body_summary", "body_documents", "combined_text").show(5, truncate=False)

# COMMAND ----------

# MAGIC %md
# MAGIC # End - Optimizations Not Applied Due to Computational Limits

# COMMAND ----------

# MAGIC %md
# MAGIC ## Join summary and documents columns

# COMMAND ----------

from pyspark.sql.functions import concat_ws, col

filtered_combined_df = filtered_spark_df.withColumn("combined_text", concat_ws(" ", col("summary"), col("documents")))
filtered_combined_df.select("summary", "documents", "combined_text").show(5, truncate=False)

# COMMAND ----------

display(filtered_combined_df)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Tokenizing text with 'split()'

# COMMAND ----------

from pyspark.sql.functions import split

token_combined_df = filtered_combined_df.withColumn("tokens", split(col("combined_text"), " "))

# COMMAND ----------

display(token_combined_df)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Vectorization with CountVectorizer and IDF

# COMMAND ----------

# MAGIC %md
# MAGIC #### Elimination of too rare and excessively frequent words in text corpora.</br>Only words present in at least 1% of the articles and with a maximum frequency of 80% were retained, thus improving the robustness of the classifier.</br>This approach also helps prevent overfitting on specific words present in a few documents.

# COMMAND ----------

from pyspark.ml.feature import CountVectorizer, IDF

# CountVectorizer on column 'tokens'
cv = CountVectorizer(inputCol="tokens", outputCol="raw_features", vocabSize=2000, maxDF=0.80, minDF=0.01)
cv_model = cv.fit(token_combined_df)
df_cv = cv_model.transform(token_combined_df)

# Calculating IDF to get TF-IDF vector
idf = IDF(inputCol="raw_features", outputCol="features")
idf_model = idf.fit(df_cv)
df_with_features = idf_model.transform(df_cv)

df_with_features.select("combined_text", "tokens", "features").show(5, truncate=False)

# COMMAND ----------

display(df_with_features)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Preparing labels and splitting the dataset

# COMMAND ----------

from pyspark.ml.feature import StringIndexer, IndexToString

# Creating the StringIndexer and training the label model
label_indexer = StringIndexer(inputCol="categoria", outputCol="label")
label_indexer_model = label_indexer.fit(df_with_features)

# COMMAND ----------



# Trasforma il DataFrame per ottenere la colonna "label"
df_final = label_indexer_model.transform(df_with_features)


# Transforming the DataFrame to get the "label" column
train_df, val_df, test_df = df_final.randomSplit([0.7, 0.15, 0.15], seed=42)
print("Train:", train_df.count(), "Validation:", val_df.count(), "Test:", test_df.count())


# COMMAND ----------

# MAGIC %md
# MAGIC ## Implementation and development of automatic classifiers

# COMMAND ----------

# MAGIC %md
# MAGIC ### Logistic Regression Algorithm

# COMMAND ----------

from pyspark.ml.classification import LogisticRegression
from pyspark.ml import Pipeline

lr = LogisticRegression(featuresCol="features", labelCol="label", maxIter=10)

pipeline_ml = Pipeline(stages=[lr])
model_ml = pipeline_ml.fit(train_df)

predictions = model_ml.transform(test_df)
predictions.select("combined_text", "categoria", "prediction", "probability").show(10, truncate=False)


# COMMAND ----------

from pyspark.ml.evaluation import MulticlassClassificationEvaluator

# Assicurati che il DataFrame predictions abbia le colonne "label" e "prediction".
# Calcola l'accuracy
accuracy_evaluator = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="accuracy")
accuracy = accuracy_evaluator.evaluate(predictions)
print("Accuracy:", accuracy)

# Calcola il f1-score
f1_evaluator = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="f1")
f1 = f1_evaluator.evaluate(predictions)
print("F1-Score:", f1)

# Per precision e recall, puoi utilizzare le metriche "weightedPrecision" e "weightedRecall":
precision_evaluator = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="weightedPrecision")
precision = precision_evaluator.evaluate(predictions)
print("Weighted Precision:", precision)

recall_evaluator = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="weightedRecall")
recall = recall_evaluator.evaluate(predictions)
print("Weighted Recall:", recall)


# COMMAND ----------

# MAGIC %md
# MAGIC ### Graphical Data Analysis

# COMMAND ----------

from pyspark.ml.evaluation import MulticlassClassificationEvaluator
import matplotlib.pyplot as plt

# Calculating metrics
accuracy_evaluator = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="accuracy")
accuracy = accuracy_evaluator.evaluate(predictions)

f1_evaluator = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="f1")
f1 = f1_evaluator.evaluate(predictions)

precision_evaluator = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="weightedPrecision")
precision = precision_evaluator.evaluate(predictions)

recall_evaluator = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="weightedRecall")
recall = recall_evaluator.evaluate(predictions)

metrics = {"Accuracy": accuracy, "F1-Score": f1, "Precision": precision, "Recall": recall}

plt.figure(figsize=(8, 6))
plt.bar(metrics.keys(), metrics.values(), color="skyblue")
plt.ylim(0, 1)
plt.ylabel("Valore")
plt.title("Metriche del Modello")
for key, value in metrics.items():
    plt.text(key, value + 0.02, f"{value:.2f}", ha="center", va="bottom")
plt.show()

# COMMAND ----------

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

# Converting Label and Prediction Columns in Pandas
predictions_pd = predictions.select("label", "prediction").toPandas()

# Confusion Matrix
cm = confusion_matrix(predictions_pd["label"], predictions_pd["prediction"])

disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot(cmap=plt.cm.Blues)
plt.title("Matrice di Confusione")
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC ### Model predictions (Logistic Regression)

# COMMAND ----------

# MAGIC %md
# MAGIC #### 1° Example

# COMMAND ----------

from pyspark.sql.functions import split, col
from pyspark.ml.feature import IndexToString

# Test input
test_text = (
    "The Evolution of Sports: From Ancient Games to Modern CompetitionsSports have played a significant role in human history, evolving from ancient competitions to modern professional leagues and international tournaments. The love for sports transcends cultures, bringing people together through physical excellence, teamwork, and competition.Ancient Origins of SportsThe history of sports dates back thousands of years, with evidence of organized physical contests found in Ancient Egypt, Greece, and China. The Olympic Games, first held in 776 BCE in Olympia, Greece, were one of the earliest recorded sporting events. These games featured disciplines such as:Running (stadion race) – A short sprint, considered the most prestigious event.Wrestling and boxing – Physical combat sports that tested endurance and technique.Chariot racing – A thrilling and dangerous sport popular among the Greeks and later the Romans.Other civilizations had their own sports traditions. The Mesoamerican ballgame, played by the Maya and Aztecs, involved a rubber ball and a hoop-like goal, while the Chinese cuju is considered an early form of soccer.The Rise of Modern SportsThe modern era of sports began in the 19th and early 20th centuries, with the formalization of rules and the establishment of international competitions. Many of today’s most popular sports were standardized during this period, including:Soccer (Football) – The Football Association (FA) was founded in 1863 in England, creating a uniform rule set for the game.Basketball – Invented in 1891 by Dr. James Naismith, it became one of the most popular sports worldwide.Tennis and Golf – Developed in the late 19th century, with international tournaments like Wimbledon (1877) and The Open Championship (1860) emerging.The re-establishment of the Olympic Games in 1896, spearheaded by Pierre de Coubertin, marked a turning point for global sports, introducing a new platform for international competition.The Professionalization of SportsBy the 20th century, sports had become more than just recreational activities—they transformed into multibillion-dollar industries. The rise of professional leagues, sponsorships, and television broadcasts allowed athletes to compete at the highest level while earning significant financial rewards.Some key moments in professional sports history include:The Formation of Major Leagues – The National Football League (NFL) (1920), National Basketball Association (NBA) (1946), and Major League Baseball (MLB) (1903) became dominant sports organizations in the U.S.The FIFA World Cup (1930) – Established as the premier international soccer tournament, drawing millions of fans.The Globalization of Sports – Events like the Tour de France (1903), Formula 1 (1950), and the UEFA Champions League (1955) expanded sports audiences worldwide.The Impact of Media and Technology on SportsThe advancement of television, the internet, and sports analytics has revolutionized the way fans engage with sports. Live broadcasts, instant replays, and social media have created new ways for audiences to interact with their favorite teams and athletes.Key technological innovations in sports include:VAR (Video Assistant Referee) in soccer, improving refereeing accuracy.Wearable technology, helping athletes optimize performance and prevent injuries.Esports, a new frontier where video games like FIFA, NBA 2K, and League of Legends attract millions of spectators.sports as a Global UnifierBeyond competition, sports have been a powerful tool for social change, fostering unity and breaking barriers. Events like the Olympics, the FIFA World Cup, and the Super Bowl bring people together, transcending political and cultural differences.Athletes like Muhammad Ali, Serena Williams, and Usain Bolt have inspired generations, not only through their athletic achievements but also through their contributions to society. Sports have become a universal language, connecting people across borders and generations.Did You Know?the longest soccer match on record lasted 3 days and 2 nights in 2016, with a final score of 872-871! Michael Phelps holds the record for most Olympic gold medals, winning 23 throughout his swimming career. The highest-scoring NBA game took place in 1983, when the Denver Nuggets and Detroit Pistons played a triple-overtime thriller, ending 186-184!"
)
test_df = spark.createDataFrame([(test_text,)], ["combined_text"])

#  Tokenization
test_df = test_df.withColumn("tokens", split(col("combined_text"), " "))

# Transforming tokens into a count vector with the pre-trained CountVectorizer
test_df = cv_model.transform(test_df)

# Transforming counts in a TF-IDF vector using the already trained IDF
test_df = idf_model.transform(test_df)

# Prediction
predictions = model_ml.transform(test_df)

# Converting the prediction index to the category name using the already trained StringIndexer model
labelConverter = IndexToString(inputCol="prediction", outputCol="predicted_category", labels=label_indexer_model.labels)
predictions_with_label = labelConverter.transform(predictions)

predictions_with_label.select("combined_text", "predicted_category", "prediction", "probability").show(truncate=False)

# COMMAND ----------

display(predictions_with_label.select("combined_text", "predicted_category", "prediction", "probability"))

# COMMAND ----------

# MAGIC %md
# MAGIC ### 2° Example

# COMMAND ----------

from pyspark.sql.functions import split, col
from pyspark.ml.feature import IndexToString

test_text = (
    "The purpose of culling is to optimize the rendering process by reducing the number of objects that need to be processed by the graphics pipeline. "
    "One of the obvious caveats of culling lie in the fact that in certain situations there is a trade-off between better overall system performance and ambiguity in the system characteristics. "
    "Therefore, it is important to understand the several ways to implement culling and their specific use case benefits."
)
test_df = spark.createDataFrame([(test_text,)], ["combined_text"])

test_df = test_df.withColumn("tokens", split(col("combined_text"), " "))

test_df = cv_model.transform(test_df)

test_df = idf_model.transform(test_df)

predictions = model_ml.transform(test_df)

labelConverter = IndexToString(inputCol="prediction", outputCol="predicted_category", labels=label_indexer_model.labels)
predictions_with_label = labelConverter.transform(predictions)

predictions_with_label.select("combined_text", "predicted_category", "prediction", "probability").show(truncate=False)


# COMMAND ----------

display(predictions_with_label.select("combined_text", "predicted_category", "prediction", "probability"))

# COMMAND ----------

# MAGIC %md
# MAGIC ### 3° Example

# COMMAND ----------

from pyspark.sql.functions import split, col

test_text = (
   "TL;DR: The new CSS Anchor API revolutionizes how we position dropdowns, tooltips, and contextual menus. Say goodbye to complex positioning logic and hello to native, efficient solutions that just work.Photo by Kelly Sikkema on UnsplashPicture this: You’ve just spent hours perfecting a dropdown menu for your web app. It works great… until a user opens it at the bottom of the screen. Suddenly, your beautiful menu is cut off, trapped beneath the viewport’s edge. Sound familiar?I’ve been there. We all have. For years, web developers have wrestled with positioning contextual elements like dropdowns, tooltips, and menus. Our solutions? Often a maze of JavaScript calculations, scroll listeners, and edge-case handling makes our code more complex than it needs to be.But things are about to change dramatically.Enter the CSS Anchor API: Your New Positioning SuperpowerThe CSS Anchor API introduces a native solution to a problem that has plagued front-end developers for years. It’s like having a skilled navigator for your UI elements — they always know exactly where to go.The Basics: Anchors and PositioningAt its core, the Anchor API operates on a simple principle: establish reference points and position elements relative to them. Here’s how it works:#anchor-button {anchor-name: --my-button-anchor;#tooltip {position-anchor: --my-button-anchor;In the above example, the tooltip will position itself w.r.t the button.A tooltip positioned to the button anchorTwo Ways to Anchor: Implicit vs ExplicitThe API offers flexibility in how you reference your anchors:Implicit Anchor#tooltip {position-anchor: --my-button-anchor;top: anchor(bottom);Explicit anchor#tooltip {top: anchor(--my-button-anchor bottom);}Unlike the former, the explicit anchor allows elements to be positioned to multiple anchors, achieved by using the names of the anchors inside the anchor function.#positioned-elem {right: anchor(--first-anchor left);bottom: anchor(--second-anchor top);}Positioning Your ElementsPositioning an anchor is based on the existing absolute positioning. To position your element, add position: absolute and then use the anchor function to apply positioning values. Let’s modify the earlier example to place the tooltip at the bottom right of the button —#anchor-button {anchor-name: --my-button-anchor;}#tooltip {position-anchor: --my-button-anchor;position: absolute;top: anchor(bottom);left: anchor(right);}The tooltip positioned to the bottom rightYou can also use the logical positioning values and the calc function to achieve the desired results.The Features You Need to Know1. position-area: The 9-Cell MagicForget complex calculations. The position-area property introduces an intuitive 9-cell grid system:#anchor-button {anchor-name: --my-button-anchor;}tooltip {position-anchor: --my-button-anchor;position: absolute;position-area: bottom right;}By default, providing singular positioning values(ex: position-area: top ) positions the element on the centre of the given side.2. anchor-size: Dynamic Sizing Made SimpleAnother useful function released as part of the anchor API is the anchor-size function that allows us to size or position the element relative to the size of the anchor. Say we want the height of the tooltip not to exceed twice the height of the anchor, we can do it using the following —#anchor-button {anchor-name: --my-button-anchor;}#tooltip {position-anchor: --my-button-anchor;position: absolute;position-area: right; max-height: calc(anchor-size(height) * 2); The Secret Weapon: @position-tryThis is where the API truly shines. Remember our dropdown that disappeared off-screen? Here’s the elegant solution: @position-tryFor the moment, let’s follow the same button-tooltip example to keep things simple. I’ll show you a full-fledged example later.#anchor-button {anchor-name: --my-button-anchor;#tooltip {position-anchor: --my-button-anchor;position: absolute;position-area: right bottom;max-height: calc(anchor-size(height) * 2);position-try-fallbacks: --button-fallback;}@position-try --button-fallback {position-area: bottom;}An example of @position-tryposition-visibility: Now You See MeYou’ve built positioned elements on your page and things look great! But, what if your anchor is inside a parent with a scroller? What happens to the positioned element?The position-visibility property helps you deal with this exact case.position-visibility: anchor-visible displays the positioned element as long as 100% of the anchor is in view. This means that the positioned element may overflow the parent element.position-visibility: no-overflow displays the positioned element as long as it does not overflow the parent.A real-world exampleLet’s summarize with a practical example: the dropdown I mentioned earlier. Let’s see how you can now build dropdown components with native code and make things simpler.:popover-open {inset: auto;}:root {--width: 200px}#dropdown {background: white;border: 2px solid #05f;border-radius: 4px;color: #05f;padding: 10px 20px;anchor-name: --dropdown}e’ll build the dropdown using the popover API. You can read more in detail about the popover API here. We remove all the default styling for the popover using inset: auto . After creating and attaching anchors to the dropdown menu, and using everything we have learnt so far, we will have something like this — dropdown component exampleThe Current State and FutureWhile the CSS Anchor API represents the future of web positioning, browser support is still evolving. However, don’t let that stop you from exploring this powerful feature. Thanks to polyfills like the one from Oddbird, you can start implementing these patterns today.Browser SupportWhat’s NextStart experimenting with the CSS Anchor API in your projects. Even if you’re using polyfills initially, you’ll be ahead of the curve when native support becomes widespread. The future of web positioning is here, and it’s more elegant than we could have imagined.Further ReadingCSS anchor positioningHandling overflow"    
)
test_df = spark.createDataFrame([(test_text,)], ["combined_text"])

test_df = test_df.withColumn("tokens", split(col("combined_text"), " "))

test_df = cv_model.transform(test_df)
test_df = idf_model.transform(test_df)

predictions = model_ml.transform(test_df)

labelConverter = IndexToString(inputCol="prediction", outputCol="predicted_category", labels=label_indexer_model.labels)
predictions_with_label = labelConverter.transform(predictions)

predictions_with_label.select("combined_text", "predicted_category", "prediction", "probability").show(truncate=False)

# COMMAND ----------

display(predictions_with_label.select("combined_text", "predicted_category", "prediction", "probability"))

# COMMAND ----------

# MAGIC %md
# MAGIC ### 4° Example

# COMMAND ----------

from pyspark.sql.functions import split, col

test_text = (
   "It has been a whole month since Dungeon Economy was launched on 13th September. Thanks for all support and trust from the community! Hope you’ve been enjoying it and we are keeping optimizing it.There are six eras in Dungeon Economy, please check below for more information.1. Weapon EraIn weapon era, Dungeon Economy is a monster-based system, in this game, the “monster” is the gold in the dungeon, the “hero” is the miner, the “weapon” is the pickaxe, and “hunting and killing monsters” is mining!As productivity increased, the island fed more people, and in turn, people contributed to a more diversified economy. Some newly arrived immigrants found jobs in operating giant monster traps, and others borrowed excess monsters from the aborigines to open up businesses. People finally didn’t have to eat monsters every day, and some of them enter other business fields by loan. The diversified economy of dungeon has given birth to many occupations, such as shack builders and ground diggers etc.2. Occupation EraWith the development of dungeon economy, products export ability increased. Soon, many large freight vehicles drove into the dungeon full of monsters, costumes, armor, and weapons. These products enjoy a high reputation throughout the dungeon due to high quality and reasonable price. People used them to exchange for fresh monsters and other commodities that had never been seen in the dungeon before.3. Trade EraBecause there was no organized and orderly common defense group in the dungeon, thieves sometimes came in groups to seal monsters, rampaged in the dungeon, and the heroes were miserable. From time to time, humans on the ground also invaded the dungeon. They were not only excellent drummers, but also fierce robbers. Once they showed their power, there would be no more monsters left in the dungeon.Obviously, heroes needed to unite and jointly maintain their own safety. They needed leaders, but surrendering power was always a risk. Once people have power, they often abuse it. In order to protect the dungeon from foreign enemies, the Senate decided to establish and supervise an army composed of heroes equipped with legendary blades.4. War EraThe banks wanted to find a safe project, and finally they focused on the sleeping shed loan market in the dungeon, believing that it was an ideal target for low-risk loans.5. Real Estate eraIn the overall economic picture of the dungeon, the sheds market has never gained a prominent position before. The sheds fits hero’s way of life very well, simple and plain. However, due to economic prosperity and lower interest rates, people began to need newer, larger, and more beautiful sheds. In the past, heroes needed to save monsters for many years, and then took out a large number of monsters at once to buy a shed. Later, banks began to provide sheds loans to people with better reputations in the dungeon. With a loan, even if the borrowers’ current savings are lower than the price of the shed, they can still afford the sheds as well and don’t have to wait any longer.6. Financial EraThe profit rate of mining is not only related to the weapons of the first era, but also related to the choice of career in the second era, and the choice of career will also affect the income method of the third era. The subsequent modules of trade, war, real estate, finance, etc. are linked together, which also enables retail investors to narrow the gap with the big and professional investors through technology and luck, and thus have more fun. It’s 2021 now! Time to own a virtual wealth by your own!The dungeon NFT economic model comes from the economics book “Dungeon Economics” of the same name. This is a unique and fascinating book on economics. Starting from the dungeon, the original esoteric economic principles are vividly reflected in each story. As the story goes on, thinking about it layer by layer, you will find that although some economic principles seem to get deeper and deeper, they are always close to our reality and always full of fun.The author advocates a free economy and no excessive interference in the market. He does not advocate quantitative easing to “print money” and he believes that currency needs to increase in value, commodities need to be continuously devalued, and only the continuous rising of purchasing power of currency can bring people a truly high-quality life."    
)
test_df = spark.createDataFrame([(test_text,)], ["combined_text"])

test_df = test_df.withColumn("tokens", split(col("combined_text"), " "))

test_df = cv_model.transform(test_df)
test_df = idf_model.transform(test_df)

predictions = model_ml.transform(test_df)

labelConverter = IndexToString(inputCol="prediction", outputCol="predicted_category", labels=label_indexer_model.labels)
predictions_with_label = labelConverter.transform(predictions)

predictions_with_label.select("combined_text", "predicted_category", "prediction", "probability").show(truncate=False)

# COMMAND ----------

display(predictions_with_label.select("combined_text", "predicted_category", "prediction", "probability"))

# COMMAND ----------

# MAGIC %md
# MAGIC ### 3° Example

# COMMAND ----------

from pyspark.sql.functions import split, col

test_text = (
    "The Importance of Pets in Human Life Pets have been an integral part of human civilization for thousands of years. From loyal dogs to affectionate cats, and even exotic birds and reptiles, animals provide companionship, emotional support, and numerous health benefits to their owners. The bond between humans and pets is more than just companionship—it is a connection that improves mental well-being, physical health, and overall quality of life The History of Pet Companionshipthe domestication of animals dates back over 15,000 years, when humans first tamed wolves, which later evolved into modern-day dogs. Cats were domesticated in ancient Egypt, where they were worshiped and considered sacred animals.Over time, different species of animals became domesticated for various purposes: Dogs – Originally used for hunting and herding, now beloved as loyal companions.Cats – Valued for pest control in ancient civilizations, now known for their independent yet affectionate nature.Horses – Used for transportation, work, and even therapeutic activities.Birds, Fish, and Small Mammals – Popular as pets for their unique characteristics and ease of care.Today, pets are present in millions of households worldwide, providing love, entertainment, and even therapeutic benefits. The Benefits of Having PetsOwning a pet is not just about fun and companionship—it also brings scientifically proven health benefits: Physical Health:Walking a dog daily helps owners stay active, reducing the risk of obesity and heart disease.Studies show that pet owners have lower blood pressure and reduced stress levels. Mental Well-Being:Pets reduce anxiety and depression, offering emotional support and comfort.Therapy animals are used to assist people with PTSD, autism, and other mental health conditions.Social Benefits:Owning a pet encourages social interactions with other pet owners.Pets teach children responsibility, empathy, and patience.Therapeutic & Medical Benefits:Therapy dogs are used in hospitals and nursing homes to bring joy to patients.Some dogs are trained to detect seizures, diabetes complications, or even cancer.the unconditional love and companionship that pets provide make them irreplaceable members of the family.Most Popular Pets Around the WorldThe type of pet a person chooses depends on lifestyle, culture, and living conditions. Here are some of the most popular pets: Dogs – The most common pet worldwide, known for their loyalty and companionship. Popular breeds include Labrador Retrievers, German Shepherds, and French Bulldogs.⃣ Cats  – Independent yet affectionate, cats are great for people in smaller homes or apartments. Breeds like Persians, Maine Coons, and Siamese cats are among the favorites.⃣ Fish – Ideal for those looking for a low-maintenance pet. Popular species include goldfish, bettas, and guppies.Birds – Parrots, budgies, and canaries are popular choices for pet owners who enjoy birds' intelligence and vocal abilities.⃣ Small Mammals  – Hamsters, guinea pigs, and rabbits are great for children and small living spaces.⃣ Reptiles  – Some people prefer snakes, turtles, and lizards as exotic pets.Regardless of the species, each pet brings its own unique joy and companionship to its owner.Caring for Your Pet: Tips for Responsible OwnershipOwning a pet is a lifelong commitment. To ensure pets live long, happy lives, here are some essential tips: Nutrition & Diet:Provide pets with a balanced diet suitable for their species and breed.Avoid feeding them harmful foods like chocolate (toxic for dogs), onions, and grapes.Veterinary Care:Regular check-ups and vaccinations are crucial to prevent diseases.Spaying and neutering help control pet populations and improve health.Exercise & Mental Stimulation:Dogs need daily walks and playtime.Cats love interactive toys and scratching posts to stay active.Safe & Comfortable Living Environment:Pets need a clean, safe space with proper shelter.Avoid leaving pets alone for long periods—they need love and attention!Training, socialization, and building trust with your pet are also essential aspects of responsible pet ownership.Fun Facts About Pet Dogs’ noses are so sensitive that they can detect certain medical conditions like cancer and diabetes.Cats can jump up to six times their body length, making them one of the most agile animals.Hamsters’ teeth never stop growing, so they need to chew on wooden toys to prevent overgrowth. Goldfish have a longer memory than most people think—they can remember things for months, not seconds!Parrots can mimic human speech, and some species can even learn over 100 words!The Future of Pet OwnershipWith advancements in pet care, nutrition, and veterinary medicine, pets are living longer and healthier lives than ever before. The rise of pet technology, such as smart feeders, pet cameras, and GPS trackers, has made pet ownership more convenient.Additionally, pet adoption and rescue organizations are working hard to ensure every pet finds a loving home. Choosing adoption over buying is a great way to provide a second chance to animals in need.With the right care and attention, pets will continue to be loyal companions, stress relievers, and even lifesavers for generations to come!"
)
test_df = spark.createDataFrame([(test_text,)], ["combined_text"])

test_df = test_df.withColumn("tokens", split(col("combined_text"), " "))

test_df = cv_model.transform(test_df)
test_df = idf_model.transform(test_df)

predictions = model_ml.transform(test_df)

labelConverter = IndexToString(inputCol="prediction", outputCol="predicted_category", labels=label_indexer_model.labels)
predictions_with_label = labelConverter.transform(predictions)

predictions_with_label.select("combined_text", "predicted_category", "prediction", "probability").show(truncate=False)

# COMMAND ----------

display(predictions_with_label.select("combined_text", "predicted_category", "prediction", "probability"))

# COMMAND ----------

# MAGIC %md
# MAGIC ### 4° Example

# COMMAND ----------

from pyspark.sql.functions import split, col

test_text = (
    "An aircraft carrier is a warship that serves as a seagoing airbase, equipped with a full-length flight deck and hangar facilities for supporting, arming, deploying and recovering shipborne aircraft.[1] Typically it is the capital ship of a fleet (known as a carrier battle group), as it allows a naval force to project seaborne air power far from homeland without depending on local airfields for staging aircraft operations. Since their inception in the early 20th century, aircraft carriers have evolved from wooden vessels used to deploy individual tethered reconnaissance balloons, to nuclear-powered supercarriers that carry dozens of fighters, strike aircraft, military helicopters, AEW&Cs and other types of aircraft such as UCAVs. While heavier fixed-wing aircraft such as airlifters, gunships and bombers have been launched from aircraft carriers, these aircraft have not landed on a carrier due to flight deck limitations."
)
test_df = spark.createDataFrame([(test_text,)], ["combined_text"])

test_df = test_df.withColumn("tokens", split(col("combined_text"), " "))

test_df = cv_model.transform(test_df)
test_df = idf_model.transform(test_df)

predictions = model_ml.transform(test_df)

labelConverter = IndexToString(inputCol="prediction", outputCol="predicted_category", labels=label_indexer_model.labels)
predictions_with_label = labelConverter.transform(predictions)

predictions_with_label.select("combined_text", "predicted_category", "prediction", "probability").show(truncate=False)

# COMMAND ----------

# MAGIC %md
# MAGIC ### 5° Example

# COMMAND ----------

from pyspark.sql.functions import split, col

test_text = (
    "The Evolution of London’s Transportation System London, one of the most vibrant and bustling cities in the world, is home to an extensive and complex transportation network. With a population exceeding 9 million residents and millions of visitors each year, the efficiency of London’s transport infrastructure is crucial for the city’s economy and daily life. From its iconic red double-decker buses to the historic London Underground, the city has continuously adapted to meet growing demands.The Birth of London’s Transport SystemThe history of London’s transportation dates back to the early 19th century when horse-drawn omnibuses were introduced in 1829. These early public transport systems provided an alternative to walking for the city’s growing population.By the 1860s, London faced increasing congestion, prompting the construction of the world’s first underground railway. The Metropolitan Railway, which opened in 1863, used steam-powered trains to connect Paddington to Farringdon. Over the years, this underground railway expanded into what is now the London Underground, commonly known as the Tube. The London Underground: A Game-ChangerThe London Underground revolutionized public transport. Today, it consists of 11 lines, 272 stations, and over 400 kilometers of track, making it one of the most extensive subway systems in the world. Key features of the Underground include:The Victoria Line (Opened in 1968) – The first London Underground line to use automatic train operation. The Jubilee Line Extension (1999) – Connected Stratford to central London, boosting connectivity.Contactless and Oyster Cards – Introduced in the 2000s, these revolutionized ticketing, reducing queues and improving efficiency.Despite its success, the Tube faces challenges such as overcrowding and delays, leading to continued modernization efforts, including Crossrail (the Elizabeth Line), which aims to provide faster connections across London.London’s Iconic Red BusesLondon’s double-decker buses are one of the city’s most recognizable symbols. The first motorized bus service began in 1904, and the classic Routemaster bus was introduced in 1956.Interesting facts about London’s bus network:Over 8,600 buses operate daily, covering more than 700 routes.The network carries over 6 million passengers per day—more than the entire Tube network!London has introduced hybrid and electric buses to reduce emissions and promote eco-friendly transport.With the rise of rideshare services like Uber and bike-sharing programs, buses have had to compete with alternative transport options. However, they remain an affordable and reliable means of travel for millions. Overground and National Rail: Connecting Greater LondonIn addition to the Underground, London boasts an extensive commuter rail network. The London Overground, introduced in 2007, helps connect outer London boroughs, reducing pressure on the Tube.Other major rail services include: Thameslink – Connecting London with the South East and East of England.High-Speed Rail (HS1) – Linking London to Paris and Brussels via the Eurostar from St Pancras. Crossrail (Elizabeth Line) – A £19 billion project designed to ease congestion and connect key areas. The Role of Taxis and Ride-Sharing in LondonLondon’s black cabs are an iconic part of the city’s transport system. To become a licensed taxi driver, cabbies must pass The Knowledge, a rigorous test requiring them to memorize over 25,000 streets!With the arrival of Uber, Bolt, and Lyft, the traditional taxi industry has faced stiff competition. Ride-sharing services offer on-demand convenience, while black cabs maintain their reputation for reliability and extensive route knowledge. Cycling and Walking: A Greener FutureTo tackle traffic congestion and air pollution, London has invested in cycling infrastructure. The city launched the Santander Cycles bike-sharing scheme (formerly known as Boris Bikes) in 2010, allowing residents and tourists to rent bicycles across central London.Other sustainable initiatives include: Cycle Superhighways – Dedicated bike lanes improving safety. Low-Emission Zones (ULEZ) – Encouraging greener transport options.Pedestrian-friendly zones – Expanding walkable areas in Oxford Street and Covent Garden.The Future of London’s TransportAs London grows, so do its transportation challenges. The city is focusing on eco-friendly innovations and smart transport solutions, including: Zero-emission buses – By 2034, all London buses aim to be fully electric. Crossrail 2 – A proposed rail project to ease congestion in north and south London. Autonomous Vehicles & Smart Traffic Management – Using AI to improve road safety and efficiency.With continued investment in sustainable transport, London is set to remain one of the most well-connected cities in the world. Did You Know? The London Underground was nicknamed The Tube due to its cylindrical tunnels.Waterloo Station is the busiest railway station in the UK, handling over 94 million passengers annually.The longest cycle lane in London stretches 8 miles from Tower Bridge to Greenwich."
)
test_df = spark.createDataFrame([(test_text,)], ["combined_text"])

test_df = test_df.withColumn("tokens", split(col("combined_text"), " "))

test_df = cv_model.transform(test_df)
test_df = idf_model.transform(test_df)

predictions = model_ml.transform(test_df)

labelConverter = IndexToString(inputCol="prediction", outputCol="predicted_category", labels=label_indexer_model.labels)
predictions_with_label = labelConverter.transform(predictions)

predictions_with_label.select("combined_text", "predicted_category", "prediction", "probability").show(truncate=False)

# COMMAND ----------

display(predictions_with_label.select("combined_text", "predicted_category", "prediction", "probability"))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Naive Bayes Algorithm

# COMMAND ----------

train_df, val_df, test_df = df_final.randomSplit([0.7, 0.15, 0.15], seed=42)

# COMMAND ----------

from pyspark.ml.classification import NaiveBayes
from pyspark.ml import Pipeline
from pyspark.ml.evaluation import MulticlassClassificationEvaluator

nb = NaiveBayes(featuresCol="features", labelCol="label", smoothing=1.0, modelType="multinomial")

pipeline_nb = Pipeline(stages=[nb])

model_nb = pipeline_nb.fit(train_df)

predictions_train = model_nb.transform(train_df)
predictions_val   = model_nb.transform(val_df)
predictions_test  = model_nb.transform(test_df)

evaluator = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="accuracy")
accuracy_train = evaluator.evaluate(predictions_train)
accuracy_val   = evaluator.evaluate(predictions_val)
accuracy_test  = evaluator.evaluate(predictions_test)

print("Training Accuracy:", accuracy_train)
print("Validation Accuracy:", accuracy_val)
print("Test Accuracy:", accuracy_test)

# COMMAND ----------

import matplotlib.pyplot as plt
from pyspark.ml.evaluation import MulticlassClassificationEvaluator

evaluator = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction")

evaluator.setMetricName("accuracy")
accuracy = evaluator.evaluate(predictions_test)

evaluator.setMetricName("f1")
f1 = evaluator.evaluate(predictions_test)

evaluator.setMetricName("weightedPrecision")
precision = evaluator.evaluate(predictions_test)

evaluator.setMetricName("weightedRecall")
recall = evaluator.evaluate(predictions_test)

metrics = {
    "Accuracy": accuracy,
    "F1-Score": f1,
    "Precision": precision,
    "Recall": recall
}

for key, value in metrics.items():
    print(f"{key}: {value:.2f}")

plt.figure(figsize=(8,6))
plt.bar(metrics.keys(), metrics.values(), color="skyblue")
plt.ylim(0, 1)
for key, value in metrics.items():
    plt.text(key, value + 0.02, f"{value:.2f}", ha="center", va="bottom")
plt.ylabel("Valore")
plt.title("Metriche del Modello")
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC ### Model Prediction (Naive Bayes)

# COMMAND ----------

# MAGIC %md
# MAGIC ### 1° Example

# COMMAND ----------

from pyspark.sql.functions import split, col
from pyspark.ml.feature import IndexToString

test_text_science = (
    "An aircraft type designator is a two-, three- or four-character alphanumeric code designating every aircraft type (and some sub-types) that may appear in flight planning. These codes are defined by both the International Civil Aviation Organization (ICAO) and the International Air Transport Association (IATA).ICAO codes are published in ICAO Document 8643 Aircraft Type Designators[1] and are used by air traffic control and airline operations such as flight planning. While ICAO designators are used to distinguish between aircraft types and variants that have different performance characteristics affecting ATC, the codes do not differentiate between service characteristics (passenger and freight variants of the same type/series will have the same ICAO code)."
)

test_df_science = spark.createDataFrame([(test_text_science,)], ["combined_text"])

test_df_science = test_df_science.withColumn("tokens", split(col("combined_text"), " "))

test_df_science = cv_model.transform(test_df_science)

test_df_science = idf_model.transform(test_df_science)

predictions_science = model_nb.transform(test_df_science)

labelConverter = IndexToString(inputCol="prediction", outputCol="predicted_category", labels=label_indexer_model.labels)
predictions_science_with_label = labelConverter.transform(predictions_science)

predictions_science_with_label.select("combined_text", "predicted_category", "prediction", "probability").show(truncate=False)

# COMMAND ----------

display(predictions_science_with_label.select("combined_text", "predicted_category", "prediction", "probability"))

# COMMAND ----------

# MAGIC %md
# MAGIC ### 2° Example

# COMMAND ----------

from pyspark.sql.functions import split, col

test_text = (
    "The Importance of Pets in Human Life Pets have been an integral part of human civilization for thousands of years. From loyal dogs to affectionate cats, and even exotic birds and reptiles, animals provide companionship, emotional support, and numerous health benefits to their owners. The bond between humans and pets is more than just companionship—it is a connection that improves mental well-being, physical health, and overall quality of life The History of Pet Companionshipthe domestication of animals dates back over 15,000 years, when humans first tamed wolves, which later evolved into modern-day dogs. Cats were domesticated in ancient Egypt, where they were worshiped and considered sacred animals.Over time, different species of animals became domesticated for various purposes: Dogs – Originally used for hunting and herding, now beloved as loyal companions.Cats – Valued for pest control in ancient civilizations, now known for their independent yet affectionate nature.Horses – Used for transportation, work, and even therapeutic activities.Birds, Fish, and Small Mammals – Popular as pets for their unique characteristics and ease of care.Today, pets are present in millions of households worldwide, providing love, entertainment, and even therapeutic benefits. The Benefits of Having PetsOwning a pet is not just about fun and companionship—it also brings scientifically proven health benefits: Physical Health:Walking a dog daily helps owners stay active, reducing the risk of obesity and heart disease.Studies show that pet owners have lower blood pressure and reduced stress levels. Mental Well-Being:Pets reduce anxiety and depression, offering emotional support and comfort.Therapy animals are used to assist people with PTSD, autism, and other mental health conditions.Social Benefits:Owning a pet encourages social interactions with other pet owners.Pets teach children responsibility, empathy, and patience.Therapeutic & Medical Benefits:Therapy dogs are used in hospitals and nursing homes to bring joy to patients.Some dogs are trained to detect seizures, diabetes complications, or even cancer.the unconditional love and companionship that pets provide make them irreplaceable members of the family.Most Popular Pets Around the WorldThe type of pet a person chooses depends on lifestyle, culture, and living conditions. Here are some of the most popular pets: Dogs – The most common pet worldwide, known for their loyalty and companionship. Popular breeds include Labrador Retrievers, German Shepherds, and French Bulldogs.⃣ Cats  – Independent yet affectionate, cats are great for people in smaller homes or apartments. Breeds like Persians, Maine Coons, and Siamese cats are among the favorites.⃣ Fish – Ideal for those looking for a low-maintenance pet. Popular species include goldfish, bettas, and guppies.Birds – Parrots, budgies, and canaries are popular choices for pet owners who enjoy birds' intelligence and vocal abilities.⃣ Small Mammals  – Hamsters, guinea pigs, and rabbits are great for children and small living spaces.⃣ Reptiles  – Some people prefer snakes, turtles, and lizards as exotic pets.Regardless of the species, each pet brings its own unique joy and companionship to its owner.Caring for Your Pet: Tips for Responsible OwnershipOwning a pet is a lifelong commitment. To ensure pets live long, happy lives, here are some essential tips: Nutrition & Diet:Provide pets with a balanced diet suitable for their species and breed.Avoid feeding them harmful foods like chocolate (toxic for dogs), onions, and grapes.Veterinary Care:Regular check-ups and vaccinations are crucial to prevent diseases.Spaying and neutering help control pet populations and improve health.Exercise & Mental Stimulation:Dogs need daily walks and playtime.Cats love interactive toys and scratching posts to stay active.Safe & Comfortable Living Environment:Pets need a clean, safe space with proper shelter.Avoid leaving pets alone for long periods—they need love and attention!Training, socialization, and building trust with your pet are also essential aspects of responsible pet ownership.Fun Facts About Pet Dogs’ noses are so sensitive that they can detect certain medical conditions like cancer and diabetes.Cats can jump up to six times their body length, making them one of the most agile animals.Hamsters’ teeth never stop growing, so they need to chew on wooden toys to prevent overgrowth. Goldfish have a longer memory than most people think—they can remember things for months, not seconds!Parrots can mimic human speech, and some species can even learn over 100 words!The Future of Pet OwnershipWith advancements in pet care, nutrition, and veterinary medicine, pets are living longer and healthier lives than ever before. The rise of pet technology, such as smart feeders, pet cameras, and GPS trackers, has made pet ownership more convenient.Additionally, pet adoption and rescue organizations are working hard to ensure every pet finds a loving home. Choosing adoption over buying is a great way to provide a second chance to animals in need.With the right care and attention, pets will continue to be loyal companions, stress relievers, and even lifesavers for generations to come!"
)
test_df = spark.createDataFrame([(test_text,)], ["combined_text"])

test_df = test_df.withColumn("tokens", split(col("combined_text"), " "))

test_df = cv_model.transform(test_df)
test_df = idf_model.transform(test_df)

predictions = model_nb.transform(test_df)

labelConverter = IndexToString(inputCol="prediction", outputCol="predicted_category", labels=label_indexer_model.labels)
predictions_with_label = labelConverter.transform(predictions)

predictions_with_label.select("combined_text", "predicted_category", "prediction", "probability").show(truncate=False)

# COMMAND ----------

display(predictions_with_label.select("combined_text", "predicted_category", "prediction", "probability"))

# COMMAND ----------

# MAGIC %md
# MAGIC ### 3° Example

# COMMAND ----------

from pyspark.sql.functions import split, col
from pyspark.ml.feature import IndexToString

test_text = (
    "This is a quick update, as many writers keep asking whether there has been any news since my farewell story. Many have expressed frustration over the silence, and I have noticed a few writers sharing their thoughts in various publications now, the silence is suddenly broken with harsh words.As of now, there is still no formal update, apart from Brittany’s latest message in my farewell story, where she encouraged patience, graciously accepted the situation, and showed willingness to investigate it. Before that, Scott acknowledged the concerns as an update in his viral blog, stating, “We hear you, we are working on it, and we will keep you updated. Thank you for your patience.” Ironically, a stupid AI detector, as mentioned by Dr Broaly, thought these wonderful words were likely written by AI.However, today, in a well-written piece by an experienced writer whom I will cite below, the CEO stated that there was no payment issue — only a handful of people making noise among themselves. He also mentioned that there was communication including an email to affected people.Since Scott’s viral story, there has been no official announcement, at least as far as I know. None of my close circles, including around 500 of my protégés, received an email from Medium. I am not claiming that Medium did not send one, as email systems can fail. I personally lost a significant portion of my income, yet I did not receive any communication. If you have received an email, please leave a comment on this story.This message is not about criticizing the CEO. I have no conflict of interest with him. Some people support him; others do not. Personally, I support him and have always appreciated his writing, even before he became CEO. I also recognize the weight of his role — what we see as users and what he sees as a leader can be entirely different.As I mentioned in a previous comment, Medium is a black box. No one truly knows how it operates, how the algorithm works, who runs the company, or who the curators are. I have observed its patterns for six years, analyzing significant public data, and even when I was a boost nominator, I could not fully decipher its workings. It is also possible that the CEO does not have complete freedom to set the strategy. The reality is — we are all speculating."
)

test_df = spark.createDataFrame([(test_text,)], ["combined_text"])

test_df = test_df.withColumn("tokens", split(col("combined_text"), " "))

test_df = cv_model.transform(test_df)

test_df = idf_model.transform(test_df)

predictions = model_nb.transform(test_df)

labelConverter = IndexToString(inputCol="prediction", outputCol="predicted_category", labels=label_indexer_model.labels)
predictions_with_label = labelConverter.transform(predictions)

predictions_with_label.select("combined_text", "predicted_category", "prediction", "probability").show(truncate=False)

# COMMAND ----------

display(predictions_with_label.select("combined_text", "predicted_category", "prediction", "probability"))

# COMMAND ----------

# MAGIC %md
# MAGIC ### 4° Example

# COMMAND ----------

from pyspark.sql.functions import split, col
from pyspark.ml.feature import IndexToString

test_text = (
    "There’s a lot of talk about assigning responsibility for the fires that are happening in Los Angeles right now to the liberal Democratic governor of California, Gavin Newsom. This is absurd. I will go out on a limb and say that the people making such statements have never been to Los Angeles and have never hiked in the hills around Los Angeles. I have. I lived in Los Angeles around the turn of the century. That phrase makes me feel old. That was a quarter century ago—half my lifetime.Anyway.No reasonable measures could have been taken to prevent these fires. The wildlands around Los Angeles are not some old-growth forest that needs to be clear-cut in strips to prevent the spread of forest fires. The reasons for this are the types of plant cover, the severity of the winds, and the extreme levels of drought. No water system in the world could have supplied sufficient water to the fire hydrants.The plant cover is mostly brush grasses and other fast-growing plants that grow back within a few days, weeks, or months after being cut down. Cutting them back would result in severe erosion and dust storms, as ground cover would no longer protect the dirt. Cutting back the vegetation would be trading one disaster for another — losing what little ground moisture is there would still result in fires. Fires are a natural part of the growth cycle for plants — any form of human interference worsens the problem. Programs for the vegetation to be cut back result in fewer short-term fires but more severe ones in the long term. Cutting back a slow-growth forest is helpful in the short term, but the area around Los Angeles is not a slow-growth forest.The next problem is the severity of the drought. This problem is a direct result of anthropogenic climate change, and there is nothing a local government can do to prevent the drought, given our national and international failure to take steps to prevent climate change. That boat has sailed. Blaming Newsom for the drought is like blaming DeSantis for hurricanes. We can blame them for the response but not for the act itself. As far as I can tell, the fire response has been as best as expected. Perfect? Of course not. Human endeavors rarely are. But reasonable."
)

test_df = spark.createDataFrame([(test_text,)], ["combined_text"])

test_df = test_df.withColumn("tokens", split(col("combined_text"), " "))

test_df = cv_model.transform(test_df)

test_df = idf_model.transform(test_df)

predictions = model_nb.transform(test_df)

labelConverter = IndexToString(inputCol="prediction", outputCol="predicted_category", labels=label_indexer_model.labels)
predictions_with_label = labelConverter.transform(predictions)

predictions_with_label.select("combined_text", "predicted_category", "prediction", "probability").show(truncate=False)

# COMMAND ----------

display(predictions_with_label.select("combined_text", "predicted_category", "prediction", "probability"))

# COMMAND ----------

# MAGIC %md
# MAGIC ### 5° Example

# COMMAND ----------

from pyspark.sql.functions import split, col
from pyspark.ml.feature import IndexToString

test_text = ("Transport for London (TfL) is a local government body responsible for most of the transport network in London, United Kingdom.[2]TfL is the successor organization of the London Passenger Transport Board, which was established in 1933, and several other bodies in the intervening years. Since the current organization's creation in 2000 as part of the Greater London Authority (GLA), TfL has been responsible for operating multiple urban rail networks, including the London Underground and Docklands Light Railway, as well as London's buses, taxis, principal road routes, cycling provision, trams, and river services. It does not control all National Rail services in London, although it is responsible for London Overground and Elizabeth line services. The underlying services are provided by a mixture of wholly owned subsidiary companies (principally London Underground), by private sector franchisees (the remaining rail services, trams and most buses) and by licensees (some buses, taxis and river services). Fares are controlled by TfL, rail services fares calculated using numbered zones across the capital.TfL has overseen various initiatives and infrastructure projects. Throughout the 2000s, a new radio communication system was implemented across its underground lines. Passenger convenience systems, such as the Oyster card and contactless payments, were also provisioned around this time. During 2008, the consumption of alcohol was banned on TfL services; this move has led to a decrease in anti-social behaviour. On 16 August 2016, TfL oversaw the launch of the Night Tube scheme, which introduced through-the-night services on both the London Underground and London Overground. Perhaps the biggest undertaking it has been responsible for, in this case shared jointly with the national Department for Transport (DfT), was the commissioning of the Crossrail Project; since its completion in 2022, TfL has been responsible for franchising its operation as the Elizabeth line.[3]In addition to the GLA, the central British government used to provide regular funding for TfL. However, this was tapered off during the 2010s with the aim of the organisation becoming self-sufficient. Direct central government funding for operations ceased during 2018.[1] During 2019–2020, TfL had a budget of £10.3 billion, 47% of which came from fares; the remainder came from grants, mainly from the GLA (33%), borrowing (8%), congestion charging and other income (12%). In 2020, during the height of the COVID-19 pandemic, fare revenues dropped by 90% and TfL obtained multiple rounds of support from the British government. It also responded with various cutbacks, including a proposal for a 40% reduction in capital expenditure.HistorySee also: History of public transport authorities in LondonLogo prior to 2013London's transportation system was unified in 1933, with the creation of the London Passenger Transport Board, which was succeeded by London Transport Executive, London Transport Board, London Transport Executive (GLC), and London Regional Transport. From 1933 until 2000, these bodies used the London Transport brand.[4]Transport for London was created in 2000 as part of the Greater London Authority (GLA) by the Greater London Authority Act 1999.[5] The first Commissioner of TfL was Bob Kiley.[6] The first chair was then-Mayor of London Ken Livingstone, and the first deputy chair was Dave Wetzel. Livingstone and Wetzel remained in office until the election of Boris Johnson as Mayor in 2008. Johnson took over as chairman, and in February 2009 fellow-Conservative Daniel Moylan was appointed as his deputy.Transport for London Corporate Archives holds business records for TfL and its predecessor bodies and transport companies. Some early records are also held on behalf of TfL Corporate Archives at the London Metropolitan Archives.On 17 February 2003, the London congestion charge was introduced, covering the approximate area of the London Inner Ring Road.[7] The congestion charge had been a manifesto promise by Ken Livingstone during the 2000 London Mayoral election.[8] It was introduced to reduce congestion in the centre of the capital as well as to make London more attractive to business investment; the resulting revenue was to be invested in London's transport system.[9] At the time of its implementation, the scheme was the largest ever undertaken by a capital city.[10]During 2003, TfL took over responsibility for the London Underground, after terms for a controversial public-private partnership (PPP) maintenance contract had been agreed.[11][12] While the Underground trains themselves were operated by the public sector, the infrastructure (track, trains, tunnels, signals, and stations) were to be leased to private firms for 30 years, during which these companies would implement various improvements.[13] The two consortiums awarded contracts were Tube Lines and Metronet.[14][15] In July 2007, following financial difficulties, Metronet was placed in administration and its responsibilities were transferred back into public ownership under TfL in May 2008.[16][17] During 2009, Tube Lines, having encountered a funding shortfall for its upgrades, was denied a request to TfL for an additional £1.75 billion; the matter was instead referred to the PPP arbiter, who stated that £400 million should be provided.[18][19] On 7 May 2010, Transport for London agreed to buy out Bechtel and Amey (Ferrovial), the shareholders of Tube Lines for £310 million, formally ending the PPP.[20][21]TfL was heavily impacted by multiple bombings on the underground and bus systems on 7 July 2005. Numerous TfL staff were recognised in the 2006 New Year honours list for the actions taken on that day, including aiding survivors, removing bodies, and restoring the transport system so that millions of commuters were able to depart London at the end of the workday.[a] The incident was heavily scrutinised, leading to various long term changes being proposed by groups such as London Assembly, including the accelerated implementation of underground radio connectivity.[25]On 20 February 2006, the DfT announced that TfL would take over management of services then provided by Silverlink Metro.[26][27][28] On 5 September 2006, the London Overground branding was announced, and it was confirmed that the extended East London line would be included.[29] On 11 November 2007, TfL took over the North London Railway routes from Silverlink Metro. At the launch, TfL undertook to revamp the routes by improving service frequencies and station facilities, staffing all stations, introducing new rolling stock and allowing Oyster pay as you go throughout the network from the outset.[30] This launch was accompanied by a marketing campaign entitled Londons new train set, with posters and leaflets carrying an image of model railway packaging containing new Overground trains, tracks and staff.[31]On 1 June 2008, the drinking of alcoholic beverages was banned on Tube and London Overground trains, buses, trams, Docklands Light Railway and all stations operated by TfL across London but not those operated by other rail companies.[32][33] Carrying open containers of alcohol was also banned on public transport operated by TfL. The then-Mayor of London Boris Johnson and TfL announced the ban with the intention of providing a safer and more pleasant experience for passengers. There were Last Round on the Underground parties on the night before the ban came into force. Passengers refusing to observe the ban may be refused travel and asked to leave the premises. The GLA reported in 2011 that assaults on London Underground staff had fallen by 15% since the introduction of the ban.[34]Between 2008 and 2022, TfL was engaged in the Crossrail programme to construct a new high-frequency hybrid urban–suburban rail service across London and into its suburbs.[35] TfL Rail took over Heathrow Connect services from Paddington to Heathrow in May 2018.[36][37] In August 2018, four months before the scheduled opening of the core section of the Elizabeth Line, it was announced that completion had been delayed and that the line would not open before autumn 2019.[38] Further postponements ensued.[39] Having an initial budget of £14.8 billion, the total cost of Crossrail rose to £18.25 billion by November 2019,[40][41] and increased further to £18.8 billion by December 2020.[42] On 17 May 2022, the line was officially opened by Queen Elizabeth II in honour of her Platinum Jubilee.[43]TfL commissioned a survey in 2013 which showed that 15% of women using public transport in London had been the subject of some form of unwanted sexual behaviour but that 90% of incidents were not reported to the police. In an effort to reduce sexual offences and increase reporting, TfL—in conjunction with the British Transport Police, Metropolitan Police Service, and City of London Police—launched Project Guardian.[44] In 2014, TfL launched the 100 years of women in transport campaign in partnership with the Department for Transport, Crossrail,[45] Network Rail,[46] the WomensEngineeringSociety[47] and the WomenTransportation Seminar (WTS). The programme was a celebration of the significant role that women had played in transport over the previous 100 years, following the centennial anniversary of the First World War, when 100,000 women entered the transport industry to take on the responsibilities held by men who enlisted for military service.[48]As early as 2014, an Ultra–Low Emission Zone (ULEZ) was under consideration since 2014 under London Mayor Boris Johnson.[49] Johnson announced in 2015 that the zone covering the same areas as the congestion charge would come into operation in September 2020. Sadiq Khan, Johnson successor, introduced an emissions surcharge, called the Toxicity Charge or T-Charge, for non-compliant vehicles from 2017.[50][51] The Toxicity Charge was replaced by the Ultra Low Emission Zone on 8 April 2019, which was introduced ahead of schedule. On 29 August 2023, the ULEZ was expanded to cover all 32 London boroughs, bringing an additional five million people into the zone.[52]During 2020, passenger numbers, along with associated revenue, went into a sharp downturn as a result of the COVID-19 pandemic in the United Kingdom. In response, TfL services were reduced; specifically, all Night Overground and Night Tube services, as well as all services on the Waterloo & City line, were suspended from 20 March, while 40 tube stations were closed on the same day.[53] The Mayor of London and TfL urged people to only use public transport if absolutely essential so that it could be used by critical workers.[54] The London Underground brought in new measures on 25 March to combat the spread of the virus; these included slowing the flow of passengers onto platforms via the imposition of queuing at ticket gates and turning off some escalators.[55] In April, TfL trialled changes encouraging passengers to board London buses by the middle doors to lessen the risks to drivers, after the deaths of 14 TfL workers including nine drivers.[56] This measure was extended to all routes on 20 April, and passengers were no longer required to pay, so that they did not need to use the card reader near the driver.[57]"
)

test_df = spark.createDataFrame([(test_text,)], ["combined_text"])

test_df = test_df.withColumn("tokens", split(col("combined_text"), " "))

test_df = cv_model.transform(test_df)

test_df = idf_model.transform(test_df)

predictions = model_nb.transform(test_df)

labelConverter = IndexToString(inputCol="prediction", outputCol="predicted_category", labels=label_indexer_model.labels)
predictions_with_label = labelConverter.transform(predictions)

predictions_with_label.select("combined_text", "predicted_category", "prediction", "probability").show(truncate=False)

# COMMAND ----------

display(predictions_with_label.select("combined_text", "predicted_category", "prediction", "probability"))

# COMMAND ----------

# MAGIC %md
# MAGIC ### 6° Example

# COMMAND ----------

from pyspark.sql.functions import split, col
from pyspark.ml.feature import IndexToString

test_text = ("TL;DR: The new CSS Anchor API revolutionizes how we position dropdowns, tooltips, and contextual menus. Say goodbye to complex positioning logic and hello to native, efficient solutions that just work.Photo by Kelly Sikkema on UnsplashPicture this: You’ve just spent hours perfecting a dropdown menu for your web app. It works great… until a user opens it at the bottom of the screen. Suddenly, your beautiful menu is cut off, trapped beneath the viewport’s edge. Sound familiar?I’ve been there. We all have. For years, web developers have wrestled with positioning contextual elements like dropdowns, tooltips, and menus. Our solutions? Often a maze of JavaScript calculations, scroll listeners, and edge-case handling makes our code more complex than it needs to be.But things are about to change dramatically.Enter the CSS Anchor API: Your New Positioning SuperpowerThe CSS Anchor API introduces a native solution to a problem that has plagued front-end developers for years. It’s like having a skilled navigator for your UI elements — they always know exactly where to go.The Basics: Anchors and PositioningAt its core, the Anchor API operates on a simple principle: establish reference points and position elements relative to them. Here’s how it works:#anchor-button {anchor-name: --my-button-anchor;#tooltip {position-anchor: --my-button-anchor;In the above example, the tooltip will position itself w.r.t the button.A tooltip positioned to the button anchorTwo Ways to Anchor: Implicit vs ExplicitThe API offers flexibility in how you reference your anchors:Implicit Anchor#tooltip {position-anchor: --my-button-anchor;top: anchor(bottom);Explicit anchor#tooltip {top: anchor(--my-button-anchor bottom);}Unlike the former, the explicit anchor allows elements to be positioned to multiple anchors, achieved by using the names of the anchors inside the anchor function.#positioned-elem {right: anchor(--first-anchor left);bottom: anchor(--second-anchor top);}Positioning Your ElementsPositioning an anchor is based on the existing absolute positioning. To position your element, add position: absolute and then use the anchor function to apply positioning values. Let’s modify the earlier example to place the tooltip at the bottom right of the button —#anchor-button {anchor-name: --my-button-anchor;}#tooltip {position-anchor: --my-button-anchor;position: absolute;top: anchor(bottom);left: anchor(right);}The tooltip positioned to the bottom rightYou can also use the logical positioning values and the calc function to achieve the desired results.The Features You Need to Know1. position-area: The 9-Cell MagicForget complex calculations. The position-area property introduces an intuitive 9-cell grid system:#anchor-button {anchor-name: --my-button-anchor;}tooltip {position-anchor: --my-button-anchor;position: absolute;position-area: bottom right;}By default, providing singular positioning values(ex: position-area: top ) positions the element on the centre of the given side.2. anchor-size: Dynamic Sizing Made SimpleAnother useful function released as part of the anchor API is the anchor-size function that allows us to size or position the element relative to the size of the anchor. Say we want the height of the tooltip not to exceed twice the height of the anchor, we can do it using the following —#anchor-button {anchor-name: --my-button-anchor;}#tooltip {position-anchor: --my-button-anchor;position: absolute;position-area: right; max-height: calc(anchor-size(height) * 2); The Secret Weapon: @position-tryThis is where the API truly shines. Remember our dropdown that disappeared off-screen? Here’s the elegant solution: @position-tryFor the moment, let’s follow the same button-tooltip example to keep things simple. I’ll show you a full-fledged example later.#anchor-button {anchor-name: --my-button-anchor;#tooltip {position-anchor: --my-button-anchor;position: absolute;position-area: right bottom;max-height: calc(anchor-size(height) * 2);position-try-fallbacks: --button-fallback;}@position-try --button-fallback {position-area: bottom;}An example of @position-tryposition-visibility: Now You See MeYou’ve built positioned elements on your page and things look great! But, what if your anchor is inside a parent with a scroller? What happens to the positioned element?The position-visibility property helps you deal with this exact case.position-visibility: anchor-visible displays the positioned element as long as 100% of the anchor is in view. This means that the positioned element may overflow the parent element.position-visibility: no-overflow displays the positioned element as long as it does not overflow the parent.A real-world exampleLet’s summarize with a practical example: the dropdown I mentioned earlier. Let’s see how you can now build dropdown components with native code and make things simpler.:popover-open {inset: auto;}:root {--width: 200px}#dropdown {background: white;border: 2px solid #05f;border-radius: 4px;color: #05f;padding: 10px 20px;anchor-name: --dropdown}e’ll build the dropdown using the popover API. You can read more in detail about the popover API here. We remove all the default styling for the popover using inset: auto . After creating and attaching anchors to the dropdown menu, and using everything we have learnt so far, we will have something like this — dropdown component exampleThe Current State and FutureWhile the CSS Anchor API represents the future of web positioning, browser support is still evolving. However, don’t let that stop you from exploring this powerful feature. Thanks to polyfills like the one from Oddbird, you can start implementing these patterns today.Browser SupportWhat’s NextStart experimenting with the CSS Anchor API in your projects. Even if you’re using polyfills initially, you’ll be ahead of the curve when native support becomes widespread. The future of web positioning is here, and it’s more elegant than we could have imagined.Further ReadingCSS anchor positioningHandling overflow"    
)

test_df = spark.createDataFrame([(test_text,)], ["combined_text"])

test_df = test_df.withColumn("tokens", split(col("combined_text"), " "))

test_df = cv_model.transform(test_df)

test_df = idf_model.transform(test_df)

predictions = model_nb.transform(test_df)

labelConverter = IndexToString(inputCol="prediction", outputCol="predicted_category", labels=label_indexer_model.labels)
predictions_with_label = labelConverter.transform(predictions)

predictions_with_label.select("combined_text", "predicted_category", "prediction", "probability").show(truncate=False)

# COMMAND ----------

display(predictions_with_label.select("combined_text", "predicted_category", "prediction", "probability"))

# COMMAND ----------

# MAGIC %md
# MAGIC ### 7° Example

# COMMAND ----------

from pyspark.sql.functions import split, col
from pyspark.ml.feature import IndexToString

test_text = (
    "The Evolution of Sports: From Ancient Games to Modern CompetitionsSports have played a significant role in human history, evolving from ancient competitions to modern professional leagues and international tournaments. The love for sports transcends cultures, bringing people together through physical excellence, teamwork, and competition.Ancient Origins of SportsThe history of sports dates back thousands of years, with evidence of organized physical contests found in Ancient Egypt, Greece, and China. The Olympic Games, first held in 776 BCE in Olympia, Greece, were one of the earliest recorded sporting events. These games featured disciplines such as:Running (stadion race) – A short sprint, considered the most prestigious event.Wrestling and boxing – Physical combat sports that tested endurance and technique.Chariot racing – A thrilling and dangerous sport popular among the Greeks and later the Romans.Other civilizations had their own sports traditions. The Mesoamerican ballgame, played by the Maya and Aztecs, involved a rubber ball and a hoop-like goal, while the Chinese cuju is considered an early form of soccer.The Rise of Modern SportsThe modern era of sports began in the 19th and early 20th centuries, with the formalization of rules and the establishment of international competitions. Many of today’s most popular sports were standardized during this period, including:Soccer (Football) – The Football Association (FA) was founded in 1863 in England, creating a uniform rule set for the game.Basketball – Invented in 1891 by Dr. James Naismith, it became one of the most popular sports worldwide.Tennis and Golf – Developed in the late 19th century, with international tournaments like Wimbledon (1877) and The Open Championship (1860) emerging.The re-establishment of the Olympic Games in 1896, spearheaded by Pierre de Coubertin, marked a turning point for global sports, introducing a new platform for international competition.The Professionalization of SportsBy the 20th century, sports had become more than just recreational activities—they transformed into multibillion-dollar industries. The rise of professional leagues, sponsorships, and television broadcasts allowed athletes to compete at the highest level while earning significant financial rewards.Some key moments in professional sports history include:The Formation of Major Leagues – The National Football League (NFL) (1920), National Basketball Association (NBA) (1946), and Major League Baseball (MLB) (1903) became dominant sports organizations in the U.S.The FIFA World Cup (1930) – Established as the premier international soccer tournament, drawing millions of fans.The Globalization of Sports – Events like the Tour de France (1903), Formula 1 (1950), and the UEFA Champions League (1955) expanded sports audiences worldwide.The Impact of Media and Technology on SportsThe advancement of television, the internet, and sports analytics has revolutionized the way fans engage with sports. Live broadcasts, instant replays, and social media have created new ways for audiences to interact with their favorite teams and athletes.Key technological innovations in sports include:VAR (Video Assistant Referee) in soccer, improving refereeing accuracy.Wearable technology, helping athletes optimize performance and prevent injuries.Esports, a new frontier where video games like FIFA, NBA 2K, and League of Legends attract millions of spectators.sports as a Global UnifierBeyond competition, sports have been a powerful tool for social change, fostering unity and breaking barriers. Events like the Olympics, the FIFA World Cup, and the Super Bowl bring people together, transcending political and cultural differences.Athletes like Muhammad Ali, Serena Williams, and Usain Bolt have inspired generations, not only through their athletic achievements but also through their contributions to society. Sports have become a universal language, connecting people across borders and generations.Did You Know?the longest soccer match on record lasted 3 days and 2 nights in 2016, with a final score of 872-871! Michael Phelps holds the record for most Olympic gold medals, winning 23 throughout his swimming career. The highest-scoring NBA game took place in 1983, when the Denver Nuggets and Detroit Pistons played a triple-overtime thriller, ending 186-184!"
)
test_df = spark.createDataFrame([(test_text,)], ["combined_text"])

test_df = test_df.withColumn("tokens", split(col("combined_text"), " "))

test_df = cv_model.transform(test_df)

test_df = idf_model.transform(test_df)

predictions = model_nb.transform(test_df)

labelConverter = IndexToString(inputCol="prediction", outputCol="predicted_category", labels=label_indexer_model.labels)
predictions_with_label = labelConverter.transform(predictions)

predictions_with_label.select("combined_text", "predicted_category", "prediction", "probability").show(truncate=False)

# COMMAND ----------

display(predictions_with_label.select("combined_text", "predicted_category", "prediction", "probability"))

# COMMAND ----------

# MAGIC %md
# MAGIC ### 8° Example

# COMMAND ----------

from pyspark.sql.functions import split, col
from pyspark.ml.feature import IndexToString

test_text = ("The Golden Age of Video Games: The 1980s RevolutionThe 1980s were a transformative decade for the gaming industry, marking the beginning of what is often called the Golden Age of Video Games. This era saw the rise of arcade machines, home consoles, and the birth of some of the most iconic franchises that continue to shape gaming today.The Rise of Arcade GamingAt the start of the decade, arcades were the dominant force in gaming. Classic arcade games such as Pac-Man (1980), Donkey Kong (1981), and Galaga (1981) became worldwide sensations, filling arcades with players eager to set high scores. Pac-Man, created by Namco, was particularly revolutionary, introducing a non-violent gameplay mechanic in a market dominated by space shooters. Meanwhile, Donkey Kong, designed by Nintendo, marked the debut of Mario, one of the most recognizable characters in gaming history.During this period, companies like Atari, Namco, Taito, and Williams Electronics developed cutting-edge arcade machines that featured colorful graphics, responsive controls, and addictive gameplay. The introduction of joysticks, trackballs, and light guns allowed developers to experiment with new gameplay mechanics. Multiplayer games also began to emerge, with titles like Joust (1982) and Gauntlet (1985) encouraging cooperative play.The Home Console RevolutionAs arcade machines dominated public gaming spaces, the home console market was evolving rapidly. Atari 2600, originally released in 1977, reached peak popularity in the early '80s, bringing arcade-quality games to living rooms. However, it was Nintendo's Famicom (1983), known internationally as the Nintendo Entertainment System (NES) (1985), that truly revolutionized home gaming.The NES introduced legendary games such as:Super Mario Bros. (1985) – Defined the platformer genre and became one of the best-selling games of all time.The Legend of Zelda (1986) – A pioneer in open-world exploration and non-linear gameplay.Metroid (1986) – Introduced atmospheric storytelling and inspired the 'Metroidvania' genre.These games, combined with Nintendo's strict licensing policies for third-party developers, helped revitalize the gaming industry after the 1983 video game crash, a period of decline caused by oversaturation of low-quality games.The Birth of PC GamingWhile consoles and arcades were dominating mainstream gaming, personal computers were becoming a viable gaming platform. The introduction of systems like the Commodore 64 (1982), ZX Spectrum (1982), and IBM PC (1981) allowed developers to create games that offered deeper mechanics and storytelling.Notable PC games from the 1980s included:Prince of Persia (1989) – A groundbreaking action-platformer known for its fluid animation.Ultima IV: Quest of the Avatar (1985) – An RPG that emphasized moral choices over combat.King’s Quest (1984) – One of the first graphical adventure games, paving the way for storytelling in video games.Many of these titles laid the groundwork for modern PC gaming, introducing elements such as text-based commands, point-and-click interfaces, and open-ended gameplay.The Influence of the 1980s on Modern GamingThe legacy of 1980s video games is still felt today. Many of the industry’s biggest franchises, including Super Mario, Zelda, Final Fantasy, and Metroid, originated in this decade. Moreover, game design concepts such as side-scrolling platformers, RPG progression systems, and interactive storytelling were pioneered during this period.The resurgence of retro gaming, with mini console re-releases like the NES Classic Edition (2016) and the popularity of pixel-art indie games, demonstrates how the 1980s continue to inspire game developers and players alike.Did you know? The arcade high score culture of the '80s led to competitive gaming and early esports. Games like Street Fighter (1987) and Tetris (1984) were the foundation for what would later become the global competitive gaming scene."    
)

test_df = spark.createDataFrame([(test_text,)], ["combined_text"])

test_df = test_df.withColumn("tokens", split(col("combined_text"), " "))

test_df = cv_model.transform(test_df)

test_df = idf_model.transform(test_df)

predictions = model_nb.transform(test_df)

labelConverter = IndexToString(inputCol="prediction", outputCol="predicted_category", labels=label_indexer_model.labels)
predictions_with_label = labelConverter.transform(predictions)

predictions_with_label.select("combined_text", "predicted_category", "prediction", "probability").show(truncate=False)

# COMMAND ----------

display(predictions_with_label.select("combined_text", "predicted_category", "prediction", "probability"))

# COMMAND ----------

# MAGIC %md
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC #### _Despite the 'Max Word Count for Category' being very unbalanced among the different categories, and despite the CountVectorizer being applied directly on the raw data (without lemmatization and without removing stopwords), the classifiers proved to be highly performing. Moreover, despite not having applied the Sliding Window technique to filter only the English words, the models obtained satisfactory results._

# COMMAND ----------

# MAGIC %md
# MAGIC #### _The insights obtained from exploratory analysis and classification will allow Wikimedia to optimize the allocation of editorial resources, with the possibility of directing its information campaigns in a more targeted way._
