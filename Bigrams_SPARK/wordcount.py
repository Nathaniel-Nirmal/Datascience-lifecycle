import sys
from pyspark import SparkContext, SparkConf

conf = SparkConf()
sc = SparkContext(conf=conf)

words = sc.textFile(sys.argv[1]).flatMap(lambda line: line.split(" "))

wordCounts = words.map(lambda word: (word, 1)).reduceByKey(lambda a, b: a + b)

word_counts_output_path = "output/word_counts"

wordCounts.coalesce(1, shuffle=True).saveAsTextFile(word_counts_output_path)
sc.stop()
