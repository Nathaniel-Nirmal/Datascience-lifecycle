from pyspark import SparkContext, SparkConf
import re
import sys

# Initialize Spark configuration and context
conf = SparkConf().setAppName("BigramModel")
sc = SparkContext.getOrCreate(conf=conf)


def preprocess_line(line):
    line = line.lower()
    line = re.sub(
        "[^a-z\s]+", " ", line
    )  # Replace all non-alphabetic characters with space
    words = line.split()
    chunk_size = 100
    return [words[i : i + chunk_size] for i in range(0, len(words), chunk_size)]


def generate_bigrams(words_list):
    return [(words_list[i], words_list[i + 1]) for i in range(len(words_list) - 1)]


text_file = sc.textFile("wiki.txt")

words_chunks = text_file.flatMap(preprocess_line)

bigrams = words_chunks.flatMap(generate_bigrams)

bigram_counts = bigrams.map(lambda bigram: (bigram, 1)).reduceByKey(lambda x, y: x + y)

word_counts = (
    words_chunks.flatMap(lambda x: x)
    .map(lambda word: (word, 1))
    .reduceByKey(lambda x, y: x + y)
)

bigram_counts_for_join = bigram_counts.map(lambda x: (x[0][0], (x[0][1], x[1])))
conditional_bigrams = bigram_counts_for_join.join(word_counts)

conditional_distribution = conditional_bigrams.map(
    lambda x: ((x[0], x[1][0][0]), x[1][0][1] / x[1][1])
)

bigram_counts_output_path = "output/bigram_counts"
conditional_distribution_output_path = "output/conditional_distribution"

# bigram_counts.saveAsTextFile(bigram_counts_output_path)
bigram_counts.coalesce(1, shuffle=True).saveAsTextFile(bigram_counts_output_path)
conditional_distribution.coalesce(1, shuffle=True).saveAsTextFile(
    conditional_distribution_output_path
)
# conditional_distribution.saveAsTextFile(conditional_distribution_output_path)

sc.stop()
