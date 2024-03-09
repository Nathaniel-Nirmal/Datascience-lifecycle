import sys
from pyspark.sql import SparkSession
import argparse
from pyspark.sql.types import *
# Convert function to udf
from pyspark.sql.functions import col, udf, array

#feel free to def new functions if you need

def create_dataframe(filepath, format, spark):
    """
    Create a spark df given a filepath and format.

    :param filepath: <str>, the filepath
    :param format: <str>, the file format (e.g. "csv" or "json")
    :param spark: <str> the spark session

    :return: the spark df uploaded
    """

    #add your code here
    if format.lower() == "csv":
        spark_df = spark.read.format("csv").option("header", "true").load(filepath)
    elif format.lower() == "json":
        spark_df = spark.read.format("json").load(filepath)
    else:
        raise ValueError("Unsupported file format: {}".format(format))

    return spark_df

def tranformAgeFromNhisToBrfss(age, ageDict = {1 : (18, 24), 2 : (25, 29), 3 : (30, 34), 4 : (35, 39), 5 : (40, 44), 6 : (45, 49), 7 : (50, 54), 8 : (55, 59), 9 : (60, 64), 10 : (65, 69), 11 : (70, 74), 12 : (74, 79), 13 : (80, 99)}):
    if age is None:
        return 14.0
    for key in ageDict.keys():
        if ageDict[int(key)][0] <= int(age) <= ageDict[int(key)][1]:
            return float(key)
    raise ValueError("Age is neither None nor within possible ranges. This is not what is expected. Add code to handle this. Age : ", age)

def tranformRaceFromNhisToBrfss(race):
    MRACBPI2 = int(race[0])
    HISPAN_I = int(race[1])
    raceMap = {1.0 : (1, 12), 2.0 : (2, 12), 3.0 : ((6, 7, 12), 12), 4.0 : (3, 12), 5.0 : (-1, tuple([i for i in range(12)])), 6.0 : (16, 12)} # -1 : any value will do. This map maps race value from brfss's race column(_IMPRACE) to values in columns of Nhis containit races(MRACBPI2 and HISPAN_I)
    if MRACBPI2 is None or HISPAN_I is None:
        return None
    for key in raceMap.keys():
        if type(raceMap[key][0]) == tuple:
            for subkey in raceMap[key][0]:
                if (MRACBPI2 == int(subkey)) and HISPAN_I == raceMap[key][1]:
                    return key
        elif type(raceMap[key][1]) == tuple:
            for subkey in raceMap[key][1]:
                if HISPAN_I == int(subkey):
                    return key
        elif MRACBPI2 == raceMap[key][0] and HISPAN_I == raceMap[key][1]:
            return key
    raise ValueError(f"Both MRACBPI2 and HISPAN_I are non-null but no brfss match is their as per raceMap. This is not what is expected. Add code to handle this. MRACBPI2 : {MRACBPI2}, and HISPAN_I : {HISPAN_I} and race(combined) : {race}")

def transform_nhis_data(nhis_df):
    """
    Transform df elements

    :param nhis_df: spark df
    :return: spark df, transformed df
    """

    #add your code here

    # nhis_df = nhis_df.withColumn("SEX", nhis_df.SEX.cast(DoubleType()))
    tranformAgeFromNhisToBrfssUDF = udf(lambda x:tranformAgeFromNhisToBrfss(x),StringType()) # UDF function for age column coversion in NHIS data 
    tranformRaceFromNhisToBrfssUDF = udf(lambda x:tranformRaceFromNhisToBrfss(x),StringType()) # UDF function for conversion of columns in NHIS contain race info to column in Brfss containing race info
   

    transformed_df = nhis_df.select(nhis_df.SEX.cast(DoubleType()), tranformAgeFromNhisToBrfssUDF(col("AGE_P")).cast(DoubleType()).alias("_AGEG5YR"), tranformRaceFromNhisToBrfssUDF(array(col("MRACBPI2"), col("HISPAN_I"))).alias('_IMPRACE').cast(DoubleType()), nhis_df.DIBEV1.cast(DoubleType())) # Transforming structure of nhis to brfss

    return transformed_df


def calculate_statistics(joined_df):
    """
    Calculate prevalence statistics

    :param joined_df: the joined df

    :return: None
    """

    #add your code here
    pass

def join_data(brfss_df, nhis_df):
    """
    Join dataframes

    :param brfss_df: spark df
    :param nhis_df: spark df after transformation
    :return: the joined df

    """
    #add your code here
    joined_df = brfss_df.join(nhis_df, (brfss_df.SEX == nhis_df.SEX) & (), how='inner')
    joined_df = None ##temporary placeholder

    return joined_df

if __name__ == '__main__':
    arg_parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    arg_parser.add_argument('nhis', type=str, default=None, help="brfss filename")
    arg_parser.add_argument('brfss', type=str, default=None, help="nhis filename")
    arg_parser.add_argument('-o', '--output', type=str, default=None, help="output path(optional)")

    #parse args
    args = arg_parser.parse_args()
    if not args.nhis or not args.brfss:
        arg_parser.usage = arg_parser.format_help()
        arg_parser.print_usage()
    else:
        brfss_filename = args.brfss
        nhis_filename = args.nhis
        
        # Start spark session
        spark = SparkSession.builder.getOrCreate()

        # load dataframes
        brfss_df = create_dataframe(brfss_filename, 'json', spark)
        nhis_df = create_dataframe(nhis_filename, 'csv', spark)

        # Perform mapping on nhis dataframe
        nhis_df = transform_nhis_data(nhis_df)
        # Join brfss and nhis df
        joined_df = join_data(brfss_df, nhis_df)
        # Calculate statistics
        calculate_statistics(joined_df)

        # Save
        if args.output:
            joined_df.write.csv(args.output, mode='overwrite', header=True)


        # Stop spark session 
        spark.stop()