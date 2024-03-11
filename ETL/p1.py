import sys
from pyspark.sql import SparkSession
from pyspark.sql.functions import when
from pyspark.sql import functions as F
import argparse

# feel free to def new functions if you need


def create_dataframe(filepath, format, spark):
    """
    Create a spark df given a filepath and format.

    :param filepath: <str>, the filepath
    :param format: <str>, the file format (e.g. "csv" or "json")
    :param spark: <str> the spark session

    :return: the spark df uploaded
    """

    # add your code here
    if format.lower() == "csv":
        spark_df = spark.read.format("csv").option("header", "true").load(filepath)
    elif format.lower() == "json":
        spark_df = spark.read.format("json").load(filepath)
    else:
        raise ValueError("Unsupported file format: {}".format(format))

    return spark_df


def transform_nhis_data(nhis_df):
    """
    Transform df elements

    :param nhis_df: spark df
    :return: spark df, transformed df
    """

    # add your code here
    transformed_df = nhis_df.withColumn("SEX", nhis_df["SEX"].cast("float"))
    _IMPRACE_COL = (
        when((nhis_df["MRACBPI2"] == 1) & (nhis_df["HISPAN_I"] == 12), 1.0)
        .when((nhis_df["MRACBPI2"] == 2) & (nhis_df["HISPAN_I"] == 12), 2.0)
        .when((nhis_df["MRACBPI2"].isin([6, 7, 12])) & (nhis_df["HISPAN_I"] == 12), 3.0)
        .when((nhis_df["MRACBPI2"] == 3) & (nhis_df["HISPAN_I"] == 12), 4.0)
        .when(
            (nhis_df["MRACBPI2"].isin([1, 2, 3, 6, 7, 12, 16, 17]))
            & (nhis_df["HISPAN_I"] != 12),
            5.0,
        )
        .when((nhis_df["MRACBPI2"] == 16) & (nhis_df["HISPAN_I"] == 12), 6.0)
        .otherwise(6.0)
    )

    _AGEG5YR = (
        when((nhis_df["AGE_P"] >= 18) & (nhis_df["AGE_P"] <= 24), 1.0)
        .when((nhis_df["AGE_P"] >= 25) & (nhis_df["AGE_P"] <= 29), 2.0)
        .when((nhis_df["AGE_P"] >= 30) & (nhis_df["AGE_P"] <= 34), 3.0)
        .when((nhis_df["AGE_P"] >= 35) & (nhis_df["AGE_P"] <= 39), 4.0)
        .when((nhis_df["AGE_P"] >= 40) & (nhis_df["AGE_P"] <= 44), 5.0)
        .when((nhis_df["AGE_P"] >= 45) & (nhis_df["AGE_P"] <= 49), 6.0)
        .when((nhis_df["AGE_P"] >= 50) & (nhis_df["AGE_P"] <= 54), 7.0)
        .when((nhis_df["AGE_P"] >= 55) & (nhis_df["AGE_P"] <= 59), 8.0)
        .when((nhis_df["AGE_P"] >= 60) & (nhis_df["AGE_P"] <= 64), 9.0)
        .when((nhis_df["AGE_P"] >= 65) & (nhis_df["AGE_P"] <= 69), 10.0)
        .when((nhis_df["AGE_P"] >= 70) & (nhis_df["AGE_P"] <= 74), 11.0)
        .when((nhis_df["AGE_P"] >= 75) & (nhis_df["AGE_P"] <= 79), 12.0)
        .when((nhis_df["AGE_P"] >= 80) & (nhis_df["AGE_P"] <= 99), 13.0)
        .otherwise(14.0)
    )

    transformed_df = transformed_df.withColumn("_IMPRACE", _IMPRACE_COL)
    transformed_df = transformed_df.withColumn("_AGEG5YR", _AGEG5YR)
    columns_to_drop = ["MRACBPI2", "HISPAN_I", "AGE_P"]
    transformed_df = transformed_df.drop(*columns_to_drop)

    return transformed_df


def calculate_statistics(joined_df):
    """
    Calculate prevalence statistics

    :param joined_df: the joined df

    :return: None
    """

    # add your code here
    for column in ["_IMPRACE", "SEX", "_AGEG5YR"]:
        statistics = joined_df.groupBy(column).agg(
            (F.sum(F.when(F.col("DIBEV1") == 1, 1).otherwise(0)) / F.count("*") * 100).alias("prevalence")
        )

        # Show the calculated prevalence statistics for the current column
        print(f"Prevalence statistics for column '{column}':")
        statistics.show()

def join_data(brfss_df, nhis_df):
    """
    Join dataframes

    :param brfss_df: spark df
    :param nhis_df: spark df after transformation
    :return: the joined df

    """
    # add your code here
    joined_df = brfss_df.join(
        nhis_df,
        (brfss_df.SEX == nhis_df.SEX)
        & (brfss_df._AGEG5YR == nhis_df._AGEG5YR)
        & (brfss_df._IMPRACE == nhis_df._IMPRACE),
        how="inner",
    ).select(*(brfss_df[col] for col in ["_LLCPWT"]), *(nhis_df[col] for col in nhis_df.columns))
    #joined_df = joined_df.drop("SEX", "_AGEG5YR", "_IMPRACE")
    print(joined_df.count())
    return joined_df


if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    arg_parser.add_argument("nhis", type=str, default=None, help="brfss filename")
    arg_parser.add_argument("brfss", type=str, default=None, help="nhis filename")
    arg_parser.add_argument(
        "-o", "--output", type=str, default=None, help="output path(optional)"
    )

    # parse args
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
        brfss_df = create_dataframe(brfss_filename, "json", spark)
        nhis_df = create_dataframe(nhis_filename, "csv", spark)

        # Perform mapping on nhis dataframe
        nhis_df = transform_nhis_data(nhis_df)

        # Join brfss and nhis df
        joined_df = join_data(brfss_df, nhis_df)
        # Calculate statistics
        #print(joined_df.show())
        calculate_statistics(joined_df)

        # Save
        if args.output:
            joined_df.write.csv(args.output, mode="overwrite", header=True)

        # Stop spark session
        spark.stop()
