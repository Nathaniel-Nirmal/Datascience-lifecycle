import sys
from pyspark.sql import SparkSession
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
    transformed_df = None  # temporary placeholder
    raceMap = {
        (1, 12): 1.0,
        (2, 12): 2.0,
        ((6, 7, 12), 12): 3.0,
        (3, 12): 4.0,
        (-1, tuple([i for i in range(12)])): 5.0,
        (16, 12): 6.0,
    }  # -1 : any value will do.

    print(raceMap)

    return transformed_df


def calculate_statistics(joined_df):
    """
    Calculate prevalence statistics

    :param joined_df: the joined df

    :return: None
    """

    # add your code here
    pass


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
    )

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
        brfss_filename = args.nhis
        nhis_filename = args.brfss

        # Start spark session
        spark = SparkSession.builder.getOrCreate()

        # load dataframes
        brfss_df = create_dataframe(brfss_filename, "json", spark)
        nhis_df = create_dataframe(nhis_filename, "csv", spark)

        # Perform mapping on nhis dataframe
        nhis_df = transform_nhis_data(nhis_df)

        exit(1)
        # Join brfss and nhis df
        joined_df = join_data(brfss_df, nhis_df)
        # Calculate statistics
        calculate_statistics(joined_df)

        # Save
        if args.output:
            joined_df.write.csv(args.output, mode="overwrite", header=True)

        # Stop spark session
        spark.stop()
