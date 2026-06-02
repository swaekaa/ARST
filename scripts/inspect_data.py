import glob

import pandas as pd
import pyarrow.parquet as pq


def main():
    print("--- train.csv ---")
    train_df = pd.read_csv("data/raw/train.csv", nrows=5)
    print(train_df.head())

    print("\n--- train_demographics.csv ---")
    demo_df = pd.read_csv("data/raw/train_demographics.csv", nrows=5)
    print(demo_df.head())

    print("\n--- Sensor Data Structure (Parquet) ---")
    train_parquet_files = glob.glob("data/raw/sensor_data/train/**/*.parquet", recursive=True)
    print(f"Total train sequence parquet files: {len(train_parquet_files)}")

    if train_parquet_files:
        first_file = train_parquet_files[0]
        print(f"\nInspecting schema of: {first_file}")
        schema = pq.read_schema(first_file)
        for _, field in enumerate(schema):
            print(f"  {field.name}: {field.type}")

        print(f"\nFirst few rows of {first_file}:")
        df = pd.read_parquet(first_file).head()
        print(df)


if __name__ == "__main__":
    main()
