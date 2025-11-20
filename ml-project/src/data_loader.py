import polars as pl
import re
from config import INPUT_DIR, SELF_BEHAVIORS, PAIR_BEHAVIORS

def load_test_dataframe():
    df = pl.read_csv(INPUT_DIR / "test.csv")
    return df

def preprocess_behavior_labels(df):
    df = (
        df.filter(pl.col("behaviors_labeled").is_not_null())
        .select(
            pl.col("lab_id"),
            pl.col("video_id"),
            pl.col("behaviors_labeled")
            .map_elements(eval, return_dtype=pl.List(pl.Utf8))
            .alias("behaviors_labeled_list"),
        )
        .explode("behaviors_labeled_list")
        .rename({"behaviors_labeled_list": "raw_behavior"})
        .with_columns([
            pl.col("raw_behavior").str.split(",").list[0].str.replace_all("'", "").alias("agent"),
            pl.col("raw_behavior").str.split(",").list[1].str.replace_all("'", "").alias("target"),
            pl.col("raw_behavior").str.split(",").list[2].str.replace_all("'", "").alias("behavior"),
        ])
        .select("lab_id", "video_id", "agent", "target", "behavior")
    )
    return df

def split_self_and_pair(df):
    return (
        df.filter(pl.col("behavior").is_in(SELF_BEHAVIORS)),
        df.filter(pl.col("behavior").is_in(PAIR_BEHAVIORS))
    )
