import numpy as np
import polars as pl
from config import *
from external.robustify import robustify

def collapse_predictions(df):
    cols = df.select(pl.exclude(INDEX_COLS)).columns
    if not cols:
        return None

    df2 = df.with_columns(
        pl.struct(pl.col(cols)).map_elements(
            lambda r: (
                "none" if sum(r.values()) == 0
                else cols[np.argmax(list(r.values()))].split("_")[0]
            )
        ).alias("prediction")
    )

    seg = (
        df2.filter(pl.col("prediction") != pl.col("prediction").shift(1))
        .with_columns(pl.col("video_frame").shift(-1).alias("stop_frame"))
        .filter(pl.col("prediction") != "none")
        .select(
            pl.col("video_id"),
            ("mouse" + pl.col("agent_mouse_id").cast(str)).alias("agent_id"),
            pl.when(pl.col("target_mouse_id") == -1)
                .then("self")
                .otherwise("mouse" + pl.col("target_mouse_id").cast(str))
                .alias("target_id"),
            pl.col("prediction").alias("action"),
            pl.col("video_frame").alias("start_frame"),
            pl.col("stop_frame"),
        )
    )

    return seg


def finalize_submission(group_submissions, test_df):
    sub = pl.concat(group_submissions, how="vertical").sort(
        "video_id", "agent_id", "target_id",
        "action", "start_frame", "stop_frame"
    )
    sub = robustify(sub, test_df, train_test="test")
    return sub.with_row_index("row_id")
