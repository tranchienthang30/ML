import gc
import polars as pl
from config import *
from external.self_features import make_self_features
from external.pair_features import make_pair_features

def extract_features_for_row(row):
    lab_id = row["lab_id"]
    video_id = row["video_id"]

    tracking_path = TEST_TRACKING_DIR / f"{lab_id}/{video_id}.parquet"
    tracking = pl.read_parquet(tracking_path)

    self_feat = make_self_features(metadata=row, tracking=tracking)
    pair_feat = make_pair_features(metadata=row, tracking=tracking)

    # save
    (WORKING_DIR / "self_features").mkdir(parents=True, exist_ok=True)
    (WORKING_DIR / "pair_features").mkdir(parents=True, exist_ok=True)

    self_feat.write_parquet(WORKING_DIR / "self_features" / f"{video_id}.parquet")
    pair_feat.write_parquet(WORKING_DIR / "pair_features" / f"{video_id}.parquet")

    del tracking, self_feat, pair_feat
    gc.collect()
