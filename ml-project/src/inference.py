import re
import numpy as np
import polars as pl
import xgboost as xgb
from config import *
from pathlib import Path

def load_features(video_id, agent_id, target_id):
    if target_id == "self":
        df = pl.scan_parquet(WORKING_DIR / "self_features" / f"{video_id}.parquet")
        df = df.filter(pl.col("agent_mouse_id") == agent_id)
    else:
        df = pl.scan_parquet(WORKING_DIR / "pair_features" / f"{video_id}.parquet")
        df = df.filter(
            (pl.col("agent_mouse_id") == agent_id) &
            (pl.col("target_mouse_id") == target_id)
        )

    index = df.select(INDEX_COLS).collect()
    feat = df.select(pl.exclude(INDEX_COLS)).collect()
    return index, feat


def run_inference_for_group(lab_id, video_id, agent, target, group):
    agent_mouse_id = int(re.search(r"mouse(\d+)", agent).group(1))
    target_mouse_id = -1 if target == "self" else int(re.search(r"mouse(\d+)", target).group(1))

    index, feat = load_features(video_id, agent_mouse_id, target_mouse_id)
    pred_df = index.clone()

    for row in group.rows(named=True):
        behavior = row["behavior"]

        fold_dirs = list((Path("external/results") / lab_id / behavior).glob("fold_*"))
        if not fold_dirs:
            continue

        predictions = []
        labels = []

        dtest = xgb.DMatrix(feat, feature_names=feat.columns)

        for fold_id, fold_dir in enumerate(fold_dirs):
            threshold = float(open(fold_dir / "threshold.txt").read())
            model = xgb.Booster(model_file=str(fold_dir / "model.json"))

            preds = model.predict(dtest)
            predictions.append(preds)
            labels.append((preds >= threshold).astype(np.int8))

        pred_df = pred_df.with_columns(
            *[pl.Series(f"{behavior}_{i}", predictions[i] * labels[i]).cast(pl.Float32)
              for i in range(len(fold_dirs))]
        )

    return pred_df
