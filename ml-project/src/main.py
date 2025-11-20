from data_loader import load_test_dataframe, preprocess_behavior_labels
from feature_extraction import extract_features_for_row
from inference import run_inference_for_group
from postprocess import collapse_predictions, finalize_submission
from config import WORKING_DIR
import polars as pl
from tqdm.auto import tqdm

def main():
    test_df = load_test_dataframe()
    behavior_df = preprocess_behavior_labels(test_df)

    # --- FEATURE EXTRACTION ---
    for row in tqdm(test_df.rows(named=True)):
        extract_features_for_row(row)

    # --- INFERENCE ---
    groups = list(behavior_df.group_by("lab_id", "video_id", "agent", "target", maintain_order=True))

    submissions = []
    for (lab_id, video_id, agent, target), group in tqdm(groups):
        pred_df = run_inference_for_group(lab_id, video_id, agent, target, group)
        seg = collapse_predictions(pred_df)
        if seg is not None:
            submissions.append(seg)

    # --- POSTPROCESS ---
    final = finalize_submission(submissions, test_df)
    final.write_csv(WORKING_DIR / "submission.csv")

if __name__ == "__main__":
    main()
