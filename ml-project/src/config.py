from pathlib import Path

INPUT_DIR = Path("/kaggle/input/MABe-mouse-behavior-detection")
TRAIN_TRACKING_DIR = INPUT_DIR / "train_tracking"
TRAIN_ANNOTATION_DIR = INPUT_DIR / "train_annotation"
TEST_TRACKING_DIR = INPUT_DIR / "test_tracking"

WORKING_DIR = Path("/kaggle/working")

INDEX_COLS = [
    "video_id",
    "agent_mouse_id",
    "target_mouse_id",
    "video_frame",
]

SELF_BEHAVIORS = [
    "biteobject", "climb", "dig", "exploreobject", "freeze",
    "genitalgroom", "huddle", "rear", "rest", "run", "selfgroom",
]

PAIR_BEHAVIORS = [
    "allogroom", "approach", "attack", "attemptmount", "avoid",
    "chase", "chaseattack", "defend", "disengage", "dominance",
    "dominancegroom", "dominancemount", "ejaculate", "escape",
    "flinch", "follow", "intromit", "mount", "reciprocalsniff",
    "shepherd", "sniff", "sniffbody", "sniffface", "sniffgenital",
    "submit", "tussle",
]
