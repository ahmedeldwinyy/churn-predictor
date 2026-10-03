"""Run once: raw CSV -> cleaned CSV in data/processed."""
from churn_predictor.data import PROCESSED_PATH, clean, load_raw


def main() -> None:
    df = clean(load_raw())
    PROCESSED_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(PROCESSED_PATH)
    print(f"Saved {df.shape[0]} rows x {df.shape[1]} columns to {PROCESSED_PATH}")


if __name__ == "__main__":
    main()
