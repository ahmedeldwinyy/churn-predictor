"""Create and load the fixed train/test split."""
import pandas as pd
from sklearn.model_selection import train_test_split

from churn_predictor.data import PROCESSED_PATH, ROOT

SEED = 42
TEST_SIZE = 0.2
TRAIN_PATH = ROOT / "data" / "processed" / "train.csv"
TEST_PATH = ROOT / "data" / "processed" / "test.csv"


def make_split(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Stratified split so train and test keep the same churn rate."""
    return train_test_split(
        df, test_size=TEST_SIZE, stratify=df["churn"], random_state=SEED
    )


def load_train() -> pd.DataFrame:
    """Training data only. The test set is deliberately not loaded anywhere yet."""
    return pd.read_csv(TRAIN_PATH, index_col="customerID")


def main() -> None:
    df = pd.read_csv(PROCESSED_PATH, index_col="customerID")
    train, test = make_split(df)
    train.to_csv(TRAIN_PATH)
    test.to_csv(TEST_PATH)
    print(f"train: {train.shape}, churn rate {train['churn'].mean():.4f}")
    print(f"test:  {test.shape}, churn rate {test['churn'].mean():.4f}")


if __name__ == "__main__":
    main()
