import pandas as pd

def load_prediction_file(filepath):
    """
    Reads a prediction or ground truth CSV file and returns a standardized DataFrame
    with columns: cell_id, cell_type.
    """
    df = pd.read_csv(filepath)

    # Standardize column names
    if "predicted_cell_type" in df.columns:
        df = df.rename(columns={"predicted_cell_type": "cell_type"})
    elif "true_cell_type" in df.columns:
        df = df.rename(columns={"true_cell_type": "cell_type"})

    # Ensure required columns exist
    if not {"cell_id", "cell_type"}.issubset(df.columns):
        raise ValueError("CSV must contain columns: cell_id, predicted_cell_type OR cell_id, cell_type")

    return df[["cell_id", "cell_type"]]


def compute_accuracy(pred_df, truth_df):
    """
    Merges predictions with ground truth on cell_id and computes accuracy.
    """
    merged = truth_df.merge(pred_df, on="cell_id", suffixes=("_true", "_pred"))

    accuracy = (merged["cell_type_true"] == merged["cell_type_pred"]).mean()

    print(f"Number of cells evaluated: {len(merged)}")
    print(f"Accuracy: {accuracy:.4f}")

    # Warn if predictions are missing
    missing = len(truth_df) - len(merged)
    if missing > 0:
        print(f"Warning: {missing} cell IDs missing in prediction file.")

    return accuracy


if __name__ == "__main__":
    pred_df = load_prediction_file("Ahuja_Bhavisha_653160655.csv")
    #truth_df = load_prediction_file("ground_truth.csv")
    #compute_accuracy(pred_df, truth_df)