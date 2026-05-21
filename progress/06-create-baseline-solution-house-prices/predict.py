import pandas as pd
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))
from solution import Model


def predict(input_csv_path, output_csv_path):
    model = Model()
    model.load()

    df = pd.read_csv(input_csv_path)
    ids = df["Id"].values
    predictions = model.predict(input_csv_path)

    result_df = pd.DataFrame({
        "Id": ids,
        "SalePrice": predictions
    })
    result_df.to_csv(output_csv_path, index=False)
    print(f"Predictions saved to {output_csv_path}")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python predict.py <input_csv> <output_csv>")
        sys.exit(1)
    
    input_csv = sys.argv[1]
    output_csv = sys.argv[2]
    predict(input_csv, output_csv)
