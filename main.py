import os
from src.pipeline import run_rolling_correlation_pipeline

if __name__ == "__main__":
    SPY_DATA_PATH = "data/spy_data.csv"
    INDICATOR_DATA_PATH = "data/CPI.csv"
    OUTPUT_RESULT_PATH = "result/y.csv"

    os.makedirs("result", exist_ok=True)
    
    results_df = run_rolling_correlation_pipeline(
        spy_path=SPY_DATA_PATH,
        indicator_path=INDICATOR_DATA_PATH,
        window_years=7,
        max_lag=14
    )

    results_df.to_csv(OUTPUT_RESULT_PATH, index=False)