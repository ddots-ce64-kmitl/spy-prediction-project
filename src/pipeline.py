import os
import numpy as np
import pandas as pd
import yfinance as yf

def run_full_pipeline(csv_path="data/spy_us_d.csv", output_dir="result"):
    # ---------------------------------------------------------
    # 1. ดาวน์โหลด/โหลดข้อมูล SPY ย้อนหลัง 7 ปี อัตโนมัติ
    # ---------------------------------------------------------
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)

    if not os.path.exists(csv_path):
        print("กำลังดาวน์โหลดข้อมูล SPY ย้อนหลัง 7 ปีจาก yfinance...")
        df_raw = yf.download("SPY", period="7y", interval="1d")
        df_raw = df_raw.reset_index()
        if isinstance(df_raw.columns, pd.MultiIndex):
            df_raw.columns = df_raw.columns.get_level_values(0)
        df_raw.columns = [str(c).lower() for c in df_raw.columns]
        df_raw.to_csv(csv_path, index=False)
        print(f"บันทึกข้อมูลตั้งต้นเรียบร้อยที่ {csv_path}")
    else:
        print(f"พบไฟล์ข้อมูลเดิมที่ {csv_path} กำลังอ่านข้อมูล...")
        df_raw = pd.read_csv(csv_path)
        df_raw.columns = [str(c).lower() for c in df_raw.columns]

    df = df_raw.copy()
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"])
        df = df.sort_values("date").reset_index(drop=True)

    # ---------------------------------------------------------
    # 2. คำนวณ Technical Indicators 20 ตัว (Features X)
    # ---------------------------------------------------------
    # Group 1: Returns & Price Spreads (3 ตัว)
    df["log_return"] = np.log(df["close"] / df["close"].shift(1))
    df["hl_spread"] = df["high"] - df["low"]
    df["co_spread"] = df["close"] - df["open"]

    # Group 2: Moving Averages & Trend (10 ตัว)
    for w in [5, 10, 20, 50, 200]:
        df[f"sma_{w}"] = df["close"].rolling(w).mean()

    df["ema_12"] = df["close"].ewm(span=12, adjust=False).mean()
    df["ema_26"] = df["close"].ewm(span=26, adjust=False).mean()
    df["macd"] = df["ema_12"] - df["ema_26"]
    df["macd_signal"] = df["macd"].ewm(span=9, adjust=False).mean()
    df["macd_hist"] = df["macd"] - df["macd_signal"]

    # Group 3: Volatility & Bollinger Bands (4 ตัว)
    df["volatility_20d"] = df["log_return"].rolling(20).std()
    bollinger_mid = df["sma_20"]
    bollinger_std = df["close"].rolling(20).std()
    df["bollinger_upper"] = bollinger_mid + (2 * bollinger_std)
    df["bollinger_lower"] = bollinger_mid - (2 * bollinger_std)
    df["bollinger_bandwidth"] = (df["bollinger_upper"] - df["bollinger_lower"]) / bollinger_mid

    # Group 4: Momentum (2 ตัว)
    delta = df["close"].diff()
    gain = (delta.where(delta > 0, 0)).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / (loss + 1e-8)
    df["rsi_14"] = 100 - (100 / (1 + rs))
    df["roc_10"] = df["close"].pct_change(10) * 100

    # Group 5: Volume Ratio (1 ตัว)
    df["vol_sma_20"] = df["volume"].rolling(20).mean()
    df["vol_ratio"] = df["volume"] / (df["vol_sma_20"] + 1e-8)

    # Target Variable: Log Return วันถัดไป (t+1)
    df["target_return_next_day"] = df["log_return"].shift(-1)

    # ลบแถวที่มีค่า NaN จากการคำนวณ Window
    df = df.dropna().reset_index(drop=True)

    indicator_cols = [
        "log_return", "hl_spread", "co_spread",
        "sma_5", "sma_10", "sma_20", "sma_50", "sma_200",
        "ema_12", "ema_26", "macd", "macd_signal", "macd_hist",
        "volatility_20d", "bollinger_upper", "bollinger_lower", "bollinger_bandwidth",
        "rsi_14", "roc_10", "vol_ratio"
    ]

    # ---------------------------------------------------------
    # 3. Correlation Matrix ด้วย Linear Algebra
    # ---------------------------------------------------------
    X_raw = df[indicator_cols].values
    
    # Standardize (Z-score Normalization)
    mu = np.mean(X_raw, axis=0)
    sigma = np.std(X_raw, axis=0) + 1e-8
    Z = (X_raw - mu) / sigma

    # Matrix multiplication: R = (1 / (n - 1)) * Z^T * Z
    n = Z.shape[0]
    R = (Z.T @ Z) / (n - 1)
    corr_df = pd.DataFrame(R, index=indicator_cols, columns=indicator_cols)

    # ---------------------------------------------------------
    # 4. Time Series OLS Regression (Normal Equation)
    # ---------------------------------------------------------
    Y = df["target_return_next_day"].values
    X_design = np.hstack([np.ones((Z.shape[0], 1)), Z])

    # Train / Test Split (80/20)
    train_size = int(len(X_design) * 0.8)
    X_train, X_test = X_design[:train_size], X_design[train_size:]
    Y_train, Y_test = Y[:train_size], Y[train_size:]

    # Normal Equation: beta = (X^T * X)^(-1) * X^T * Y
    beta = np.linalg.inv(X_train.T @ X_train) @ X_train.T @ Y_train
    Y_pred = X_test @ beta

    # Calculation of Metrics
    mse = np.mean((Y_test - Y_pred) ** 2)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(Y_test - Y_pred))

    # ---------------------------------------------------------
    # 5. บันทึกผลลัพธ์ลงโฟลเดอร์ result/
    # ---------------------------------------------------------
    pred_path = os.path.join(output_dir, "predictions.csv")
    metrics_path = os.path.join(output_dir, "metrics.txt")

    pred_df = pd.DataFrame({
        "date": df["date"].iloc[train_size:].values if "date" in df.columns else np.arange(len(Y_test)),
        "actual_return": Y_test,
        "predicted_return": Y_pred
    })
    pred_df.to_csv(pred_path, index=False)

    with open(metrics_path, "w", encoding="utf-8") as f:
        f.write("=====================================================\n")
        f.write(" SPY 7-YEAR TIME SERIES PREDICTION (LINEAR ALGEBRA)  \n")
        f.write("=====================================================\n\n")
        f.write(f"Total Daily Records: {len(df)}\n")
        f.write(f"Number of Feature Indicators: {len(indicator_cols)}\n\n")
        f.write("--- Regression Evaluation Metrics ---\n")
        f.write(f"Mean Squared Error (MSE)     : {mse:.8f}\n")
        f.write(f"Root Mean Squared Error (RMSE): {rmse:.8f}\n")
        f.write(f"Mean Absolute Error (MAE)    : {mae:.8f}\n\n")
        f.write("--- Correlation with Next Day Return ---\n")
        corr_with_target = pd.Series(
            (Z.T @ Y) / (n - 1), index=indicator_cols
        ).sort_values(ascending=False)
        f.write(corr_with_target.to_string() + "\n")

    print(f"ประมวลผลสำเร็จ! ผลลัพธ์ถูกบันทึกไว้ในโฟลเดอร์ {output_dir}/")