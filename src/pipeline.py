import pandas as pd
import numpy as np

def run_rolling_correlation_pipeline(
    spy_path: str,
    indicator_path: str,
    window_years: int = 7,
    max_lag: int = 14,
    min_samples: int = 100
) -> pd.DataFrame:
    
    # 1. โหลด spy_data.csv (ข้าม Header 3 บรรทัดแรกตามโครงสร้าง yfinance)
    spy_df = pd.read_csv(
        spy_path,
        skiprows=3,
        header=None,
        names=['date', 'close', 'high', 'low', 'open', 'volume']
    )
    
    # 2. โหลดไฟล์ Indicator
    ind_df = pd.read_csv(indicator_path)

    # อ่านชื่อคอลัมน์อัตโนมัติ (คอลัมน์แรกเป็นวันที่, คอลัมน์สองเป็นค่า Indicator)
    indicator_date_col = ind_df.columns[0]  # เช่น 'observation_date'
    indicator_val_col = ind_df.columns[1]   # เช่น 'CPIAUCSL' หรือชื่อ Indicator นั้นๆ

    # 3. แปลง Date เป็น Datetime และตั้งค่าเป็น Index
    spy_df['date'] = pd.to_datetime(spy_df['date'])
    ind_df[indicator_date_col] = pd.to_datetime(ind_df[indicator_date_col])

    spy_df = spy_df.sort_values('date').set_index('date')
    ind_df = ind_df.sort_values(indicator_date_col).set_index(indicator_date_col)

    # 4. Alignment ข้อมูล ยึดวันทำการของ S&P 500 เป็นหลัก
    combined_df = pd.DataFrame(index=spy_df.index)
    combined_df['spy_open'] = pd.to_numeric(spy_df['open'], errors='coerce')
    
    # ffill เติมค่าล่าสุดสำหรับ Indicator ที่ประกาศรายเดือน/รายสัปดาห์
    combined_df['indicator'] = pd.to_numeric(ind_df[indicator_val_col], errors='coerce').reindex(combined_df.index, method='ffill')
    
    # ดักกรณีราคาเปิด SPY เป็นค่าว่าง
    combined_df = combined_df.dropna(subset=['spy_open'])

    # 5. คำนวณช่วงปี 7 ปี
    years = sorted(combined_df.index.year.unique())
    start_year = years[0]
    end_year = years[-1]
    max_start_year = end_year - window_years + 1

    results = []

    # 6. ลูป Rolling Windows & Lead/Lag
    for win_start in range(start_year, max_start_year + 1):
        win_end = win_start + window_years - 1
        window_label = f"{win_start}-{win_end}"

        for lag in range(-max_lag, max_lag + 1):
            # Shift บนชุดข้อมูลเต็มก่อนตัดกรอบ 7 ปี เพื่อดักขอบข้อมูลต้นปี
            shifted_ind = combined_df['indicator'].shift(lag)
            
            temp_df = pd.DataFrame({
                'spy_open': combined_df['spy_open'],
                'indicator_shifted': shifted_ind
            })

            win_data = temp_df.loc[f"{win_start}-01-01":f"{win_end}-12-31"]
            valid_pairs = win_data.dropna(subset=['spy_open', 'indicator_shifted'])

            if len(valid_pairs) >= min_samples:
                corr_val = valid_pairs['spy_open'].corr(valid_pairs['indicator_shifted'])
                
                # ----------------------------------------------------
                # [เพิ่มใหม่] คำนวณช่วงวันที่ของ SPY
                # ----------------------------------------------------
                spy_start_dt = valid_pairs.index.min().strftime('%Y-%m-%d')
                spy_end_dt = valid_pairs.index.max().strftime('%Y-%m-%d')
                spy_date_range = f"{spy_start_dt} to {spy_end_dt}"
                
                # ----------------------------------------------------
                # [เพิ่มใหม่] คำนวณช่วงวันที่ของ Indicator ที่ย้อน/นำหน้าตาม lag
                # ----------------------------------------------------
                valid_positions = combined_df.index.get_indexer(valid_pairs.index)
                ind_positions = valid_positions - lag  # คำนวณตำแหน่งวันที่ของ Indicator ก่อน shift
                
                ind_start_dt = combined_df.index[ind_positions.min()].strftime('%Y-%m-%d')
                ind_end_dt = combined_df.index[ind_positions.max()].strftime('%Y-%m-%d')
                indicator_date_range = f"{ind_start_dt} to {ind_end_dt}"
                
            else:
                corr_val = np.nan
                spy_date_range = np.nan
                indicator_date_range = np.nan

            results.append({
                'window': window_label,
                'start_year': win_start,
                'end_year': win_end,
                'indicator_name': indicator_val_col,
                'lag': lag,
                'valid_days_count': len(valid_pairs),
                'spy_date_range': spy_date_range,              # <-- คอลัมน์ใหม่ 1
                'indicator_date_range': indicator_date_range,  # <-- คอลัมน์ใหม่ 2
                'correlation': corr_val
            })

    return pd.DataFrame(results)