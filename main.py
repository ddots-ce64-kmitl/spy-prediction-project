from src.pipeline import run_full_pipeline

if __name__ == "__main__":
    print("เริ่มการทำงานโปรเจกต์วิเคราะห์ SPY ด้วย Linear Algebra...")
    run_full_pipeline(
        csv_path="data/spy_us_d.csv",
        output_dir="result"
    )
    print("เสร็จสิ้นขั้นตอนทั้งหมด! ตรวจสอบไฟล์ใน result/predictions.csv และ result/metrics.txt")