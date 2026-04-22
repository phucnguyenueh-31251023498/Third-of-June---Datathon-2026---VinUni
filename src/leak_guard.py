import pandas as pd
import os

def prepare_data(data_folder):
    # Load sales.csv và tạo file csv để train model bằng data trước 1/1/2023
    df = pd.read_csv(os.path.join(data_folder, 'sales.csv'), parse_dates=['Date'])
    train_df = df[df['Date'] < '2023-01-01'].copy()

    # Tạo 1 file để chạy thử và kiểm tra tính chính xác của model (validation)
    val_df = df[(df['Date'] >= '2022-01-01') & (df['Date'] < '2023-01-01')].copy()
    
    # Bỏ đi những cột có khả năng làm hỏng dự đoán của model
    drop_cols = [
        'return_date', 'return_reason', 'refund_amount', 
        'ship_date', 'delivery_date', 'order_status', 
        'Revenue', 'COGS' 
    ]
    features = [c for c in train_df.columns if c not in drop_cols]
    # Trả về features and kết quả của file train model và file kiểm tra model (validation)
    return train_df[features], train_df[['Revenue', 'COGS']], val_df[features], val_df[['Revenue', 'COGS']]

