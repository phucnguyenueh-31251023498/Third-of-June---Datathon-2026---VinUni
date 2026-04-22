import pandas as pd
import os

# Liệt kê các cột có ngày
date_map = {
    'customers': ['signup_date'],
    'orders': ['order_date'],
    'promotions': ['start_date', 'end_date'],
    'returns': ['return_date'],
    'reviews': ['review_date'],
    'inventory': ['snapshot_date'],
    'sales': ['Date'],            # Capitalized as per your file
    'shipments': ['ship_date', 'delivery_date'],
    'web_traffic': ['date']
}

def load_all_data(folder_path, schema):
    all_data = {}
    
    # Liệt kê tất cả file csv trong data/
    for filename in os.listdir(folder_path):
        if filename.endswith(".csv"):
            name = filename.replace(".csv", "")
            file_path = os.path.join(folder_path, filename)
            
            # Lấy danh sách các cột ngày trong file, nếu không có trả về [] và đọc file
            parse_cols = schema.get(name, [])
            df = pd.read_csv(file_path, parse_dates=parse_cols)
            
            all_data[name] = df
            
            # In ra shape và null count
            print(f"--- {filename} ---")
            print(f"Shape: {df.shape}")
            print(f"Null Values per Column:\n{df.isnull().sum()}\n")
            
    return all_data

script_dir = os.path.dirname(os.path.abspath(__file__))
data_folder = os.path.join(script_dir, '..', 'data')
    
data_dict = load_all_data(data_folder, date_map)
    