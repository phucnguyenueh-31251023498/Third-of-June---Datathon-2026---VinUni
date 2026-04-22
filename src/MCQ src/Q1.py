import pandas as pd 
# Đọc dữ liệu
orders = pd.read_csv(r"C:\Users\Admin\Documents\orders.csv")

# Convert sang datetime
orders["order_date"] = pd.to_datetime(orders["order_date"])

# Sắp xếp
orders = orders.sort_values(["customer_id", "order_date"])

# Tính khoảng cách giữa các lần mua
orders["prev_date"] = orders.groupby("customer_id")["order_date"].shift(1)
orders["days_between"] = (orders["order_date"] - orders["prev_date"]).dt.days

# Lọc khách mua > 1 lần (tức là có days_between)
df = orders.dropna(subset=["days_between"])

# Tính median
median_days = df["days_between"].median()

print("Median days between purchases:", median_days)