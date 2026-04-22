import pandas as pd
# Đọc dữ liệu
customers = pd.read_csv(r"C:\Users\Admin\Documents\customers.csv")
orders = pd.read_csv(r"C:\Users\Admin\Documents\orders.csv")

# Drop
val_cus = customers.dropna(subset=['age_group'])

# Join
df = orders.merge(val_cus, on="customer_id")

# Groupby
avg_order_per_ag = df.groupby("age_group")["order_id"].nunique()
avg_customer_per_ag = df.groupby("age_group")["customer_id"].nunique()
avg_order_per_cus = avg_order_per_ag / avg_customer_per_ag
# Tinh
print(avg_order_per_cus.idxmax(), avg_order_per_cus.max())