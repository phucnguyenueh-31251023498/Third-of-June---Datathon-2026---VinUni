import pandas as pd 
# Đọc dữ liệu
payments = pd.read_csv(r"C:\Users\Admin\Documents\payments.csv")

# Groupby
payment_value_foreachinstallments = payments.groupby("installments")["payment_value"].mean()

# Tinh
print(payment_value_foreachinstallments.idxmax(), payment_value_foreachinstallments.max())