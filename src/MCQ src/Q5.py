import pandas as pd

df = pd.read_csv(r'C:\material for learning\Datathon vinuni 2026\Khu data\order_items.csv')

promo_rate = df['promo_id'].notna().mean() * 100

print(promo_rate)