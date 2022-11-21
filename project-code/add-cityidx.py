import pandas as pd

# order cities by how expensive they are
raw = pd.read_csv("data/processed/cleaned-final.csv")
grouped = raw.groupby('Local_area').sum().reset_index()
total = grouped.sort_values('Price', ascending=False)

cities = []
for idx, row in total.iterrows():
    if row["Local_area"] not in cities:
        cities.append(row["Local_area"])

# assign cities their indices
city_idx = []
for i in range(len(raw)):
    city_idx.append(0)

def get_city_idx(name):
    i = 0
    for city in cities:
        if name == city:
            return i
        else:
            i += 1
    return i

# assign city indices to each row
raw["city_index"] = city_idx
for idx, row in raw.iterrows():
    raw.at[idx, "city_index"] = get_city_idx(row["Local_area"])

# save data
raw.to_csv("data/processed/final-final.csv")
