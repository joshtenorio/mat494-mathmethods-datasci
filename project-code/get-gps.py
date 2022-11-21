import matplotlib.pyplot as plt
import pandas as pd
import pgeocode

housing_raw = pd.read_csv("data/raw/AZhousingData.csv")
nomi = pgeocode.Nominatim('us')

#print(nomi.query_postal_code("85286"))

###################################
# add gps data to housing columns #
###################################
latitude = []
for i in range(len(housing_raw)):
    latitude.append(0)
longitude = []
for i in range(len(housing_raw)):
    longitude.append(0)

housing_raw["latitude"] = latitude
housing_raw["longitude"] = longitude
to_remove = []
for idx, row in housing_raw.iterrows():
    zip = row["zipcode"]
    if(zip.isdigit()):
        query = nomi.query_postal_code(zip)
        housing_raw.at[idx, "latitude"] = query["latitude"]
        housing_raw.at[idx, "longitude"] = query["longitude"]
    else:
        to_remove.append(idx)

housing_raw.drop(to_remove)

housing_raw.to_csv("data/interim/with-gps.csv")