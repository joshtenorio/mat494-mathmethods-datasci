import matplotlib.pyplot as plt
import pandas as pd
from scipy.cluster.vq import vq, kmeans, whiten, kmeans2
import numpy as np

gps = pd.read_csv("data/interim/with-gps.csv")
coords = []
for idx, row in gps.iterrows():
    lat = row["latitude"]
    lon = row["longitude"]
    pair = []
    pair.append(lon) # x axis
    pair.append(lat) # y axis
    coords.append(pair)

coords_np = np.array(coords)

# Whiten data
whitened = whiten(coords_np)

# add whitened data to dataframe
latitude = []
for i in range(len(gps)):
    latitude.append(0)
longitude = []
for i in range(len(gps)):
    longitude.append(0)
groups = []
for i in range(len(gps)):
    groups.append(0)

gps["group"] = groups
gps["w-latitude"] = latitude
gps["w-longitude"] = longitude
for idx, row in gps.iterrows():
    gps.at[idx, "w-latitude"] = whitened[idx][1]
    gps.at[idx, "w-longitude"] = whitened[idx][0]

print(gps.head())
# Find 3 clusters in the data
codebook, distort = kmeans(whitened, 3)


# find out which cluster center is which
north_az = codebook[0]
if codebook[1,1] > north_az[1]:
    north_az = codebook[1]
elif codebook[2,1] > north_az[1]:
    north_az = codebook[2]

south_az = codebook[0]
if codebook[1,1] < south_az[1]:
    south_az = codebook[1]
elif codebook[2,1] < south_az[1]:
    south_az = codebook[2]

centr_az = codebook[0]
if codebook[0,1] > south_az[1] and codebook[0,1] < north_az[1]:
    centr_az = codebook[0]
if codebook[1,1] > south_az[1] and codebook[1,1] < north_az[1]:
    centr_az = codebook[1]
if codebook[2,1] > south_az[1] and codebook[2,1] < north_az[1]:
    centr_az = codebook[2]

# separate coords into groups (NAZ, CAZ, SAZ)
ncoords = []
ccoords = []
scoords = []
for idx, row in gps.iterrows():
    n_dist = (north_az[1] - row["w-latitude"])**2 + (north_az[0] - row["w-longitude"])**2
    c_dist = (centr_az[1] - row["w-latitude"])**2 + (centr_az[0] - row["w-longitude"])**2
    s_dist = (south_az[1] - row["w-latitude"])**2 + (south_az[0] - row["w-longitude"])**2
    
    dists = [n_dist, c_dist, s_dist]
    dists.sort()

    pair = []
    if dists[0] == n_dist:
        pair.append(row["w-longitude"])
        pair.append(row["w-latitude"])
        ncoords.append(pair)
        gps.at[idx, "group"] = 0
    elif dists[0] == c_dist:
        pair.append(row["w-longitude"])
        pair.append(row["w-latitude"])
        ccoords.append(pair)
        gps.at[idx, "group"] = 1
    elif dists[0] == s_dist:
        pair.append(row["w-longitude"])
        pair.append(row["w-latitude"])
        scoords.append(pair)
        gps.at[idx, "group"] = 2

ncoords_np = np.array(ncoords)
ccoords_np = np.array(ccoords)
scoords_np = np.array(scoords)


print(len(ncoords_np))
print(len(ccoords_np))
print(len(scoords_np))

# Plot data and cluster centers in red
#plt.scatter(codebook[:, 0], codebook[:, 1], c = 'r')

#plt.scatter(coords_np[:, 0], coords_np[:, 1])
plt.scatter(whitened[:, 0], whitened[:, 1])
#plt.scatter(ncoords_np[:, 0], ncoords_np[:, 1], c='y')
#plt.scatter(ccoords_np[:, 0], ccoords_np[:, 1], c='r')
#plt.scatter(scoords_np[:, 0], scoords_np[:, 1], c='orange')

plt.scatter(north_az[0], north_az[1], c='y')
plt.scatter(centr_az[0], centr_az[1], c='r')
plt.scatter(south_az[0], south_az[1], c='orange')

plt.xlabel("Longitude")
plt.ylabel("Latitude")
plt.title("GPS locations of houses analyzed")
plt.show()

plt.scatter(ncoords_np[:, 0], ncoords_np[:, 1], c='y')
plt.scatter(ccoords_np[:, 0], ccoords_np[:, 1], c='r')
plt.scatter(scoords_np[:, 0], scoords_np[:, 1], c='orange')

plt.xlabel("Longitude")
plt.ylabel("Latitude")
plt.title("GPS locations of houses analyzed")
plt.show()

gps.to_csv("data/interim/groups-assigned.csv")

