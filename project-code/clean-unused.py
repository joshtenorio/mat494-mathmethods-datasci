import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


raw = pd.read_csv("data/interim/groups-assigned-final.csv")

raw.drop(columns=["idx2","a"], inplace=True)

print(raw.describe())

raw.to_csv("data/processed/cleaned-final.csv")
