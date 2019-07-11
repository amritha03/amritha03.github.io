import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

data = pd.read_csv('headbrain.csv')

x = data["Head Size(cm^3)"].values
y = data.iloc[:,3].values