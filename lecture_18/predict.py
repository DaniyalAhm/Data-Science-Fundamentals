from sklearn.linear_model import LinearRegression
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
df = pd.read_csv("data.csv")

LinearRegression().fit(df[["alarm (%H:%M:%S)"]], df["day"]).predict(np.array([[1]]))


#ploting the data
plt.scatter(df["alarm"], df["day"])
plt.plot(df["alarm"], LinearRegression().fit(df[["alarm"]], df["day"]).predict(df[["alarm"]]))
plt.show()
