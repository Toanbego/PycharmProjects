import numpy as np
from sklearn import preprocessing
input_data = np.array([[5.1, -2.9, 3.3],
                       [-1.2, 7.8, -6.1],
                       [3.9, 0.4, 2.1],
                       [7.3, -9.9, -4.5]])

print("\nBefore:")
print("Mean =", input_data.mean(axis=0))
print("Std deviation =", input_data.std(axis=0))

data_scaled = preprocessing.scale(input_data)
print("\nAfter:")
print("Mean =", data_scaled.mean(axis=0))
print("Std deviation =", data_scaled.std(axis=0))

#min max scaling
dataScaler = preprocessing.MinMaxScaler(feature_range=(0,1))
data_scaled_minMax = dataScaler.fit_transform(input_data)
print("\nMin max scaled data:\n", data_scaled_minMax)