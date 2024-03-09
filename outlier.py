import numpy as np

#assuming that X_tmp is the input dataset and y_tmp is the output

# outlier detection function 
def outlier(x):
    return (np.all(x==X_tmp[58]) or np.all(x==X_tmp[412]))

# Identify non-outlier indices
non_outlier_indices = ~np.array([outlier(x) for x in X_tmp])

outlier_indices = np.array([outlier(x) for x in X_tmp])

outliers = X_tmp[outlier_indices]

# Create X and y arrays without outliers
X = X_tmp[non_outlier_indices]
y = y_tmp[non_outlier_indices]

# Print the result
# print("X:", X)
# print("y:", y)
