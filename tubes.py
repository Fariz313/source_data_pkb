import numpy as np
import pandas as pd

# Fungsi untuk menghitung variance
def calculate_variance(values):
    mean_value = np.mean(values)
    variance = np.mean((values - mean_value) ** 2)
    return variance

# Fungsi untuk mempartisi data berdasarkan fitur dan threshold
def split_dataset(data, feature, threshold):
    left = data[data[feature] <= threshold]
    right = data[data[feature] > threshold]
    return left, right

# Fungsi untuk menghitung pengurangan variance
def calculate_variance_reduction(data, feature, threshold, target):
    left, right = split_dataset(data, feature, threshold)
    
    if len(left) == 0 or len(right) == 0:
        return 0
    
    total_variance = calculate_variance(data[target])
    left_variance = calculate_variance(left[target])
    right_variance = calculate_variance(right[target])
    
    weighted_variance = (len(left) / len(data)) * left_variance + (len(right) / len(data)) * right_variance
    variance_reduction = total_variance - weighted_variance
    return variance_reduction

# Fungsi untuk menemukan split terbaik
def find_best_split(data, features, target):
    best_feature = None
    best_threshold = None
    best_variance_reduction = -1
    
    for feature in features:
        unique_values = data[feature].unique()
        for threshold in unique_values:
            variance_reduction = calculate_variance_reduction(data, feature, threshold, target)
            if variance_reduction > best_variance_reduction:
                best_variance_reduction = variance_reduction
                best_feature = feature
                best_threshold = threshold
    
    return best_feature, best_threshold

# Kelas untuk Decision Tree Regressor
class DecisionTreeRegressor:
    def __init__(self, max_depth=None):
        self.max_depth = max_depth
        self.tree = None
    
    def fit(self, data, features, target, depth=0):
        if self.max_depth is not None and depth >= self.max_depth:
            return np.mean(data[target])
        
        best_feature, best_threshold = find_best_split(data, features, target)
        
        if best_feature is None:
            return np.mean(data[target])
        
        left, right = split_dataset(data, best_feature, best_threshold)
        if len(left) == 0 or len(right) == 0:
            return np.mean(data[target])
        
        node = {}
        node["feature"] = best_feature
        node["threshold"] = best_threshold
        node["left"] = self.fit(left, features, target, depth + 1)
        node["right"] = self.fit(right, features, target, depth + 1)
        return node
    
    def predict_one(self, x, tree):
        if not isinstance(tree, dict):
            return tree
        
        feature = tree["feature"]
        threshold = tree["threshold"]
        if x[feature] <= threshold:
            return self.predict_one(x, tree["left"])
        else:
            return self.predict_one(x, tree["right"])
    
    def predict(self, data):
        return data.apply(lambda x: self.predict_one(x, self.tree), axis=1)

# Contoh data
# data = [
#     {"CondensedBinary2DGeometry": 10, "BandGapLocation": 4970116577, "BandGapWidth": 226972663},
#     {"CondensedBinary2DGeometry": 20, "BandGapLocation": 4970116578, "BandGapWidth": 326972663},
#     {"CondensedBinary2DGeometry": 30, "BandGapLocation": 4970116579, "BandGapWidth": 426972663},
#     # Tambahkan data lainnya di sini
# ]
data = []
with open('dataset.json', 'r') as f:
    data = json.load(f)


df = pd.DataFrame(data)

# Fitur dan target
features = ["CondensedBinary2DGeometry", "BandGapLocation"]
target = "BandGapWidth"

# Melatih model
model = DecisionTreeRegressor(max_depth=3)
model.tree = model.fit(df, features, target)

# Memprediksi
predictions = model.predict(df)
print(predictions)
