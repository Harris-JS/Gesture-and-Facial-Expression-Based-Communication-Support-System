import os
import numpy as np

# Resolve path from project root regardless of where this script is called from
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
data_path = os.path.join(_PROJECT_ROOT, 'dataset', 'gesture_data.csv')

try:
    data = np.loadtxt(data_path, delimiter=',')
    if data.ndim == 1:
        data = np.expand_dims(data, axis=0)

    def normalize_row(row):
        label = row[0]
        coords = row[1:].reshape(-1, 2).copy()
        
        base_x = coords[0, 0]
        base_y = coords[0, 1]
        
        coords[:, 0] -= base_x
        coords[:, 1] -= base_y
        
        max_val = np.max(np.abs(coords))
        if max_val == 0:
            max_val = 1.0
            
        coords /= max_val
        return np.concatenate(([label], coords.flatten()))

    new_data = np.array([normalize_row(row) for row in data])
    np.savetxt(data_path, new_data, delimiter=',', fmt='%g')
    print("Successfully normalized gesture_data.csv")
except Exception as e:
    print(f"Could not normalize data: {e}")
