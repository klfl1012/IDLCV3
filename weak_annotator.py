import numpy as np
from scipy.ndimage import label

def image_to_bounding_box(mask_array):
    # 1. Label connected components pixel wise, which object it belongs to
    labeled_array, num_features = label(mask_array)

    # 2. Find bounding box coordinates - For each of the connected component object
    bounding_box_coordinates_tuples = []
    for i in range(1, num_features + 1):
        indices = np.argwhere(labeled_array == i)
        min_y, min_x = indices.min(axis=0) # Min coordinates, we are looking vertically
        max_y, max_x = indices.max(axis=0) # Max coordinates, we are also looking vertically

        bounding_box_coordinates_tuples.append((min_x, min_y, max_x, max_y))

    return bounding_box_coordinates_tuples
