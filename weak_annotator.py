import numpy as np
from scipy.ndimage import label
from collections import deque

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

# Logic is somewhat similar to Leetcode #200
def region_growing(image, seeds, threshold=0.05):
    # 1. Initialization - Create mask & visited, initialized as zeros / false of the same shape as the image
    mask = np.full_like(image, -1,  dtype=np.int8)
    visited = np.zeros_like(image, dtype=bool)
    neighbors = [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)]
    
    seed_intensities = [image[y, x] for x, y in seeds]
    median_intensity = np.median(seed_intensities)
    foreground_seeds = [(x, y) for (x, y), intensity in zip(seeds, seed_intensities) if intensity <= median_intensity]
    background_seeds = [(x, y) for (x, y), intensity in zip(seeds, seed_intensities) if intensity > median_intensity]
    
    queue = deque([(x, y, 1) for x, y in foreground_seeds] + [(x, y, 0) for x, y in background_seeds])
    for x, y, clas in queue:
        mask[y, x] = clas
        visited[y, x] = True

    # 2. Perform region growing - Iterative BFS 
    while queue:
        x, y, clas = queue.popleft()
        for dx, dy in neighbors:
            # Process neighbor coords that are within bounds
            nx, ny = x + dx, y + dy
            if 0 <= nx < image.shape[1] and 0 <= ny < image.shape[0] and not visited[ny, nx]: 
                if (abs(image[ny, nx]) - image[y, x]) <= threshold:
                    mask[ny, nx] = clas
                    visited[ny, nx] = True
                    queue.append((nx, ny, clas))
    
    # 3. Post-process the mask to set unvisited pixels to background (0)
    mask[mask == -1] = 0

    print("Mask unique values:", np.unique(mask))
    print("Foreground pixel count:", np.sum(mask == 1))
    print("Background pixel count:", np.sum(mask == 0))
    return mask