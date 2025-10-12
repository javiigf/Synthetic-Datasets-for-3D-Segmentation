from __future__ import division
import time
import numpy as np
from scipy.spatial.distance import cdist
from scipy.spatial import Voronoi, Delaunay, ConvexHull, SphericalVoronoi, geometric_slerp, cKDTree
import tifffile
from numba import njit, prange # Import prange
import math
from skimage.morphology import binary_dilation, ball, binary_erosion
import elasticdeform
import matplotlib.pyplot as plt
import cv2 
from PIL import Image, ImageDraw
import random
import SimpleITK as sitk
from skimage.segmentation import find_boundaries
from scipy.ndimage import center_of_mass

start = time.time()

# (Keep all your existing helper functions here, as they were in your provided file)

# Define functions
def ellipsoid_surface_area(a, b, c):
    p = 1.6075
    term1 = (a * b)**p
    term2 = (a * c)**p
    term3 = (b * c)**p
    mean = (term1 + term2 + term3) / 3
    surface_area = 4 * np.pi * (mean**(1/p))
    return surface_area


@njit
def evaluate_ellipsoid(point, center, radii):
    """Evaluates the ellipsoid equation for a given point.
    Returns (x/a)^2 + (y/b)^2 + (z/c)^2.
    Value < 1 means inside, = 1 means on surface, > 1 means outside.
    """
    adjusted_point = point - center
    return np.sum((adjusted_point / radii)**2)

@njit
def is_inside_ellipsoid(point, center, radii):
    """Checks if a point is strictly inside an ellipsoid."""
    if point.ndim == 1:
        return evaluate_ellipsoid(point, center, radii) < 1.0
    else: # Handle array of points
        results = np.empty(point.shape[0], dtype=np.bool_)
        for i in range(point.shape[0]):
            results[i] = evaluate_ellipsoid(point[i], center, radii) < 1.0
        return results

@njit
def is_outside_ellipsoid(point, center, radii):
    """Checks if a point is strictly outside an ellipsoid."""
    if point.ndim == 1:
        return evaluate_ellipsoid(point, center, radii) > 1.0
    else: # Handle array of points
        results = np.empty(point.shape[0], dtype=np.bool_)
        for i in range(point.shape[0]):
            results[i] = evaluate_ellipsoid(point[i], center, radii) > 1.0
        return results

@njit
def project_point_to_ellipsoid_surface(point, center, radii):
    """Projects a point to the surface of an ellipsoid along the ray from the center."""
    # Special case: if point is at center, cannot project meaningfully.
    if np.all(point == center):
        # Handle this case by returning a point on the surface, e.g., (radii[0], 0, 0)
        return center + np.array([radii[0], 0, 0])
        
    direction = point - center
    
    # Calculate the scale factor 'k' such that k*direction is on the ellipsoid surface
    # (k*dx/a)^2 + (k*dy/b)^2 + (k*dz/c)^2 = 1
    # k^2 * [ (dx/a)^2 + (dy/b)^2 + (dz/c)^2 ] = 1
    # k = 1 / sqrt( (dx/a)^2 + (dy/b)^2 + (dz/c)^2 )
    
    term_sum_sq = np.sum((direction / radii)**2)
    
    if term_sum_sq == 0: # Should not happen if point is not center
        return center # Fallback
        
    k = 1.0 / math.sqrt(term_sum_sq)
    
    projected_point = center + k * direction
    return projected_point


@njit
def compute_min_distance(new_point, seeds):
    min_dist = np.inf
    for seed in seeds:
        dist = np.linalg.norm(new_point - seed)
        if dist < min_dist:
            min_dist = dist
    return min_dist

@njit
def createSeeds(nSeeds, points, minimumSeparation):
    seeds = []
    pstarts = []
    randomSort = np.random.permutation(points.shape[0])
    for i in range(points.shape[0]):
        randomDotIx = randomSort[i]
        randomDot = points[randomDotIx]
    
        if len(seeds) == 0:
            seeds.append(randomDot)
            pstarts.append(randomDotIx)
            continue
        
        min_distance = compute_min_distance(randomDot, seeds)
        
        if min_distance > minimumSeparation:
            seeds.append(randomDot)
            pstarts.append(randomDotIx)
    
        if len(seeds) == nSeeds:
            break
    if len(seeds) < nSeeds:
        raise ValueError('Error occurred with seed localization.')
    return np.array(pstarts)

def get_augmented_centroids(augmented_grid, initial_centroids):
    # Generate augmented ellipsoid grid point
    
    # Find nearest points to initial centroids
    tree = cKDTree(augmented_grid)
    final_centroids_augmented = np.array([augmented_grid[tree.query(x, k=1)[1]] for x in initial_centroids])
    
    return final_centroids_augmented


@njit
def generate_points_within_triangle(triangle, num_points):
    u = np.random.rand(num_points)
    v = np.random.rand(num_points)
    mask = u + v <= 1
    u, v = u[mask], v[mask]
    w = 1 - (u + v)
    points = np.empty((len(u), 3))
    for i in range(len(u)):
        points[i] = u[i] * triangle[0] + v[i] * triangle[1] + w[i] * triangle[2]
    return points

def generate_ellipsoid_points(center, radius, resolution):
    u = np.linspace(0, 2 * np.pi, resolution)
    v = np.linspace(0, np.pi, resolution)
    X = center[0] + radius[0] * np.outer(np.cos(u), np.sin(v))
    Y = center[1] + radius[1] * np.outer(np.sin(u), np.sin(v))
    Z = center[2] + radius[2] * np.outer(np.ones(np.size(u)), np.cos(v))
    coordinates = np.column_stack([X.flatten(), Y.flatten(), Z.flatten()])
    coordinates = np.unique(coordinates, axis=0)
    return coordinates

def generate_grid_points(center, side_lengths, num_points):
    sideX, sideY = side_lengths
    cx, cy = center
    
    # Calculate number of points along each axis
    num_points_x = int(np.sqrt(num_points * (sideX / sideY)))
    num_points_y = int(np.sqrt(num_points * (sideY / sideX)))
    
    # Adjust the number of points to make sure we have at least `num_points`
    while num_points_x * num_points_y < num_points:
        if (num_points_x + 1) * num_points_y <= num_points:
            num_points_x += 1
        else:
            num_points_y += 1
    
    # Create the grid (ensure all coordinates are positive)
    x = np.linspace(0, sideX, num_points_x)
    y = np.linspace(0, sideY, num_points_y)
    
    # Generate points
    xv, yv = np.meshgrid(x, y)
    coordinates = np.column_stack([xv.flatten(), yv.flatten()])
    
    # Shift grid to align with the center
    coordinates[:, 0] += (cx - sideX / 2)
    coordinates[:, 1] += (cy - sideY / 2)
    
    # If more points are generated, select only the required number of points
    if len(coordinates) > num_points:
        coordinates = coordinates[:num_points]
    
    return coordinates

def normalize_points(points, shape):
    min_bounds = np.min(points, axis=0)
    max_bounds = np.max(points, axis=0)
    normalized_points = (points - min_bounds) / (max_bounds - min_bounds) * (np.array(shape) - 1)
    return np.round(normalized_points).astype(int)

def sample_points_on_edges(lines, num_samples):
    sampled_points = []
    for line in lines:
        start, end = line
        t_values = np.linspace(0, 1, num_samples)
        for t in t_values:
            sampled_points.append(start * (1 - t) + end * t)
    return np.array(sampled_points)

def create_hollow_spheroid(center, axis_lengths, cellHeight, matrix_shape):
    x = np.arange(matrix_shape[0]) - center[0]
    y = np.arange(matrix_shape[1]) - center[1]
    z = np.arange(matrix_shape[2]) - center[2]
    xx, yy, zz = np.meshgrid(x, y, z, indexing='ij')

    # Calculate normalized distances from the center
    distances = ((xx / axis_lengths[0]) ** 2 +
                 (yy / axis_lengths[1]) ** 2 +
                 (zz / axis_lengths[2]) ** 2)

    # Define the normalized distances for boundaries
    # Whole spheroid
    whole_spheroid_inner_radius = 1.0  # Equivalent to axis_lengths in normalized coordinates
    whole_spheroid_outer_radius = 1.0 + (cellHeight / max(axis_lengths))  # Normalized distance

    # Inner spheroid
    inner_spheroid_inner_radius = 1.0  # Equivalent to axis_lengths in normalized coordinates
    inner_spheroid_outer_radius = 1.0 - 0.05  # Normalized distance

    # Outer spheroid
    outer_spheroid_inner_radius = 1.0 + (cellHeight / max(axis_lengths))  # Normalized distance
    outer_spheroid_outer_radius = 1.0 + (cellHeight / max(axis_lengths)) + 0.05  # Normalized distance

    # Create binary masks
    spheroid = ((distances >= whole_spheroid_inner_radius) & 
                (distances <= whole_spheroid_outer_radius)).astype(int)
    
    innerSpheroid = ((distances >= inner_spheroid_outer_radius) & 
                     (distances <= inner_spheroid_inner_radius)).astype(int)

    outerSpheroid = ((distances >= outer_spheroid_inner_radius) & 
                     (distances <= outer_spheroid_outer_radius)).astype(int)



    return spheroid, innerSpheroid, outerSpheroid

def create_rectangle(center, axis_lengths, matrix_shape):
    # Create an empty 2D array of the given shape
    rectangle = np.zeros(matrix_shape, dtype=int)
    outerRectangle = np.zeros(matrix_shape, dtype=int)
    
    # Calculate the start and end indices for rows and columns
    start_row = int(center[0] - axis_lengths[0] / 2)
    end_row = int(center[0] + axis_lengths[0] / 2)
    start_col = int(center[1] - axis_lengths[1] / 2)
    end_col = int(center[1] + axis_lengths[1] / 2)

    # Ensure the indices are within the matrix boundaries
    start_row = max(start_row, 0)
    end_row = min(end_row, matrix_shape[0])
    start_col = max(start_col, 0)
    end_col = min(end_col, matrix_shape[1])

    # Fill the rectangle with ones
    rectangle[start_row:end_row, start_col:end_col] = 1
    
    outerRectangle[start_row, start_col:end_col] = 1
    outerRectangle[end_row - 1, start_col:end_col] = 1
    # Left and right edges
    outerRectangle[start_row:end_row, start_col] = 1
    outerRectangle[start_row:end_row, end_col - 1] = 1
    
    return rectangle, outerRectangle

def process_voronoi_cells(vor, samplesPerFace): # Changed arguments to take 'vor' object
    all_surface_points_list = []

    # Iterate over each generator point's associated Voronoi region
    # vor.point_region maps generator indices to region indices.
    # The length of vor.point_region corresponds to the number of generator points.
    for i in range(len(vor.point_region)):
        region_idx = vor.point_region[i]
        
        # Skip the region for the infinite vertex or invalid regions
        if region_idx == -1:
            continue
        
        # Get the vertex indices for this region
        vertex_indices = vor.regions[region_idx] # Access regions from vor object
        
        # Ensure the region is valid and contains actual vertices (not -1 for unbounded)
        # For 3D ConvexHull, typically at least 4 non-coplanar points are needed to form a 3D volume.
        # If the vertices are coplanar, ConvexHull might still work but produce a 2D hull.
        # We explicitly skip if there are too few vertices for a 3D hull.
        if -1 in vertex_indices or len(vertex_indices) < 4:
            continue
        
        vertices_of_cell = vor.vertices[vertex_indices] # Access vertices from vor object

        try:
            # Compute the Convex Hull of the cell's vertices. This will give the facets (triangles) on its surface.
            hull = ConvexHull(vertices_of_cell)
            
            # hull.simplices gives the indices of the vertices that form the triangles of the hull's surface.
            # We then use these indices to get the actual 3D points forming each triangle.
            triangles = vertices_of_cell[hull.simplices]
            
            for triangle in triangles:
                # Ensure the 'triangle' extracted from hull.simplices indeed has 3 vertices.
                # This should generally be true for `hull.simplices`.
                if triangle.shape[0] == 3:
                    surface_points = generate_points_within_triangle(triangle, samplesPerFace)
                    if surface_points.size > 0: # Check if points were actually generated
                        all_surface_points_list.append(surface_points)
        except Exception as e:
            # Catch exceptions from ConvexHull (e.g., if points are collinear/coplanar and don't form a 3D hull)
            # print(f"Warning: ConvexHull failed for cell {i} (region {region_idx}) with {len(vertices_of_cell)} vertices. Error: {e}")
            continue # Skip this problematic cell

    if not all_surface_points_list:
        # If no surface points were generated from any cell, return an empty array of the correct shape.
        # This prevents the ValueError from np.concatenate.
        return np.empty((0, 3), dtype=np.float64)
    else:
        return np.concatenate(all_surface_points_list, axis=0)

@njit
def process_voronoi_cells_2d(vor_vertices, vor_regions, vor_point_region, samplesPerEdge):
    all_edge_points_list = []

    for cell_index in range(len(vor_regions) - 1):
        vertex_indices = vor_regions[vor_point_region[cell_index]]
        if -1 not in vertex_indices and len(vertex_indices) > 0:
            vertices_of_cell = vor_vertices[vertex_indices]

            num_vertices = len(vertices_of_cell)
            for i in range(num_vertices):
                p1 = vertices_of_cell[i]
                p2 = vertices_of_cell[(i + 1) % num_vertices]  # Wrap around to the first vertex
                edge_points = np.linspace(p1, p2, samplesPerEdge, endpoint=False)
                all_edge_points_list.append(edge_points)

    return np.concatenate(all_edge_points_list, axis=0)

def lloyd_relaxation(points, iterations, width, height):
    for _ in range(iterations):
        vor = Voronoi(points)
        new_points = []
        for region in vor.regions:
            if not -1 in region and len(region) > 0:
                polygon = [tuple(vor.vertices[i]) for i in region]
                new_points.append(np.mean(polygon, axis=0))
        points = np.clip(new_points, [0, 0], [width, height])
    return points, vor

def lloyd_relaxation3D(points, iterations, width, height, depth, num_samples=100000):
    for _ in range(iterations):
        # Step 1: Randomly sample many points in the 3D volume
        samples = np.random.rand(num_samples, 3) * [width, height, depth]
        
        # Step 2: Assign each sample to the nearest generator point
        tree = cKDTree(points)
        _, indices = tree.query(samples)

        # Step 3: Compute the centroid of the samples in each Voronoi cell
        new_points = np.zeros_like(points)
        counts = np.zeros(len(points))
        
        for i, idx in enumerate(indices):
            new_points[idx] += samples[i]
            counts[idx] += 1

        # Avoid division by zero
        mask = counts > 0
        new_points[mask] /= counts[mask][:, None]
        new_points[~mask] = np.random.rand(np.sum(~mask), 3) * [width, height, depth]

        points = new_points

    # Final Voronoi diagram
    vor = Voronoi(points)
    return points, vor

def check_points_in_ellipsoidal_shell(points, center, radii_inner, radii_outer, verbose=True):
    shifted = points - center
    norm2_inner = np.sum((shifted / radii_inner)**2, axis=1)
    norm2_outer = np.sum((shifted / radii_outer)**2, axis=1)

    inside_shell = (norm2_inner >= 1.0) & (norm2_outer <= 1.0)
    outside_mask = ~inside_shell

    n_outside = np.count_nonzero(outside_mask)
    if verbose:
        print(f"→ Points outside ellipsoidal shell: {n_outside}/{len(points)}")
        if n_outside > 0:
            print("   Max inner norm²:", norm2_inner.min())
            print("   Max outer norm²:", norm2_outer.max())

    return inside_shell, outside_mask

def lloyd_relaxation3D_constrained(points, iterations, width, height, depth,
                                   ellipsoid_center, ellipsoid_radii_inner, ellipsoid_radii_outer,
                                   num_samples=100000):
    for _ in range(iterations):
        samples = np.random.rand(num_samples, 3) * [width, height, depth]
        samples = np.asarray(samples)
        if samples.ndim != 2 or samples.shape[1] != 3:
            samples = samples.T  # Transpose if accidentally shaped (3, N)
        
        # Unpack the returned tuple here
        inside_mask, _ = check_points_in_ellipsoidal_shell(samples, ellipsoid_center, ellipsoid_radii_inner, ellipsoid_radii_outer)
        samples = samples[inside_mask]

        tree = cKDTree(points)
        _, indices = tree.query(samples)

        new_points = np.zeros_like(points)
        counts = np.zeros(len(points))
        
        for i, idx in enumerate(indices):
            new_points[idx] += samples[i]
            counts[idx] += 1

        # Avoid division by zero
        mask = counts > 0
        new_points[mask] /= counts[mask][:, None]
        new_points[~mask] = np.random.rand(np.sum(~mask), 3) * [width, height, depth]

        # Again unpack the returned tuple here
        inside_mask, _ = check_points_in_ellipsoidal_shell(
            new_points, ellipsoid_center, ellipsoid_radii_inner, ellipsoid_radii_outer, verbose=False)
        new_points = new_points[inside_mask]

        points = new_points

    vor = Voronoi(points)
    return points, vor

def draw_voronoi_with_oscillating_membranes(vor, width, height, min_outlineW, max_outlineW, crop_width, crop_height): #pedro @todo jesus copiao
    img = Image.new('RGB', (round(width), round(height)), 'white')
    draw = ImageDraw.Draw(img)

    for region in vor.regions:
        if not -1 in region and len(region) > 0:
            polygon = [tuple(vor.vertices[i]) for i in region]
            draw.polygon(polygon, outline="black")

    for ridge_vertices in vor.ridge_vertices:
        if -1 not in ridge_vertices:
            v0, v1 = vor.vertices[ridge_vertices]
            outline_width = random.randint(min_outlineW, max_outlineW)
            draw.line([tuple(v0), tuple(v1)], fill="black", width=outline_width)

    # Crop to the central region
    left = (width - crop_width) // 2
    top = (height - crop_height) // 2
    right = left + crop_width
    bottom = top + crop_height
    img = img.crop((left, top, right, bottom))

    return img

@njit
def filter_surfaces(allSurfaces, matrix_shape):
    mask = (allSurfaces[:, 0] >= 0) & (allSurfaces[:, 0] < matrix_shape[0]) & \
           (allSurfaces[:, 1] >= 0) & (allSurfaces[:, 1] < matrix_shape[1]) & \
           (allSurfaces[:, 2] >= 0) & (allSurfaces[:, 2] < matrix_shape[2])
    return allSurfaces[mask]

@njit
def filter_surfaces2D(allSurfaces, matrix_shape):
    mask = (allSurfaces[:, 0] >= 0) & (allSurfaces[:, 0] < matrix_shape[0]) & \
           (allSurfaces[:, 1] >= 0) & (allSurfaces[:, 1] < matrix_shape[1])
    return allSurfaces[mask]

def closest_distance_on_ellipsoid(n, a, b, c):
    """
    Calculate the approximate Euclidean distance between the closest points 
    on the surface of an ellipsoid with semi-major axes a, b, c and n points.
    
    Parameters:
    n (int): Number of points on the surface of the ellipsoid.
    a (float): Semi-major axis along the x-axis.
    b (float): Semi-major axis along the y-axis.
    c (float): Semi-major axis along the z-axis.
    
    Returns:
    float: Approximate distance between the closest points.
    """
    
    # Approximate the surface area of the ellipsoid using Ramanujan's formula
    p = 1.6075 # This needs to be defined within the function if not a global constant
    term1 = (a * b) ** p
    term2 = (a * c) ** p
    term3 = (b * c) ** p
    mean = (term1 + term2 + term3) / 3 # Re-added definition of mean
    S = 4 * np.pi * (mean**(1/p))
    
    # Calculate the average area per point
    A = S / n
    
    # Estimate the distance between the closest points
    d = math.sqrt(A)
    
    return d

def generate_random_points_on_sphere(num_points, radius):
    phi = np.random.uniform(0, np.pi, num_points)  # azimuthal angle
    theta = np.random.uniform(0, 2 * np.pi, num_points)  # polar angle

    x = radius * np.sin(phi) * np.cos(theta)
    y = radius * np.sin(phi) * np.sin(theta)
    z = radius * np.cos(phi)

    points = np.vstack((x, y, z)).T
    return points

def morph_label_outlines(labels):
    # `connectivity=1` means face-connected neighbors (3D 6-neighborhood)
    outlines = find_boundaries(labels, connectivity=1, mode='outer')
    return (outlines.astype(np.uint8)) * 255

def generate_mask_outlines(img, selem_erode, selem_dilate):
    # Find unique cell values excluding zero
    uniq_cells = np.unique(img)
    uniq_cells = uniq_cells[uniq_cells != 0] # Exclude background label 0

    # Initialize output mask with zeros (uint8 type for image saving)
    mask_outlines = np.zeros_like(img, dtype=np.uint8)

    # Loop over unique cells (this loop will now run in standard Python,
    # but the operations inside using skimage are still highly optimized)
    for nC in uniq_cells: # Changed back to direct iteration as prange is not applicable without njit
        # Create binary mask for the current cell
        mask_cell = (img == nC)

        # Step 1: Find perimeter using 3D erosion
        # This line will now work correctly because the function is no longer compiled by Numba.
        eroded = binary_erosion(mask_cell, footprint=selem_erode)

        # Perimeter = original_cell_mask XOR eroded_cell_mask
        perimeter = mask_cell ^ eroded

        # Step 2: Dilate the perimeter in 3D to create a thicker outline
        dilated_perim = binary_dilation(perimeter, footprint=selem_dilate)

        # Update output mask
        mask_outlines[dilated_perim > 0] = 255

    return mask_outlines



def preview_voronoi_function(nSeeds, ellipsoidAxis1, ellipsoidAxis2, ellipsoidAxis3, progress_bar_3d=None, step=0, steps=100):
    
    # Set input data
    points = generate_random_points_on_sphere(nSeeds)
    
    radius = max([ellipsoidAxis1,ellipsoidAxis2, ellipsoidAxis3])+min([ellipsoidAxis1,ellipsoidAxis2, ellipsoidAxis3])/2
                  
    center = np.array([0, 0, 0])
    
    # progress_bar_3d['value'] = step+10 / steps * 100 # Commented out as progress_bar_3d is not defined in this scope
    # progress_bar_3d.update_idletasks() # Commented out as progress_bar_3d is not defined in this scope
    
    # Create the spherical Voronoi diagram
    sv = SphericalVoronoi(points, radius, center)
    
    # Sort vertices (optional, helpful for plotting)
    sv.sort_vertices_of_regions()
    
    # Define the elongation factors for each axis
    elongation_factors = np.array([ellipsoidAxis1/radius, ellipsoidAxis2/radius, ellipsoidAxis3/radius])  # Elongate x-axis by 1.5, y by 1.0, z by 0.5
    
    # progress_bar_3d['value'] = step+10 / steps * 100 # Commented out as progress_bar_3d is not defined in this scope
    # progress_bar_3d.update_idletasks() # Commented out as progress_bar_3d is not defined in this scope
    
    # Create the elongation transformation matrix
    elongation_matrix = np.diag(elongation_factors)
    
    # Apply the transformation to the vertices and points
    transformed_vertices = sv.vertices @ elongation_matrix.T
    transformed_points = points @ elongation_matrix.T
    
    # progress_bar_3d['value'] = step+10 / steps * 100 # Commented out as progress_bar_3d is not defined in this scope
    # progress_bar_3d.update_idletasks() # Commented out as progress_bar_3d is not defined in this scope
    
    # Plotting
    t_vals = np.linspace(0, 1, 2000)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot the unit sphere for reference (optional)
    u = np.linspace(0, 2 * np.pi, 100)
    v = np.linspace(0, np.pi, 100)
    x = np.outer(np.cos(u), np.sin(v))
    y = np.outer(np.sin(u), np.sin(v))
    z = np.outer(np.ones(np.size(u)), np.cos(v))
    ax.plot_surface(x, y, z, color='y', alpha=0.1)
    
    # progress_bar_3d['value'] = step+10 / steps * 100 # Commented out as progress_bar_3d is not defined in this scope
    # progress_bar_3d.update_idletasks() # Commented out as progress_bar_3d is not defined in this scope
    
    # Plot generator points
    ax.scatter(transformed_points[:, 0], transformed_points[:, 1], transformed_points[:, 2], c='b')
    
    # Plot Voronoi vertices
    ax.scatter(transformed_vertices[:, 0], transformed_vertices[:, 1], transformed_vertices[:, 2], c='g')
    
    # Indicate Voronoi regions (as Euclidean polygons)
    for region in sv.regions:
        n = len(region)
        for i in range(n):
            start = sv.vertices[region][i]
            end = sv.vertices[region][(i + 1) % n]
            result = geometric_slerp(start, end, t_vals)
            transformed_result = result @ elongation_matrix.T
            ax.plot(transformed_result[..., 0], transformed_result[..., 1], transformed_result[..., 2], c='k')
            
    # progress_bar_3d['value'] = step+10 / steps * 100 # Commented out as progress_bar_3d is not defined in this scope
    # progress_bar_3d.update_idletasks() # Commented out as progress_bar_3d is not defined in this scope
        
    plt.show()

# def main_spheroid(
#     cell_height,
#     n_seeds,
#     ax1, ax2, ax3, # These are now considered the OUTER ellipsoid axes
#     matrix_shape,
#     file_path,
#     elastic_deformation=False,
#     elastic_value=0,
#     lloyd_iters=5
# ):
#     import time
#     import numpy as np
#     import tifffile
#     import elasticdeform
#     from scipy.spatial import Voronoi, cKDTree
#     from skimage.morphology import ball, binary_erosion, binary_dilation
#     import SimpleITK as sitk # Ensure SimpleITK is imported here for main_spheroid

#     start_time = time.time()
#     center = np.array(matrix_shape) / 2
#     n_dots = 1000

#     # Validate cell_height against the smallest ellipsoid axis
#     min_axis_length = min(ax1, ax2, ax3)
#     if cell_height >= min_axis_length:
#         raise ValueError(f"Cell height ({cell_height}) cannot be greater than or equal to "
#                          f"the smallest ellipsoid axis half length ({min_axis_length}). "
#                          "This would result in a non-positive inner radius for the shell. Please reduce cell_height.")

#     # Define radii based on the user's new interpretation
#     outer_radii = [ax1, ax2, ax3]
#     inner_radii = [ax1 - cell_height, ax2 - cell_height, ax3 - cell_height]
#     # Seeds are placed on the surface of an ellipsoid with axes "ellipsoidAxis - cellHeight/2"
#     seed_placement_radii = [ax1 - cell_height / 2, ax2 - cell_height / 2, ax3 - cell_height / 2]

#     coords_inner = generate_ellipsoid_points(center, inner_radii, resolution=n_dots)
#     coords_outer = generate_ellipsoid_points(center, outer_radii, resolution=n_dots)
    
#     coords_seed_placement = generate_ellipsoid_points(center, seed_placement_radii, resolution=n_dots)

#     min_sep = int(closest_distance_on_ellipsoid(n_seeds, *seed_placement_radii))

#     while min_sep >= 1:
#         try:
#             central_idx = createSeeds(n_seeds, coords_seed_placement, min_sep)
#             break
#         except ValueError:
#             min_sep -= 5 if min_sep > 10 else 1
#             print(f"Retrying seed creation with min_sep = {min_sep}")

#     central_points = coords_seed_placement[central_idx]
    
#     print(f"→ Initial central seeds: {len(central_points)}")

#     # Lloyd relaxation inside the shell, using the actual inner and outer radii for constraint
#     relaxed_points, vor_for_relaxation = lloyd_relaxation3D_constrained(
#         central_points, lloyd_iters, *matrix_shape,
#         ellipsoid_center=center,
#         ellipsoid_radii_inner=inner_radii, 
#         ellipsoid_radii_outer=outer_radii 
#     )

#     inside_mask, _ = check_points_in_ellipsoidal_shell(relaxed_points, center, inner_radii, outer_radii)
#     relaxed_points = relaxed_points[inside_mask]
#     print(f"→ Kept relaxed seeds in ellipsoidal shell: {len(relaxed_points)}")

#     print("→ Directly partitioning volume with Voronoi seeds (Optimized)...")
    
#     # --- OPTIMIZATION: Chunking the cKDTree.query operation ---
#     voronoi_start_time = time.time() # Start timer for this section

#     labels_volume = np.zeros(matrix_shape, dtype=np.uint16)
#     tree = cKDTree(relaxed_points)

#     total_voxels = matrix_shape[0] * matrix_shape[1] * matrix_shape[2]
#     chunk_size = int(8 * 512 * 512) # ~8 million voxels per chunk

#     x_indices, y_indices, z_indices = np.indices(matrix_shape)
#     all_voxel_indices_flat = np.column_stack([x_indices.ravel(), y_indices.ravel(), z_indices.ravel()])

#     for i in prange(0, total_voxels, chunk_size):
#         end_idx = min(i + chunk_size, total_voxels)
        
#         voxel_coords_chunk = all_voxel_indices_flat[i:end_idx].astype(np.float32)

#         _, labels_flat_chunk = tree.query(voxel_coords_chunk)
        
#         labels_volume.ravel()[i:end_idx] = (labels_flat_chunk + 1).astype(np.uint16)
    
#     print(f"Time for Voronoi partitioning (voxel labeling): {time.time() - voronoi_start_time:.2f}s")

#     # Calculate the mask for the hollow shell based on the outer and inner radii
#     x_coords, y_coords, z_coords = np.meshgrid(np.arange(matrix_shape[0]) - center[0],
#                                                np.arange(matrix_shape[1]) - center[1],
#                                                np.arange(matrix_shape[2]) - center[2],
#                                                indexing='ij')
    
#     print("→ here 1")


#     dist_outer_solid = ((x_coords / outer_radii[0]) ** 2 +
#                         (y_coords / outer_radii[1]) ** 2 +
#                         (z_coords / outer_radii[2]) ** 2)

#     dist_inner_void = ((x_coords / inner_radii[0]) ** 2 +
#                        (y_coords / inner_radii[1]) ** 2 +
#                        (z_coords / inner_radii[2]) ** 2)
    
#     shell_mask = ((dist_outer_solid <= 1.0) & (dist_inner_void >= 1.0)).astype(np.uint8)

#     # Apply the shell mask to the labeled volume: Set labels outside the shell to 0 (background)
#     labels_volume_in_shell = labels_volume * shell_mask.astype(np.uint16) 
#     print("→ here 2")


#     # Decide which labeled volume to use for final output (potentially deformed)
#     if elastic_deformation:
#         print("→ Applying elastic deformation to labels")
#         labels_final = elasticdeform.deform_random_grid(
#             labels_volume_in_shell.astype(np.float32), sigma=elastic_value,
#             points=5, order=0, mode='nearest' 
#         ).astype(np.uint16)
#         # Re-apply shell mask in case deformation pushed labels out or pulled background in
#         labels_final = labels_final * shell_mask.astype(np.uint16)
#     else:
#         labels_final = labels_volume_in_shell

#     # Generate outlines from the final labeled volume
#     print("→ here 3")
#     # selem_erode_arg = ball(1) # Original code, not used for morph_label_outlines
#     # selem_dilate_arg = ball(1) # Original code, not used for morph_label_outlines
#     outlines = morph_label_outlines(labels_final);
#     print("→ here 4")

#     # Save pre-watershed (now the outlines derived directly from the labeled Voronoi volume)
#     prewtshd_path = file_path.replace(".tiff", "_prewtsh.tiff")
#     tifffile.imwrite(prewtshd_path, outlines) 

#     # Watershed segmentation
#     print("Applying 3D watershed segmentation...")
#     img = sitk.GetImageFromArray(outlines)
#     smoothed = sitk.SmoothingRecursiveGaussian(img, sigma=4.0)
#     sm_array = sitk.GetArrayFromImage(smoothed)
#     sm_norm = ((sm_array - sm_array.min()) / (sm_array.ptp() + 1e-8)) * 255
#     norm_img = sitk.GetImageFromArray(sm_norm.astype(np.uint8))

#     dist_map = sitk.SignedMaurerDistanceMap(norm_img, insideIsPositive=True, useImageSpacing=True)
#     segmentation = sitk.MorphologicalWatershed(sitk.GetImageFromArray(sm_array), 0.1)
#     segmentation = sitk.RelabelComponent(segmentation, 1)

#     labels = sitk.GetArrayFromImage(segmentation).astype(np.uint16)
    
#     unique_labels = np.unique(labels)
#     unique_labels = unique_labels[unique_labels != 0]
    
#     # Step 2: Compute centroids for all labels using center_of_mass
#     centroids = np.array(center_of_mass(labels > 0, labels, unique_labels))
    
#     # Step 3: Check all centroids at once
#     is_in_shell, _ = check_points_in_ellipsoidal_shell(
#         centroids, center, inner_radii, outer_radii, verbose=False
#     )
    
#     # Step 4: Remove all labels whose centroid is not in the shell
#     labels_to_remove = unique_labels[~is_in_shell]
#     mask = np.isin(labels, labels_to_remove)
#     filtered_labels = np.copy(labels)
#     filtered_labels[mask] = 0

    
#     outlines = morph_label_outlines(labels)

#     # Save output files
#     tifffile.imwrite(file_path, outlines)
#     label_path = file_path.replace(".tiff", "_labels.tiff")
#     tifffile.imwrite(label_path, labels)

#     print(f"✓ Completed in {round(time.time() - start_time, 2)}s")
#     return None
def main_spheroid(
    cell_height,
    n_seeds,
    ax1, ax2, ax3, # These are now considered the OUTER ellipsoid axes
    matrix_shape,
    file_path,
    elastic_deformation=False,
    elastic_value=0,
    lloyd_iters=5,
    watershedBool=True
):
    import time
    import numpy as np
    import tifffile
    import elasticdeform
    from scipy.spatial import Voronoi, cKDTree
    from skimage.morphology import ball, binary_erosion, binary_dilation
    import SimpleITK as sitk # Ensure SimpleITK is imported here for main_spheroid

    start_time = time.time()
    center = np.array(matrix_shape) / 2
    n_dots = 1000

    # Validate cell_height against the smallest ellipsoid axis
    min_axis_length = min(ax1, ax2, ax3)
    if cell_height >= min_axis_length:
        raise ValueError(f"Cell height ({cell_height}) cannot be greater than or equal to "
                         f"the smallest ellipsoid axis half length ({min_axis_length}). "
                         "This would result in a non-positive inner radius for the shell. Please reduce cell_height.")

    # Define radii based on the user's new interpretation
    outer_radii = [ax1, ax2, ax3]
    inner_radii = [ax1 - cell_height, ax2 - cell_height, ax3 - cell_height]
    # Seeds are placed on the surface of an ellipsoid with axes "ellipsoidAxis - cellHeight/2"
    seed_placement_radii = [ax1 - cell_height / 2, ax2 - cell_height / 2, ax3 - cell_height / 2]

    coords_inner = generate_ellipsoid_points(center, inner_radii, resolution=n_dots)
    coords_outer = generate_ellipsoid_points(center, outer_radii, resolution=n_dots)
    
    coords_seed_placement = generate_ellipsoid_points(center, seed_placement_radii, resolution=n_dots)

    min_sep = int(closest_distance_on_ellipsoid(n_seeds, *seed_placement_radii))

    while min_sep >= 1:
        try:
            central_idx = createSeeds(n_seeds, coords_seed_placement, min_sep)
            break
        except ValueError:
            min_sep -= 5 if min_sep > 10 else 1
            print(f"Retrying seed creation with min_sep = {min_sep}")

    central_points = coords_seed_placement[central_idx]
    
    print(f"→ Initial central seeds: {len(central_points)}")

    # Lloyd relaxation inside the shell, using the actual inner and outer radii for constraint
    # relaxed_points, vor_for_relaxation = lloyd_relaxation3D_constrained(
    #     central_points, lloyd_iters, *matrix_shape,
    #     ellipsoid_center=center,
    #     ellipsoid_radii_inner=inner_radii, 
    #     ellipsoid_radii_outer=outer_radii 
    # )
    center = np.array([matrix_shape[0], matrix_shape[1], matrix_shape[2]]) / 2

    relaxed_points, _ = lloyd_relaxation3D_constrained(
        central_points,
        lloyd_iters,
        matrix_shape[0],
        matrix_shape[1],
        matrix_shape[2],
        center,
        inner_radii,  
        outer_radii   
    )

    inside_mask, _ = check_points_in_ellipsoidal_shell(relaxed_points, center, inner_radii, outer_radii)
    relaxed_points = relaxed_points[inside_mask]
    print(f"→ Kept relaxed seeds in ellipsoidal shell: {len(relaxed_points)}")

    print("→ Directly partitioning volume with Voronoi seeds (Optimized)...")
    
    # --- OPTIMIZATION: Chunking the cKDTree.query operation ---
    voronoi_start_time = time.time() # Start timer for this section

    labels_volume = np.zeros(matrix_shape, dtype=np.uint16)
    tree = cKDTree(relaxed_points)
    
    print('here1')

    total_voxels = matrix_shape[0] * matrix_shape[1] * matrix_shape[2]
    chunk_size = int(8 * 512 * 512) # ~8 million voxels per chunk
    print('here2')
    print(matrix_shape)
    x_indices, y_indices, z_indices = np.indices(matrix_shape)
    all_voxel_indices_flat = np.column_stack([x_indices.ravel(), y_indices.ravel(), z_indices.ravel()])
    print('here3')

    for i in prange(0, total_voxels, chunk_size):
        end_idx = min(i + chunk_size, total_voxels)
        
        voxel_coords_chunk = all_voxel_indices_flat[i:end_idx].astype(np.float32)

        _, labels_flat_chunk = tree.query(voxel_coords_chunk)
        
        labels_volume.ravel()[i:end_idx] = (labels_flat_chunk + 1).astype(np.uint16)
    
    print(f"Time for Voronoi partitioning (voxel labeling): {time.time() - voronoi_start_time:.2f}s")

    # Calculate the mask for the hollow shell based on the outer and inner radii
    x_coords, y_coords, z_coords = np.meshgrid(np.arange(matrix_shape[0]) - center[0],
                                               np.arange(matrix_shape[1]) - center[1],
                                               np.arange(matrix_shape[2]) - center[2],
                                               indexing='ij')
    
    print("→ Calculating shell mask...") # Added print for clarity


    dist_outer_solid = ((x_coords / outer_radii[0]) ** 2 +
                        (y_coords / outer_radii[1]) ** 2 +
                        (z_coords / outer_radii[2]) ** 2)

    dist_inner_void = ((x_coords / inner_radii[0]) ** 2 +
                       (y_coords / inner_radii[1]) ** 2 +
                       (z_coords / inner_radii[2]) ** 2)
    
    shell_mask = ((dist_outer_solid <= 1.0) & (dist_inner_void >= 1.0)).astype(np.uint8)

    # Apply the shell mask to the labeled volume: Set labels outside the shell to 0 (background)
    labels_volume_in_shell = labels_volume * shell_mask.astype(np.uint16) 
    print("→ Applied shell mask to labels.") # Added print for clarity


    # Decide which labeled volume to use for final output (potentially deformed)
    if elastic_deformation:
        print("→ Applying elastic deformation to labels")
        labels_final = elasticdeform.deform_random_grid(
            labels_volume_in_shell.astype(np.float32), sigma=elastic_value,
            points=5, order=0, mode='nearest' 
        ).astype(np.uint16)
        # Re-apply shell mask in case deformation pushed labels out or pulled background in
        labels_final = labels_final * shell_mask.astype(np.uint16)
    else:
        labels_final = labels_volume_in_shell

    # Generate outlines from the final labeled volume
    print("→ Generating outlines from labels...") # Added print for clarity
    outlines = morph_label_outlines(labels_final);
    print("→ Outlines generated.") # Added print for clarity

    # Save pre-watershed (now the outlines derived directly from the labeled Voronoi volume)
    prewtshd_path = file_path.replace(".tiff", ".tiff")
    tifffile.imwrite(prewtshd_path, outlines) 

    if watershedBool:
        # Watershed segmentation
        print("→ Applying 3D watershed segmentation...") # Changed print for clarity
        img = sitk.GetImageFromArray(outlines)
        smoothed = sitk.SmoothingRecursiveGaussian(img, sigma=4.0)
        sm_array = sitk.GetArrayFromImage(smoothed)
        sm_norm = ((sm_array - sm_array.min()) / (sm_array.ptp() + 1e-8)) * 255
        norm_img = sitk.GetImageFromArray(sm_norm.astype(np.uint8))
    
        dist_map = sitk.SignedMaurerDistanceMap(norm_img, insideIsPositive=True, useImageSpacing=True)
        segmentation = sitk.MorphologicalWatershed(sitk.GetImageFromArray(sm_array), 0.1, markWatershedLine=False, fullyConnected=True)
        segmentation = sitk.RelabelComponent(segmentation, 1)
    
        labels = sitk.GetArrayFromImage(segmentation).astype(np.uint16)
        
        unique_labels = np.unique(labels)
        unique_labels = unique_labels[unique_labels != 0]
        
        # Step 2: Compute centroids for all labels using center_of_mass
        centroids = np.array(center_of_mass(labels > 0, labels, unique_labels))
        
        # Step 3: Check all centroids at once
        is_in_shell, _ = check_points_in_ellipsoidal_shell(
            centroids, center, inner_radii, outer_radii, verbose=False
        )
        
        # Step 4: Remove all labels whose centroid is not in the shell
        labels_to_remove = unique_labels[~is_in_shell]
        mask = np.isin(labels, labels_to_remove)
        filtered_labels = np.copy(labels)
        filtered_labels[mask] = 0
    
        
        outlines = morph_label_outlines(filtered_labels) # Changed to use filtered_labels for outline generation
    
        # Save output files
        outlines_path = file_path.replace(".tiff", "watershed_outlines.tiff")
        tifffile.imwrite(file_path, outlines)
        label_path = file_path.replace(".tiff", "watershed_labels.tiff")
        tifffile.imwrite(label_path, filtered_labels) # Changed to save filtered_labels
        

    print(f"✓ Completed in {round(time.time() - start_time, 2)}s")
    return None
def calculate_mean_distance(side_1, side_2, num_seeds):
    # Calculate the total area of the rectangle
    total_area = side_1 * side_2
    
    # Calculate the approximate area per seed
    area_per_seed = total_area / num_seeds
    
    # Calculate the side lengths of the rectangle
    side_length_1 = math.sqrt(area_per_seed * side_1 / side_2)
    side_length_2 = math.sqrt(area_per_seed * side_2 / side_1)
    
    # Calculate the mean distance between seeds
    mean_distance = math.sqrt((side_length_1 ** 2 + side_length_2 ** 2) / 2)
    
    return mean_distance

def draw_seeds(matrix, points, radius):
    # Ensure matrix is uint8, required by OpenCV
    matrix = matrix.astype(np.uint8)
    for point in points:
        # Ensure the center is within the bounds of the matrix
        center = (int(point[0]), int(point[1]))  # Note: OpenCV expects (col, row)
        cv2.circle(matrix, center, radius, 1, -1)  # 1 for value, -1 to fill the circle
    return matrix

def preview_voronoi_2D(nSeeds, sideX, sideY, matrix_shape, progress_bar_2d=None, step=0, steps=100, elasticDeformation=False, elasticDeformation_value=5, showNuclei=False, nucleiSize=5, minMembraneThickness=3, maxMembraneThickness=7, lloydIters=5):
    timeDef = time.time()

    # Advanced data
    nDots = 1000
    outerNSeeds = nSeeds
    samplesPerFace = 10000 
    thickness = round(0.5*max(sideX, sideY))

    planeCenter1 = matrix_shape[0] / 2
    planeCenter2 = matrix_shape[1] / 2

    minimumSeparation = calculate_mean_distance(sideX, sideY, nSeeds)
    minimumSeparationOut = calculate_mean_distance(sideX + thickness, sideY + thickness, nSeeds)

    central_coordinates = generate_grid_points([thickness + planeCenter1, thickness + planeCenter2], [sideX, sideY], nDots)
    
    outer_full_coordinates = generate_grid_points([thickness + planeCenter1, thickness + planeCenter2], [sideX + 2 * thickness, sideY + 2 * thickness], nDots)
    
    # Calculate central region bounds
    central_min_x = thickness + planeCenter1 - sideX / 2
    central_max_x = thickness + planeCenter1 + sideX / 2
    central_min_y = thickness + planeCenter2 - sideY / 2
    central_max_y = thickness + planeCenter2 + sideY / 2
    
    # Filter outer coordinates to form a proper frame
    outer_coordinates = []
    for coord in outer_full_coordinates:
        x, y = coord
        if (
            x < central_min_x or x > central_max_x or  # Left or right of central region
            y < central_min_y or y > central_max_y     # Above or below central region
        ):
            outer_coordinates.append(coord)
    
    # Convert to NumPy array for further processing
    outer_coordinates = np.array(outer_coordinates)
    
    # if progress_bar_2d: # Commented out as progress_bar_2d is not defined in this scope
    #     progress_bar_2d['value'] = (step + 10) / steps * 100
    #     progress_bar_2d.update_idletasks()

    while minimumSeparation >= 1:
        try:
            central_pstarts = createSeeds(nSeeds, central_coordinates, minimumSeparation)
            break
        except ValueError:
            if minimumSeparation > 10:
                minimumSeparation -= 10
            else:
                minimumSeparation -= 1
            print('Retrying with minSep_', minimumSeparation)

    central_points = central_coordinates[central_pstarts]

    # if progress_bar_2d: # Commented out as progress_bar_2d is not defined in this scope
    #     progress_bar_2d['value'] = (step + 10) / steps * 100
    #     progress_bar_2d.update_idletasks()

    while minimumSeparationOut >= 1:
        try:
            outer_pstarts = createSeeds(outerNSeeds, outer_coordinates, minimumSeparationOut)
            break
        except ValueError:
            if minimumSeparationOut > 10:
                minimumSeparationOut -= 10
            else:
                minimumSeparationOut -= 1
            print('Retrying with minSep_', minimumSeparationOut)

    # if progress_bar_2d: # Commented out as progress_bar_2d is not defined in this scope
    #     progress_bar_2d['value'] = (step + 10) / steps * 100
    #     progress_bar_2d.update_idletasks()

    outer_points = outer_coordinates[outer_pstarts]

    points = np.concatenate([outer_points, central_points])

    # if progress_bar_2d: # Commented out as progress_bar_2d is not defined in this scope
    #     progress_bar_2d['value'] = (step + 10) / steps * 100
    #     progress_bar_2d.update_idletasks()

    print('time locating dots')
    timeDots = time.time() - timeDef
    print(timeDots)
    timeDots = time.time()

    points, vor = lloyd_relaxation(points, lloydIters, sideX+2*thickness, sideY+2*thickness);
        
    print('time voronoi')
    timeVor = time.time() - timeDots
    print(timeVor)
    timeVor = time.time()
    
    img = draw_voronoi_with_oscillating_membranes(vor, sideX+2*thickness, sideY+2*thickness, minMembraneThickness, maxMembraneThickness, sideX+2*thickness, sideY+2*thickness)
    matrix = np.array(img)
    matrix = 255-matrix
    matrix = np.all(matrix == 255, axis=-1).astype(np.float64)
    
    print('time doing image stuff')
    timeImage = time.time() - timeVor
    print(timeImage)
    timeImage = time.time()

    # Increase the size of the matrix
    larger_shape = (matrix_shape[0] + 2 * thickness, matrix_shape[1] + 2 * thickness)
    
    if showNuclei:
        matrixNuclei = np.zeros_like(matrix, dtype=np.float64)
        matrixNuclei = draw_seeds(matrixNuclei, points, nucleiSize)
        matrixNuclei = matrixNuclei.astype(np.float64)

    # Deform the image (if desired)
    if elasticDeformation:
        if showNuclei:
            matrix_deformed, matrixNuclei_deformed = elasticdeform.deform_random_grid([matrix, matrixNuclei], sigma=elasticDeformation_value, points=3)
            matrixNuclei = (matrixNuclei_deformed > 0.2) * 1
        else:
            matrix_deformed = elasticdeform.deform_random_grid(matrix, sigma=elasticDeformation_value, points=3)

        matrix = (matrix_deformed > 0.2) * 1

    # Clip the matrix back to the original size
    clipped_matrix = matrix[thickness:thickness + matrix_shape[0], thickness:thickness + matrix_shape[1]]
    if showNuclei:
        clipped_matrixNuclei = matrixNuclei[thickness:thickness + matrix_shape[0], thickness:thickness + matrix_shape[1]]
        clipped_matrix = np.stack((clipped_matrix, clipped_matrixNuclei), axis=-1)
    # Convert the matrix to an image
    binary_matrix, outerRectangle = create_rectangle([planeCenter1, planeCenter2], [sideX, sideY], matrix_shape)

    # clipped_matrix = clipped_matrix * binary_matrix #+ outerRectangle
    clipped_matrix[clipped_matrix > 0] = 255
    clipped_matrix = clipped_matrix.astype('uint8')

    # Convert the matrix to an image
    binary_matrix, outerRectangle = create_rectangle([planeCenter1, planeCenter2], [sideX, sideY], matrix_shape)

    # if progress_bar_2d: # Commented out as progress_bar_2d is not defined in this scope
    #     progress_bar_2d['value'] = (step + 10) / steps * 100
    #     progress_bar_2d.update_idletasks()

    if showNuclei:
        clipped_matrix = clipped_matrix[:, :, 0]+clipped_matrix[:, :, 1]
    
    #clipped_matrix = clipped_matrix * binary_matrix # + outerRectangle
    clipped_matrix[clipped_matrix > 0] = 255

    # if progress_bar_2d: # Commented out as progress_bar_2d is not defined in this scope
    #     progress_bar_2d['value'] = (step + 20) / steps * 100
    #     progress_bar_2d.update_idletasks()

    return clipped_matrix



def main_2D(nSeeds, sideX, sideY, matrix_shape, name, elasticDeformation=False, elasticDeformation_value=5, showNuclei=False, nucleiSize=5, minMembraneThickness=3, maxMembraneThickness=4, lloydIters=5, saveAs_tif=True, saveAs_jpg=True):
    timeDef = time.time()

    nDots = 1000
    outerNSeeds = 2 * nSeeds
    samplesPerFace = 10000

    thickness = round(0.5 * max(sideX, sideY))

    planeCenter1 = matrix_shape[0] / 2
    planeCenter2 = matrix_shape[1] / 2

    minimumSeparation = calculate_mean_distance(sideX, sideY, nSeeds)
    minimumSeparationOut = calculate_mean_distance(sideX + thickness, sideY + thickness, nSeeds)

    central_coordinates = generate_grid_points([thickness + planeCenter1, thickness + planeCenter2], [sideX, sideY], nDots)
    outer_full_coordinates = generate_grid_points([thickness + planeCenter1, thickness + planeCenter2], [sideX + 2 * thickness, sideY + 2 * thickness], nDots)

    central_min_x = thickness + planeCenter1 - sideX / 2
    central_max_x = thickness + planeCenter1 + sideX / 2
    central_min_y = thickness + planeCenter2 - sideY / 2
    central_max_y = thickness + planeCenter2 + sideY / 2

    outer_coordinates = [
        coord for coord in outer_full_coordinates
        if coord[0] < central_min_x or coord[0] > central_max_x or
           coord[1] < central_min_y or coord[1] > central_max_y
    ]
    outer_coordinates = np.array(outer_coordinates)

    while minimumSeparation >= 1:
        try:
            inner_pstarts = createSeeds(nSeeds, central_coordinates, minimumSeparation)
            break
        except ValueError:
            minimumSeparation -= 10 if minimumSeparation > 10 else 1

    central_points = central_coordinates[inner_pstarts]

    while minimumSeparationOut >= 1:
        try:
            outer_pstarts = createSeeds(outerNSeeds, outer_coordinates, minimumSeparationOut)
            break
        except ValueError:
            minimumSeparationOut -= 10 if minimumSeparationOut > 10 else 1

    outer_points = outer_coordinates[outer_pstarts]
    points = np.concatenate([outer_points, central_points])

    points, vor = lloyd_relaxation(points, lloydIters, sideX + 2 * thickness, sideY + 2 * thickness)

    img = draw_voronoi_with_oscillating_membranes(
        vor, sideX + 2 * thickness, sideY + 2 * thickness,
        minMembraneThickness, maxMembraneThickness,
        sideX + 2 * thickness, sideY + 2 * thickness
    )
    matrix = np.array(img)
    matrix = 255 - matrix
    matrix = np.all(matrix == 255, axis=-1).astype(np.float64)

    if showNuclei:
        matrixNuclei = np.zeros_like(matrix, dtype=np.float64)
        matrixNuclei = draw_seeds(matrixNuclei, points, nucleiSize).astype(np.float64)

    if elasticDeformation:
        if showNuclei:
            matrix, matrixNuclei = elasticdeform.deform_random_grid([matrix, matrixNuclei], sigma=elasticDeformation_value, points=3)
            matrix = (matrix > 0.2) * 1
            matrixNuclei = (matrixNuclei > 0.2) * 1
        else:
            matrix = elasticdeform.deform_random_grid(matrix, sigma=elasticDeformation_value, points=3)
            matrix = (matrix > 0.2) * 1

    # Clip to final matrix shape
    clipped_matrix = matrix[thickness:thickness + matrix_shape[0], thickness:thickness + matrix_shape[1]]
    if showNuclei:
        clipped_matrixNuclei = matrixNuclei[thickness:thickness + matrix_shape[0], thickness:thickness + matrix_shape[1]]
        clipped_matrix = np.stack((clipped_matrix, clipped_matrixNuclei), axis=-1)

    clipped_matrix[clipped_matrix > 0] = 255
    clipped_matrix = clipped_matrix.astype('uint8')

    if showNuclei:
        clipped_matrix = np.transpose(clipped_matrix, (2, 0, 1))

    if saveAs_tif:
        tifffile.imwrite(name, clipped_matrix)

    if saveAs_jpg:
        if showNuclei:
            name_jpg = name.replace("tiff", "jpg")
            cv2.imwrite(name_jpg, clipped_matrix[0])
            name_nuclei_jpg = name.replace("tiff", "_nuclei.jpg")
            cv2.imwrite(name_nuclei_jpg, clipped_matrix[1])
        else:
            name_jpg = name.replace("tiff", "jpg")
            cv2.imwrite(name_jpg, clipped_matrix)

    return vor
