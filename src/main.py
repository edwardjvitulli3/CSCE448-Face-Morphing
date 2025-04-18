import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.spatial import Delaunay
import json

def loadImages(image1_name, image2_name):
    base_dirrectory = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    image1_path = os.path.join(base_dirrectory, "Images", image1_name)
    image2_path = os.path.join(base_dirrectory, "Images", image2_name)
    image1 = plt.imread(image1_path)
    image2 = plt.imread(image2_path)
    return image1, image2

def collectCorrespondences(image1, image2):
    points_image1 = []
    points_image2 = []

    # Creating plot
    figure, (axis1, axis2) = plt.subplots(1, 2, figsize=(10, 5))
    figure.canvas.manager.set_window_title("Collect Correspondences")
    axis1.imshow(image1)
    axis1.set_title("Image 1")
    axis2.imshow(image2)
    axis2.set_title("Image 2")
    plt.suptitle("Select 15 point pairs. Start with Image 1, then go back and forth.")

    # Collecting 15 point pairs
    for i in range(15):
        # Selecting image1 point
        plt.subplot(1, 2, 1)
        point1 = plt.ginput(1, timeout=-1)
        points_image1.append(point1[0])
        # Drawing point
        axis1.scatter(point1[0][0], point1[0][1], c='red', marker='x')
        figure.canvas.draw()

        # Select image2 point
        plt.subplot(1, 2, 2)
        point2 = plt.ginput(1, timeout=-1)
        points_image2.append(point2[0])
        # Drawing point
        axis2.scatter(point2[0][0], point2[0][1], c='blue', marker='x')
        figure.canvas.draw()

    plt.close(figure)

    # Converting points to numpy arrays
    points_image1 = np.array(points_image1, dtype=np.float32)
    points_image2 = np.array(points_image2, dtype=np.float32)

    # Adding fixed points
    fixed_points_image1, fixed_points_image2 = getFixedPoints(image1, image2)
    points_image1 = np.vstack([points_image1, fixed_points_image1])
    points_image2 = np.vstack([points_image2, fixed_points_image2])

    return points_image1, points_image2

def getFixedPoints(image1, image2):
    # Image 1
    h, w, _ = image1.shape
    corners1 = np.array([
        [0, 0],
        [w - 1, 0],
        [0, h - 1],
        [w - 1, h - 1]], dtype=np.float32)
    edge_centers1 = np.array([
        [w // 2, 0],
        [w // 2, h - 1],
        [0, h // 2],
        [w - 1, h // 2]], dtype=np.float32)
    fixed_points_image1 = np.vstack([corners1, edge_centers1])

    # Image 2
    h, w, _ = image2.shape
    corners2 = np.array([
        [0, 0],
        [w - 1, 0],
        [0, h - 1],
        [w - 1, h - 1]], dtype=np.float32)
    edge_centers2 = np.array([
        [w // 2, 0],
        [w // 2, h - 1],
        [0, h // 2],
        [w - 1, h // 2]], dtype=np.float32)
    fixed_points_image2 = np.vstack([corners2, edge_centers2])

    return fixed_points_image1, fixed_points_image2

def displayTriangulation(image1, image2, points_image1, points_image2, triangles_img1):
    # Creating plot
    figure, (axis1, axis2) = plt.subplots(1, 2, figsize=(15, 7))
    figure.canvas.manager.set_window_title("Delaunay Triangulation")

    # Image 1
    axis1.imshow(image1)
    axis1.triplot(points_image1[:, 0], points_image1[:, 1], triangles_img1.simplices.copy(), color='red')
    axis1.set_title("Delaunay Triangulation on Image 1")

    # Image 2
    axis2.imshow(image2)
    axis2.triplot(points_image2[:, 0], points_image2[:, 1], triangles_img1.simplices.copy(), color='blue')
    axis2.set_title("Delaunay Triangulation from Image 1 Applied to Image 2")

    plt.show()

def saveCorrespondencesToJSON(points_image1, points_image2, filename="correspondences.json"):
    data = {
        "points_image1": points_image1.tolist(),
        "points_image2": points_image2.tolist()
    }
    with open(filename, 'w') as file:
        json.dump(data, file)
    print(f"Correspondences saved to {filename}")

def loadCorrespondencesFromJSON(filename="correspondences.json"):
    with open(filename, 'r') as file:
        data = json.load(file)
    points_image1 = np.array(data['points_image1'], dtype=np.float32)
    points_image2 = np.array(data['points_image2'], dtype=np.float32)
    print(f"Correspondences loaded from {filename}")
    return points_image1, points_image2

def main():
    # Loading images
    image1_name = 'musk.jpg'
    image2_name = 'trump.jpg'
    image1, image2 = loadImages(image1_name, image2_name)

    # Collecting/loading correspondences
    choice = input("Load existing correspondences (L) or select new ones (S)? ").strip().lower()
    if choice == 'l' or choice == 'L':
        points_image1, points_image2 = loadCorrespondencesFromJSON()
    else:
        points_image1, points_image2 = collectCorrespondences(image1, image2)
        save_choice = input("Save these correspondences to JSON? (Y/N) ").strip().lower()
        if save_choice == 'y':
            saveCorrespondencesToJSON(points_image1, points_image2)

    # Delaunay triangulation
    triangles_image1 = Delaunay(points_image1)

    # Displaying triangulation
    displayTriangulation(image1, image2, points_image1, points_image2, triangles_image1)

    # Looping through triangles, finding intermediate triangle, getting matrix for points from image1 and image2, and warping
    num_frames = 30
    for i in range(1, num_frames + 1): # 30 frames
        for simplex in triangles_image1.simplices: # Each triangle
            # Getting points for intermediate triangle
            alpha = i / num_frames
            interpolated_points = (1- alpha) * points_image1[simplex] + alpha * points_image2[simplex]
            print(f"Interpolated points for triangle {simplex}: {interpolated_points}")

            # Getting matrix for points from image1 and image2

            # Warp points and save frame

    # Save as GIF

if __name__ == "__main__":
    main()