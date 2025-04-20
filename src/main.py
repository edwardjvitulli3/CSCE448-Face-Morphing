import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.spatial import Delaunay
import json
import PySimpleGUI as sg
from PIL import Image, ImageTk, ImageSequence
import imageio.v3 as iio

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
    with open('../' + filename, 'r') as file:
        data = json.load(file)
    points_image1 = np.array(data['points_image1'], dtype=np.float32)
    points_image2 = np.array(data['points_image2'], dtype=np.float32)
    print(f"Correspondences loaded from {filename}")
    return points_image1, points_image2

def computeTransformMatrix(interpolated, source):
    A = np.hstack([interpolated, np.ones((3, 1))])
    B = source
    M = np.linalg.solve(A, B)
    return M.T

def computeTransform(transform, x, y):
    pt = np.array([x, y, 1.0])
    return transform @ pt

def getBoundingBox(triangle):
    x = triangle[:, 0]
    y = triangle[:, 1]
    
    minX = int(np.floor(np.min(x)))
    maxX = int(np.ceil(np.max(x)))
    
    minY = int(np.floor(np.min(y)))
    maxY = int(np.ceil(np.max(y)))
    
    return minX, maxX, minY, maxY

def edge(p1, p2, p):
    return (p2[0] - p1[0]) * (p[1] - p1[1]) - (p2[1] - p1[1]) * (p[0] - p1[0])

def pointsInBound(pts, triangle):
    a, b, c = triangle
    v0 = c - a
    v1 = b - a
    v2 = pts - a
    dot00 = np.dot(v0, v0)
    dot01 = np.dot(v0, v1)
    dot11 = np.dot(v1, v1)
    dot02 = np.sum(v2 * v0, axis = 1)
    dot12 = np.sum(v2 * v1, axis = 1)
    denom = dot00 * dot11 - dot01 * dot01
    inv_denom = 1.0 / denom
    u = (dot11 * dot02 - dot01 * dot12) * inv_denom
    v = (dot00 * dot12 - dot01 * dot02) * inv_denom
    
    return (u >= 0) & (v >= 0) & (u + v <= 1)
    
def bilinearInterpolation(img, pts):
    h, w, c = img.shape
    x = pts[:, 0]
    y = pts[:, 1]
    
    x0 = np.clip(np.floor(x).astype(int), 0, w - 1)
    x1 = np.clip(x0 + 1, 0, w - 1)
    y0 = np.clip(np.floor(y).astype(int), 0, h - 1)
    y1 = np.clip(y0 + 1, 0, h - 1)
    
    a = x - x0
    b = y - y0
    
    w1 = (1 - a) * (1 - b)
    w2 = a * (1 - b)
    w3 = a * b
    w4 = (1 - a) * b
    
    return w1[:, None] * img[y0, x0] + w2[:, None] * img[y0, x1] + w3[:, None] * img[y1, x1] + w4[:, None] * img[y1, x0]

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
    #displayTriangulation(image1, image2, points_image1, points_image2, triangles_image1)
    
    #solve for transform matrix from image

    # Looping through triangles, finding intermediate triangle, getting matrix for points from image1 and image2, and warping
    num_frames = 30
    frames = []
    for i in range(1, num_frames + 1): # 30 frames
        print("computing frame", i)
        alpha = i / num_frames
        newImage = np.zeros_like(image1, dtype=np.uint8)
        for simplex in triangles_image1.simplices: # Each triangle
            # Getting points for intermediate triangle
            tri_A_pts = points_image1[simplex]
            tri_B_pts = points_image2[simplex]
            tri_T_pts = (1- alpha) * tri_A_pts + alpha * tri_B_pts
            #print(f"Interpolated points for triangle {simplex}: {interpolated_points}")
            #find the affine transform
            A_transform_matrix = computeTransformMatrix(tri_T_pts, tri_A_pts)
            B_transform_matrix = computeTransformMatrix(tri_T_pts, tri_B_pts)
            #using vectorized operations for optimization
            #create bound box
            minX, maxX, minY, maxY = getBoundingBox(tri_T_pts)
            #clamp values to image dimensions
            minX = max(minX, 0)
            maxX = min(maxX, newImage.shape[1])
            minY = max(minY, 0)
            maxY = min(maxY, newImage.shape[0])
            #create a pixel grid
            px, py = np.meshgrid(np.arange(minX, maxX), np.arange(minY, maxY))
            pts = np.vstack((px.ravel(), py.ravel())).T
            #create mask for points inside triangle
            mask = pointsInBound(pts, tri_T_pts)
            if not np.any(mask):
                continue
            valid_pts = pts[mask]
            px_valid, py_valid = valid_pts[:, 0], valid_pts[:, 1]
            #apply affine transform
            ones = np.ones_like(px_valid)
            homogeneous = np.vstack([px_valid, py_valid, ones])
            A_pts = (A_transform_matrix @ homogeneous).T
            B_pts = (B_transform_matrix @ homogeneous).T
            #interpolate pixel values
            pixels_A = bilinearInterpolation(image1, A_pts)
            pixels_B = bilinearInterpolation(image2, B_pts)
            #blend pixel values from both images
            cross_dissolved = ((1 - alpha) * pixels_A + alpha * pixels_B).astype(np.uint8)
            newImage[py_valid, px_valid] = cross_dissolved
        frames.append(newImage)

    # Save as GIF
    path = '../Results/'
    name = image1_name.replace('.jpg', '') + '_' + image2_name.replace('.jpg', '') + '_morph.gif'
    iio.imwrite(path + name, frames, duration=0.5, loop=0)
    
    #uncomment to see gif in a gui
    
    # layout = [[sg.Image(key='-IMAGE-')]]
    # window = sg.Window('GIF Output', layout, element_justification='c')
    
    # while True:
    #     for frame in ImageSequence.Iterator(Image.open(path + name)):
    #         event, values = window.read(timeout=100)
    #         if event == sg.WIN_CLOSED:
    #             exit(0)
    #         window['-IMAGE-'].update(data=ImageTk.PhotoImage(frame))

if __name__ == "__main__":
    main()