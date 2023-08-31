import cv2
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

import configs


COLOR_LIST_1 = ["red", "blue", "green", "slateblue", "purple", "orange", "pink", "brown", "gray", "olive", "cyan", "magenta"]
COLOR_LIST_2 = ["lime", "teal", "aqua", "maroon", "navy", "gold", "darkred", "darkblue", "darkgreen", "thistle", "violet", "burlywood"]


def create_standard_calibration_points():
    calibration_points_1 = []
    for i in range(configs.row_num):
        calibration_points_2 = []
        for j in range(configs.col_num):
            x = configs.left + configs.text_width / 2 + j * configs.text_width
            y = configs.top + configs.text_height / 2 + i * configs.text_height
            calibration_points_2.append([x, y])
        calibration_points_1.append(calibration_points_2)
    return calibration_points_1


def compute_centroids(df_cali):
    centroids_list = [[[] for _ in range(configs.col_num)] for _ in range(configs.row_num)]

    df_cali_grouped_by_matrix_y = df_cali.groupby("matrix_y")
    for matrix_y, df_cali_matrix_y in df_cali_grouped_by_matrix_y:
        df_cali_grouped_by_matrix_x = df_cali_matrix_y.groupby("matrix_x")
        for matrix_x, df_cali_matrix_x in df_cali_grouped_by_matrix_x:
            df_cali_matrix_x = df_cali_matrix_x[df_cali_matrix_x["gaze_x"] != "failed"]
            df_cali_matrix_x["gaze_x"] = df_cali_matrix_x["gaze_x"].astype(float)
            df_cali_matrix_x["gaze_y"] = df_cali_matrix_x["gaze_y"].astype(float)
            centroids_x = np.mean(df_cali_matrix_x["gaze_x"].tolist())
            centroids_y = np.mean(df_cali_matrix_x["gaze_y"].tolist())
            centroids_list[int(matrix_y)][int(matrix_x)] = [centroids_x, centroids_y]

    return centroids_list


def apply_homography_transform(centroids, std_cali_points, col_num=configs.col_num):
    list_centroids = np.array(centroids).reshape(-1, 1, 2)
    list_std_cali_points = np.array(std_cali_points).reshape(-1, 1, 2)
    M, mask = cv2.findHomography(list_centroids, list_std_cali_points, cv2.RANSAC, 5.0)
    centroids_transformed = cv2.perspectiveTransform(list_centroids, M)
    centroids_transformed = centroids_transformed.reshape(-1, col_num, 2)
    return M, centroids_transformed


def get_homography_matrix_for_calibration(df_cali):
    centroids_list = compute_centroids(df_cali)
    homography_matrix, _ = apply_homography_transform(centroids_list, create_standard_calibration_points())
    return homography_matrix


def visualize_original_cali_centroids(file_path):
    '''
    visual centroids of each calibraiton
    :param file_path:
    :param scaling_factor:
    :return:
    '''

    df_cali = pd.read_csv(file_path)
    centroid_list = compute_centroids(df_cali)
    list_std_cali = create_standard_calibration_points()
    H, centroid_list_after_homography = apply_homography_transform(centroid_list, list_std_cali)
    print(H)

    fig, ax = plt.subplots(figsize=(16, 9))
    ax.set_aspect('equal')
    ax.set_xlim(0, 1920)
    ax.set_ylim(0, 1200)
    ax.invert_yaxis()

    # centroid_list_1d = np.array(centroid_list).reshape(-1, 2).tolist()
    # list_std_cali_1d = np.array(list_std_cali).reshape(-1, 2).tolist()

    for i in range(len(centroid_list)):
        for j in range(len(centroid_list[0])):
            # color = None
            # if j % 2 == 0:
            #     color = COLOR_LIST_1[i]
            # else:
            #     color = COLOR_LIST_2[i]
            ax.scatter(list_std_cali[i][j][0], list_std_cali[i][j][1], color=[0.6, 0.6, 0.6], s=10)
            ax.scatter(centroid_list[i][j][0], centroid_list[i][j][1], color=[0.7, 0.2, 0.2], s=10)
            ax.scatter(centroid_list_after_homography[i][j][0], centroid_list_after_homography[i][j][1], color=[0.2, 0.2, 0.7], s=10)

    plt.title('Centroid Data Visualization')
    plt.xlabel('Centroid X')
    plt.ylabel('Centroid Y')

    plt.show()


def compute_bias_between_cali_centroids_and_std_points(cali_points_1d, std_points_1d, H):
    transformed_cali_points_1d = cv2.perspectiveTransform(cali_points_1d.reshape(-1, 1, 2), H).reshape(-1, 2)
    bias = np.mean(np.sqrt(np.sum(np.square(transformed_cali_points_1d - std_points_1d), axis=1)))
    return bias

