import cv2
import numpy as np


def apply_homography_to_reading(df_reading, homography_matrix):
    gaze_pos_list = []
    for gaze_index in range(df_reading.shape[0]):
        gaze_x = df_reading.iloc[gaze_index]["gaze_x"]
        gaze_y = df_reading.iloc[gaze_index]["gaze_y"]
        gaze_pos_list.append([[gaze_x, gaze_y]])

    transformed_gaze_pos_list = cv2.perspectiveTransform(np.array(gaze_pos_list), homography_matrix)
    transformed_gaze_pos_list = transformed_gaze_pos_list.reshape(-1, 2)
    df_reading["gaze_x"] = transformed_gaze_pos_list[:, 0]
    df_reading["gaze_y"] = transformed_gaze_pos_list[:, 1]
    return df_reading

