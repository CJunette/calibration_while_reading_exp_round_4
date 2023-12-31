import ast
import concurrent.futures

import math
import os
import random
import time
from multiprocessing import Pool
import cv2
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from scipy.spatial import distance
from sklearn.cluster import DBSCAN
from sklearn.ensemble import IsolationForest
from scipy.optimize import linear_sum_assignment
from scipy.linalg import svd
import analyse_calibration_data
import configs
import event_detection
import read_files
import test_pull_gaze_trace


'''--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------'''
'''COMPUTE DISTANCE'''


def gaussian_pdf(x, mean, cov):
    k = len(mean)
    det_cov = np.linalg.det(cov)
    inv_cov = np.linalg.inv(cov)
    norm_const = 1.0 / (np.power(2 * np.pi, k / 2) * np.sqrt(det_cov))
    delta = x - mean
    return norm_const * np.exp(-0.5 * np.dot(delta.T, np.dot(inv_cov, delta)))


def compute_distance_simple(x, mean, cov):
    # pdf = gaussian_pdf(x, mean, cov)
    # return 1 / (pdf + 1e-10)
    dist = np.linalg.norm(x - mean)
    return dist


def compute_absolute_distance_matrix(gaze_data, std_data):
    std_data_matrix = std_data[:, np.newaxis, :]
    gaze_data_matrix = gaze_data[np.newaxis, :, :]
    distance_vector_matrix = std_data_matrix - gaze_data_matrix
    distance_matrix = np.linalg.norm(distance_vector_matrix, axis=-1)
    return distance_matrix


def compute_structural_distance_matrix(gaze_data, std_data, gaze_density, text_mapping_with_weight, gaze_para_id_list):
    '''
    本函数目的在于对文本中出现标点、空格或没有任何内容的text_unit，施加一定的惩罚。
    同时根据gaze_density，对可能存在的“视觉重点”（目前仅包含每行的首末）进行判定。
    :param gaze_data:
    :param std_data:
    :param gaze_density:
    :param text_mapping_with_weight:
    :param gaze_para_id_list:
    :return:
    '''
    def add_weight_to_blank_weight_list(df, key):
        blank_weight_list = [configs.empty_text_unit_penalty for _ in range(configs.row_num * configs.col_num)]

        for text_unit_index in range(df.shape[0]):
            row = df["row"].iloc[text_unit_index]
            col = df["col"].iloc[text_unit_index]
            weight = df[key].iloc[text_unit_index]
            token = df["word"].iloc[text_unit_index]
            if token.strip() == "":
                continue
            if token in configs.punctuation_list:
                weight = configs.punctuation_text_unit_penalty
            blank_weight_list[row * configs.col_num + col] = weight

        return blank_weight_list
    # 首先根据不同的para_id，把weight分配好。
    para_id_list = text_mapping_with_weight["para_id"].unique()
    para_id_list = np.sort(para_id_list)

    gaze_para_id_segment = []
    gaze_para_id_sub_segment = []
    para_id_index = 0
    gaze_index = 0
    while gaze_index < len(gaze_para_id_list) and para_id_index <= max(para_id_list):
        if gaze_para_id_list[gaze_index] == para_id_list[para_id_index]:
            gaze_para_id_sub_segment.append(gaze_index)
            gaze_index += 1
        else:
            gaze_para_id_segment.append(gaze_para_id_sub_segment)
            gaze_para_id_sub_segment = []
            para_id_index += 1
            if gaze_para_id_list[gaze_index] != para_id_list[para_id_index]:
                raise Exception("para_id_list和gaze_para_id_list不匹配。")
        if gaze_index == len(gaze_para_id_list) - 1:
            gaze_para_id_segment.append(gaze_para_id_sub_segment)
            gaze_index += 1

    # 然后对每个para_id，分别分配不同的weight。
    horizontal_weight_matrix = np.array([[configs.empty_text_unit_penalty for _ in range(len(gaze_data))] for _ in range(len(std_data))])
    vertical_weight_matrix = np.array([[configs.empty_text_unit_penalty for _ in range(len(gaze_data))] for _ in range(len(std_data))])
    first_row_weight_matrix = np.array([[configs.empty_text_unit_penalty for _ in range(len(gaze_data))] for _ in range(len(std_data))])

    horizontal_weight_list_1 = []
    vertical_weight_list_1 = []
    first_row_weight_list_1 = []

    for para_id_index in range(len(para_id_list)):
        df = text_mapping_with_weight[text_mapping_with_weight["para_id"] == para_id_list[para_id_index]]
        horizontal_weight_list_2 = add_weight_to_blank_weight_list(df, "horizontal_edge_weight")
        vertical_weight_list_2 = add_weight_to_blank_weight_list(df, "vertical_edge_weight")
        first_row_weight_list_2 = add_weight_to_blank_weight_list(df, "first_row_weight")

        horizontal_weight_list_1.append(horizontal_weight_list_2)
        vertical_weight_list_1.append(vertical_weight_list_2)
        first_row_weight_list_1.append(first_row_weight_list_2)

        # 将获得的weight添加到matrix中。
        for gaze_index in gaze_para_id_segment[para_id_index]:
            horizontal_weight_matrix[:, gaze_index] = horizontal_weight_list_2
            vertical_weight_matrix[:, gaze_index] = vertical_weight_list_2
            first_row_weight_matrix[:, gaze_index] = first_row_weight_list_2
        # 根据每个gaze的密度，对是空格或标点的text_unit的weight进行调整。
        text_covered_std_indices = []
        for text_unit_index in range(df.shape[0]):
            row = df["row"].iloc[text_unit_index]
            col = df["col"].iloc[text_unit_index]
            token = df["word"].iloc[text_unit_index]
            text_covered_std_indices.append(row * configs.col_num + col)
            if token.strip() == "" or token in configs.punctuation_list:
                for gaze_index in gaze_para_id_segment[para_id_index]:
                    density = gaze_density[gaze_index]
                    if density < configs.text_unit_density_threshold_for_empty:
                        # FIXME 这里写的有点疑惑，我考虑修改一下。
                        horizontal_weight_matrix[row * configs.col_num + col, gaze_index] -= -density / 2.5
                        vertical_weight_matrix[row * configs.col_num + col, gaze_index] -= -density / 2.5
                        first_row_weight_matrix[row * configs.col_num + col, gaze_index] -= -density / 2.5
        # 对没有任何字符的text_unit，根据gaze密度进行修正。
        for text_unit_index in range(len(std_data)):
            if text_unit_index in text_covered_std_indices:
                continue
            else:
                for gaze_index in gaze_para_id_segment[para_id_index]:
                    density = gaze_density[gaze_index]
                    if density < configs.text_unit_density_threshold_for_empty:
                        density = gaze_density[gaze_index]
                        horizontal_weight_matrix[text_unit_index, gaze_index] -= -density / 2.5
                        vertical_weight_matrix[text_unit_index, gaze_index] -= -density / 2.5
                        first_row_weight_matrix[text_unit_index, gaze_index] -= -density / 2.5

    horizontal_weight_list_1 = np.array(horizontal_weight_list_1)
    vertical_weight_list_1 = np.array(vertical_weight_list_1)
    first_row_weight_list_1 = np.array(first_row_weight_list_1)

    # 合并各个weight matrix，得到最终的结果。
    structural_weight_matrix = 0.9 * horizontal_weight_matrix + 0.01 * vertical_weight_matrix + 0.09 * first_row_weight_matrix
    # gaze_density_exp = 1 + 4 / (1 + np.exp(-1 * (gaze_density - 5)))
    gaze_density_exp = 1 + 4 / (1 + np.exp(-1.1 * (gaze_density - 6)))
    gaze_density_matrix = gaze_density_exp[np.newaxis, :]
    structural_distance_matrix = np.abs(structural_weight_matrix - gaze_density_matrix)
    structural_distance_matrix = np.square(structural_distance_matrix) / 2
    # 密度匹配正确与否的差距大概在2倍，一般是0.6, 1.3或2, 4。
    return structural_distance_matrix, (horizontal_weight_matrix, vertical_weight_matrix, first_row_weight_matrix, structural_weight_matrix), (horizontal_weight_list_1, vertical_weight_list_1, first_row_weight_list_1)


def compute_semantic_distance_matrix(gaze_data, std_data, gaze_density, text_mapping_with_weight, gaze_para_id_list):
    '''
    本函数目的在于对文本中出现标点、空格或没有任何内容的text_unit，施加一定的惩罚。
    同时根据gaze_density，对可能存在的“视觉重点”（目前仅包含每行的首末）进行判定。
    :param gaze_data:
    :param std_data:
    :param gaze_density:
    :param text_mapping_with_weight:
    :param gaze_para_id_list:
    :return: combined_distance_matrix, (horizontal_weight_matrix, vertical_weight_matrix, first_row_weight_matrix, structural_weight_matrix, gpt_weight_matrix), (horizontal_weight_list, vertical_weight_list, first_row_weight_list, gpt_weight_list)
    :return: combined_distance_matrix: 综合了structural和gpt的距离的距离矩阵。
    :return: (horizontal_weight_matrix, vertical_weight_matrix, first_row_weight_matrix, structural_weight_matrix, gpt_weight_matrix): 各个weight matrix。
    :return: (horizontal_weight_list, vertical_weight_list, first_row_weight_list, gpt_weight_list): 各个weight list，用于给外面的std points上色。
    '''

    def blur_weight(weight_list):
        gaussian_filter = configs.gaussian_filter
        weight_2d = np.array(weight_list.copy()).reshape(configs.row_num, configs.col_num)
        blurred_weight_2d = np.array(weight_list.copy()).reshape(configs.row_num, configs.col_num)
        pad_row, pad_col = len(gaussian_filter) // 2, len(gaussian_filter[0]) // 2

        for row_index in range(configs.row_num):
            for col_index in range(configs.col_num):
                weighted_sum = 0
                weight_total = 0
                center_value = weight_2d[row_index][col_index]
                # Apply the Gaussian filter
                for x in range(-pad_row, pad_row + 1):
                    for y in range(-pad_col, pad_col + 1):
                        # Check if the filter goes out of bounds
                        if row_index + x < 0 or row_index + x >= configs.row_num or col_index + y < 0 or col_index + y >= configs.col_num:
                            continue
                        # Get the value from the original matrix
                        value = weight_2d[row_index + x][col_index + y]
                        # Get the corresponding weight from the Gaussian filter
                        weight = gaussian_filter[pad_row + x][pad_col + y]

                        # Only allow positive values to affect negative values
                        if center_value >= 0 and value < 0:
                            continue
                        else:
                            # (center_value < 0 and value > 0) or (center_value < 0 and value < 0) and (center_value > 0 and value > 0)
                            weighted_sum += value * weight
                            weight_total += weight

                # Calculate the new blurred value
                if weight_total != 0:
                    blurred_value = weighted_sum / weight_total
                    blurred_weight_2d[row_index][col_index] = blurred_value
        blurred_weight = blurred_weight_2d.reshape(configs.col_num * configs.row_num)
        return blurred_weight

    def compute_single_matrix_and_list(key, gaze_para_id_segment):
        weight_matrix = np.array([[configs.empty_text_unit_penalty for _ in range(len(gaze_data))] for _ in range(len(std_data))])
        weight_list_1 = []
        for para_index in range(len(para_id_list)):
            df_text_mapping = text_mapping_with_weight[text_mapping_with_weight["para_id"] == para_id_list[para_index]]
            weight_list_2 = [configs.empty_text_unit_penalty for _ in range(configs.row_num * configs.col_num)]
            for text_unit_index in range(df_text_mapping.shape[0]):
                row = df_text_mapping["row"].iloc[text_unit_index]
                col = df_text_mapping["col"].iloc[text_unit_index]
                weight = df_text_mapping[key].iloc[text_unit_index]
                token = df_text_mapping["word"].iloc[text_unit_index]
                if token.strip() == "":
                    continue
                if token in configs.punctuation_list:
                    weight = configs.punctuation_text_unit_penalty
                weight_list_2[row * configs.col_num + col] = weight

            # 如果需要，则对weight进行模糊化。
            if configs.bool_blur_weight:
                weight_list_2 = blur_weight(weight_list_2)

            weight_list_1.append(weight_list_2)

            for gaze_index in gaze_para_id_segment[para_index]:
                weight_matrix[:, gaze_index] = weight_list_2

            # 根据每个gaze的密度，对是空格或标点的text_unit的weight进行调整。
            text_covered_std_indices = []
            for text_unit_index in range(df_text_mapping.shape[0]):
                row = df_text_mapping["row"].iloc[text_unit_index]
                col = df_text_mapping["col"].iloc[text_unit_index]
                token = df_text_mapping["word"].iloc[text_unit_index]
                text_covered_std_indices.append(row * configs.col_num + col)
                if token.strip() == "" or token in configs.punctuation_list:
                    for gaze_index in gaze_para_id_segment[para_index]:
                        density = gaze_density[gaze_index]
                        if density < configs.text_unit_density_threshold_for_empty:
                            weight_matrix[row * configs.col_num + col, gaze_index] = -density - 1
            # 对没有任何字符的text_unit，根据gaze密度进行修正。
            for text_unit_index in range(len(std_data)):
                if text_unit_index in text_covered_std_indices:
                    continue
                else:
                    for gaze_index in gaze_para_id_segment[para_index]:
                        density = gaze_density[gaze_index]
                        if density < configs.text_unit_density_threshold_for_empty:
                            weight_matrix[text_unit_index, gaze_index] = -density - 1

        return weight_matrix, np.array(weight_list_1)

    def combine_structural_and_gpt_distance_matrix(structural_distance_matrix, gpt_distance_matrix, structural_weight_matrix):
        coeff_gpt = configs.coeff_gpt
        coeff_structural = configs.coeff_structural
        coeff_gpt_for_non_structural = configs.coeff_gpt_for_non_structural
        if coeff_gpt == 0 and coeff_structural != 0:
            return coeff_structural * structural_distance_matrix
        elif coeff_structural == 0 and coeff_gpt != 0:
            return coeff_gpt * gpt_distance_matrix
        elif coeff_gpt == 0 and coeff_structural == 0:
            return np.zeros(structural_distance_matrix.shape)
        else:
            combined_distance_matrix = np.array([[0 for _ in range(len(gaze_data))] for _ in range(len(std_data))])
            for std_index in range(len(std_data)):
                for gaze_index in range(len(gaze_data)):
                    # 对于那些没有structural_weight的text_unit，直接使用gpt_weight。
                    if structural_weight_matrix[std_index, gaze_index] == 0:
                        combined_distance_matrix[std_index, gaze_index] = gpt_distance_matrix[std_index, gaze_index] * coeff_gpt_for_non_structural
                    else:
                        combined_distance_matrix[std_index, gaze_index] = gpt_distance_matrix[std_index, gaze_index] * coeff_gpt + structural_distance_matrix[std_index, gaze_index] * coeff_structural
            return combined_distance_matrix


    # 首先根据不同的para_id，把weight分配好。
    para_id_list = text_mapping_with_weight["para_id"].unique()
    para_id_list = np.sort(para_id_list)

    gaze_para_id_segment = [[] for _ in range(len(para_id_list))]
    para_id_index_mapping = {para_id_list[i]: i for i in range(len(para_id_list))}
    gaze_index = 0
    while gaze_index < len(gaze_para_id_list):
        gaze_para_id = gaze_para_id_list[gaze_index]
        if gaze_para_id in para_id_list:
            para_id_index = para_id_index_mapping[gaze_para_id]
            gaze_para_id_segment[para_id_index].append(gaze_index)
            gaze_index += 1
        else:
            raise Exception("invalid para id")

    # 然后对每个para_id，分别分配不同的weight。
    horizontal_weight_matrix, horizontal_weight_list = compute_single_matrix_and_list("horizontal_edge_weight", gaze_para_id_segment)
    vertical_weight_matrix, vertical_weight_list = compute_single_matrix_and_list("vertical_edge_weight", gaze_para_id_segment)
    first_row_weight_matrix, first_row_weight_list = compute_single_matrix_and_list("first_row_weight", gaze_para_id_segment)
    gpt_weight_matrix, gpt_weight_list = compute_single_matrix_and_list("gpt_weight", gaze_para_id_segment)

    # 合并各个weight matrix，得到最终的结果。
    structural_weight_matrix = 0.9 * horizontal_weight_matrix + 0.01 * vertical_weight_matrix + 0.09 * first_row_weight_matrix
    # gaze_density_exp = 1 + 4 / (1 + np.exp(-1 * (gaze_density - 5)))
    gaze_density_exp = 1 + 4 / (1 + np.exp(-1.1 * (gaze_density - 6)))
    gaze_density_matrix = gaze_density_exp[np.newaxis, :]
    structural_distance_matrix = np.abs(structural_weight_matrix - gaze_density_matrix)
    structural_distance_matrix = np.square(structural_distance_matrix) * configs.coeff_semantic_before_output
    # 密度匹配正确与否的差距大概在2倍，一般是0.6, 1.3或2, 4。

    gpt_distance_matrix = np.abs(gpt_weight_matrix - gaze_density_matrix)
    gpt_distance_matrix = np.square(gpt_distance_matrix) * configs.coeff_semantic_before_output

    combined_distance_matrix = combine_structural_and_gpt_distance_matrix(structural_distance_matrix, gpt_distance_matrix, structural_weight_matrix)

    return combined_distance_matrix, (horizontal_weight_matrix, vertical_weight_matrix, first_row_weight_matrix, structural_weight_matrix, gpt_weight_matrix), (horizontal_weight_list, vertical_weight_list, first_row_weight_list, gpt_weight_list)


def compute_distance(absolute_distance_matrix, semantic_distance_matrix):
    hybrid_distance_matrix = absolute_distance_matrix + semantic_distance_matrix * 0.05
    return hybrid_distance_matrix


'''--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------'''
'''GENERIC ALGORITHM'''


def compute_homography_penalty(H):
    H_normalized = H / H[2, 2]
    A = H_normalized[0:2, 0:2]
    U, S_values, Vt = svd(A)
    R = np.dot(U, Vt)
    S = np.diag(S_values)

    # Calculate rotation angle in degrees
    theta = np.arctan2(R[1, 0], R[0, 0])
    theta = np.degrees(theta)

    # Calculate scale factors
    scale_x = S[0, 0]
    scale_y = S[1, 1]

    # Calculate shear parameters
    Shear = np.dot(np.linalg.inv(R), A)
    shear_x = Shear[0, 1]
    shear_y = Shear[1, 0]

    # Perspective parameters
    perspective_x = H_normalized[2, 0]
    perspective_y = H_normalized[2, 1]

    penalty = 0
    if abs(theta) > 10:
        penalty += configs.H_rotation_penalty
        # print(f"rotation penalty, theta: {theta}")
    if (not 0.85 <= scale_x <= 1.6) or (not 0.85 <= scale_y <= 1.15):
    # if (not 0.7 <= scale_x <= 1.3) or (not 0.7 <= scale_y <= 1.3):
        penalty += configs.H_scale_penalty
        # print(f"scale penalty, scale_x: {scale_x}, scale_y: {scale_y}")
    if abs(shear_x) > 0.15 or abs(shear_y) > 0.15:
        penalty += configs.H_shear_penalty
        # print(f"shear penalty, shear_x: {shear_x}, shear_y: {shear_y}")
    if abs(perspective_x) > 0.002 or abs(perspective_y) > 0.002:
        penalty += configs.H_projection_penalty
        # print(f"projection penalty, perspective_x: {perspective_x}, perspective_y: {perspective_y}")
    # 假设一个以(0, 0)和(1920, 1200)为对角点的矩形rect_1，现在希望比较经过H变换后的四边形rect_2和rect_1的面积之比。如果面积比过大，则施加惩罚。
    rect_1 = np.array([[0, 0], [1920, 0], [1920, 1200], [0, 1200]], dtype=np.float32)
    rect_2 = cv2.perspectiveTransform(rect_1.reshape(-1, 1, 2), H).reshape(-1, 2)
    space_1 = cv2.contourArea(rect_1)
    space_2 = cv2.contourArea(rect_2)
    space_ratio = space_2 / space_1
    if not 0.9 < space_ratio < 1.2:
        penalty += configs.H_space_ratio_penalty

    return penalty, theta, scale_x, scale_y, shear_x, shear_y, perspective_x, perspective_y, space_ratio


def compute_first_row_penalty(init_first_row_penalty_copy, text_unit_num_list):
    total_penalty = 0
    for para_index in range(len(init_first_row_penalty_copy)):
        init_first_row_penalty_copy[para_index] = [i if i > 0 else 0 for i in init_first_row_penalty_copy[para_index]]
        total_penalty += sum(init_first_row_penalty_copy[para_index][:30]) / text_unit_num_list[para_index][0] + sum(init_first_row_penalty_copy[para_index][-30:]) / text_unit_num_list[para_index][1]
    return total_penalty


def compute_continuous_empty_text_unit_penalty(std_point_gaze_count):
    penalty_list = []
    std_index = 0
    while std_index < len(std_point_gaze_count):
        if std_point_gaze_count[std_index] == 0:
            probe_index = 1
            while std_index + probe_index < len(std_point_gaze_count) and std_point_gaze_count[std_index + probe_index] < configs.text_unit_density_threshold_for_empty:
                probe_index += 1
            if probe_index > configs.continuous_empty_text_unit_threshold:
                penalty_list.append((probe_index - configs.continuous_empty_text_unit_threshold))
            std_index += probe_index
        std_index += 1
    penalty = 0
    for penalty_index in range(len(penalty_list)):
        penalty += penalty_list[penalty_index] * configs.continuous_empty_text_unit_penalty
    return penalty


def compute_unfitness_in_generic(H, gaze_points, std_points_1d, semantic_distance_matrix, gaze_para_id_list, init_first_row_penalty_list, init_first_row_std_index_set, text_unit_num_list, text_mapping):
    init_first_row_penalty_copy = [init_first_row_penalty_list[i].copy() for i in range(len(init_first_row_penalty_list))]
    para_id_list = np.unique(gaze_para_id_list)
    para_id_list = np.sort(para_id_list)

    # 添加一个“空置”惩罚项。对于那些没有任何gaze点匹配的std点，添加一个惩罚。
    # 先准备好需要的矩阵。
    text_unit_gaze_count = [-1 for _ in range(configs.col_num * configs.row_num)]
    row_list = text_mapping["row"].unique()
    for row_index in range(len(row_list)):
        col_list = text_mapping[text_mapping["row"] == row_index]["col"].unique()
        for col_index in range(len(col_list)):
            text_unit_gaze_count[row_index * configs.col_num + col_index] = 0

    transformed_src_pts = cv2.perspectiveTransform(gaze_points.reshape(-1, 1, 2), H).reshape(-1, 2)
    absolute_distance_matrix = compute_absolute_distance_matrix(transformed_src_pts, std_points_1d)
    absolute_dist_list = [0 for _ in range(len(gaze_points))]
    semantic_dist_list = [0 for _ in range(len(gaze_points))]
    hybrid_dist_list = [0 for _ in range(len(gaze_points))]
    gaze_corresponding_std_list = [None for _ in range(len(gaze_points))]
    for gaze_index in range(len(absolute_distance_matrix[0])):
        min_dist_to_std_index = np.argmin(absolute_distance_matrix[:, gaze_index])
        gaze_corresponding_std_list[gaze_index] = min_dist_to_std_index
        min_dist_to_std = absolute_distance_matrix[min_dist_to_std_index, gaze_index]
        # 完全离群的点，给一个定值惩罚。
        if min_dist_to_std > 200:
            min_dist_to_std = 200 * configs.far_from_text_unit_penalty
        # 对于离最近的std特别远的点（但还不至于离群），给一个额外的惩罚。
        elif min_dist_to_std > configs.dist_threshold_from_std:
            min_dist_to_std = min_dist_to_std + (min_dist_to_std - configs.dist_threshold_from_std) * configs.far_from_text_unit_penalty

        # if min_dist_to_std > 40:
        #     min_dist_to_std = configs.far_from_text_unit_penalty
        absolute_dist_list[gaze_index] += min_dist_to_std
        semantic_dist_to_std = semantic_distance_matrix[min_dist_to_std_index, gaze_index]
        semantic_dist_list[gaze_index] = semantic_dist_to_std
        # if semantic_dist_to_std > 10000:
        #     print()
        hybrid_dist_list[gaze_index] += min_dist_to_std + semantic_dist_to_std
        # hybrid_dist_list[gaze_index] += min_dist_to_std

        # 在计算距离的同时，把first_row_penalty所需要的信息也记录下来。
        para_id = int(gaze_para_id_list[gaze_index])
        para_index = np.where(para_id_list == para_id)[0][0]
        if min_dist_to_std_index in init_first_row_std_index_set[para_index]:
            init_first_row_penalty_copy[para_index][min_dist_to_std_index] = -1

        text_unit_gaze_count[min_dist_to_std_index] += 1

    # dist_from_hybrid指每个gaze点到最近的std点的物理距离与结构距离之和。
    dist_from_hybrid = np.mean(hybrid_dist_list)
    dist_from_absolute = np.mean(absolute_dist_list)
    dist_from_semantic = np.mean(semantic_dist_list)

    # dist_from_H指H过大的变化量带来的惩罚。
    dist_from_H, theta, scale_x, scale_y, shear_x, shear_y, perspective_x, perspective_y, space_ratio = compute_homography_penalty(H)

    # dist_from_fist_row指第一行缺失文字造成的惩罚。
    dist_from_fist_row = compute_first_row_penalty(init_first_row_penalty_copy, text_unit_num_list)

    # dist_from_empty_text_unit指空置的text_unit造成的惩罚。
    dist_from_empty_text_unit = compute_continuous_empty_text_unit_penalty(text_unit_gaze_count)

    total_dist = dist_from_hybrid + dist_from_H + dist_from_fist_row + dist_from_empty_text_unit

    return total_dist, (dist_from_hybrid, dist_from_absolute, dist_from_semantic, dist_from_H, dist_from_fist_row, dist_from_empty_text_unit), (theta, scale_x, scale_y, shear_x, shear_y, perspective_x, perspective_y, space_ratio), (absolute_dist_list, semantic_dist_list, hybrid_dist_list)


def prepare_first_row_penalty(text_mapping):
    para_id_list = text_mapping["para_id"].unique()
    para_id_list = np.sort(para_id_list)

    row_to_use = 2

    penalty_list_1 = []
    std_index_list_1 = []
    text_unit_num_list_1 = []
    for para_index in range(len(para_id_list)):
        df = text_mapping[text_mapping["para_id"] == para_id_list[para_index]]
        row_list = df["row"].unique()
        row_list.sort()

        length_of_row = min(row_to_use, len(row_list))
        max_row = row_list[length_of_row - 1]

        penalty_list_2 = [0 for _ in range(configs.col_num * (max_row + 1))]
        std_index_list_2 = [-1 for _ in range(configs.col_num * (max_row + 1))]
        text_unit_num_list_2 = []
        for row_index in range(min(row_to_use, len(row_list))):
            col_list = df[df["row"] == row_list[row_index]]["col"].unique()
            text_unit_num = 0
            pass_index = 0
            for index in range(len(col_list)):
                token = df[df["row"] == row_list[row_index]][df["col"] == col_list[index]]["word"].iloc[0]
                if token.strip() == "" or token in configs.punctuation_list:
                    pass_index += 1
                    penalty = 0
                else:
                    if index - pass_index == 0:
                        penalty = configs.first_row_text_penalty / ((row_index + 1) * 2) # 对于第一个非符号的text_unit，其惩罚减小。
                    else:
                        if col_list[index] < configs.col_num / 4:
                            penalty = configs.first_row_text_penalty / (row_index + 1)
                            text_unit_num += 1
                        elif col_list[index] < configs.col_num / 2:
                            penalty = configs.first_row_text_penalty / ((row_index + 1) * 2)
                            text_unit_num += 1
                        else:
                            penalty = 0
                    pass_index = 0

                penalty_list_2[row_list[row_index] * configs.col_num + index] = penalty
                std_index_list_2[row_list[row_index] * configs.col_num + index] = row_list[row_index] * configs.col_num + col_list[index]
            text_unit_num_list_2.append(text_unit_num)

        penalty_list_1.append(penalty_list_2)
        std_index_list_1.append(set(std_index_list_2))
        text_unit_num_list_1.append(text_unit_num_list_2)

    return penalty_list_1, std_index_list_1, text_unit_num_list_1


def generic_algorithm_to_find_best_homography(src_pts, dst_pts, gaze_points, std_points_1d, semantic_distance_matrix, H_init, gaze_para_id_list, text_mapping, log_file,
                                              init_first_row_penalty_list, init_first_row_std_index_set, text_unit_num_list,
                                              std_points_color_list=None, population_size=configs.generic_population_size, generations=configs.generic_population_generation):
    '''
    :param src_pts: 来自gaze的匹配点。
    :param dst_pts: 来自标准校准点的匹配点。
    :param gaze_points: 原始gaze眼动数据。
    :param std_points_1d: 标准校准点数据。
    :param semantic_distance_matrix: 结构语义距离矩阵。
    :param H_init: 初始的H，第一轮时，是对角阵；之后是上一轮最优的矩阵。
    :param log_file:
    :param text_mapping:
    :param gaze_para_id_list:
    :param std_points_color_list: 在可视化时使用的std points的颜色列表。如果存在多个para，则只能呈现一个，具体哪一个需要在外面设置。
    :param population_size: 种群大小。
    :param generations: 迭代次数。
    :return:
    '''
    def crossover(parent_1, parent_2):
        child = (parent_1 + parent_2) / 2
        return child

    def mutate(H):
        mutation_matrix = np.array([
            [np.random.uniform(-0.1, 0.1), np.random.uniform(-0.01, 0.01), 0],
            [np.random.uniform(-0.01, 0.01), np.random.uniform(-0.1, 0.1), 0],
            [np.random.uniform(-0.01, 0.01), np.random.uniform(-0.01, 0.01), 0]])
        H += mutation_matrix
        H[0][2] = np.random.uniform(-configs.text_width*0.5, configs.text_width*0.5)
        H[1][2] = np.random.uniform(-configs.text_height*0.5, configs.text_height*0.5)
        return H

    def select(population, unfitness_result):
        unfitness_result = np.array(unfitness_result)
        unfitness = unfitness_result[:, 0]
        unfitness_args = unfitness_result[:, 2]
        unfitness_dist_details = unfitness_result[:, 1]
        unfitness_dist_list_details = unfitness_result[:, 3]
        sorted_indices = np.argsort(unfitness)
        sorted_unfitness = np.sort(unfitness)
        selected_population = [population[i] for i in sorted_indices[:population_size // 2]]
        selected_args = [unfitness_args[i] for i in sorted_indices[:population_size // 2]]
        semantic_dist_list_of_best = unfitness_dist_list_details[sorted_indices[0]][1]
        return selected_population, sorted_unfitness, unfitness_dist_details, selected_args, semantic_dist_list_of_best

    def init(H_init):
        population = [H_init]
        for population_index in range(int(population_size / 4)):
            random_set = np.random.choice(len(src_pts), size=int(len(src_pts) / 5))
            selected_scr_points = src_pts[random_set]
            selected_dst_points = dst_pts[random_set]
            try:
                H, _ = cv2.findHomography(selected_scr_points.reshape(-1, 1, 2), selected_dst_points.reshape(-1, 1, 2))
            except Exception as error:
                break
            if H is not None and not np.isnan(H).any() and not np.isinf(H).any():
                population.append(H)
        while len(population) < population_size:
            # 添加一批随机矩阵。
            random_scale_matrix = np.array([
                [np.random.uniform(0.8, 1.2), 0, 0],
                [0, np.random.uniform(0.8, 1.2), 0],
                [0, 0, 1]])
            random_theta = np.radians(np.random.uniform(-10, 10))
            random_rotation_matrix = np.array([
                [np.cos(random_theta), -np.sin(random_theta), 0],
                [np.sin(random_theta), np.cos(random_theta), 0],
                [0, 0, 1]])
            random_transform_matrix = np.array([
                [1, 0, np.random.uniform(-configs.text_width * 2, configs.text_width * 2)],
                [0, 1, np.random.uniform(-configs.text_width * 2, configs.text_width * 2)],
                [0, 0, 1]])
            random_matrix = np.dot(random_transform_matrix, np.dot(random_rotation_matrix, random_scale_matrix))
            # random_matrix = np.array([
            #     [np.random.uniform(-0.2, 0.2) + 1, 0, np.random.uniform(-configs.text_width * 2, configs.text_width * 2)],
            #     [0, np.random.uniform(-0.2, 0.2) + 1, np.random.uniform(-configs.text_height * 2, configs.text_height * 2)],
            #     [0, 0, 1]])
            population.append(random_matrix)
        return population

    # initiate
    population = init(H_init)

    with Pool(configs.num_of_processes) as p:
    # with concurrent.futures.ThreadPoolExecutor() as executor:
        last_gaze_points = gaze_points.copy()

        for generation in range(generations):
            print(f"generation: {generation}")
            if configs.bool_log:
                log_file.write("-" * 100 + "\n" + f"generation: {generation}\ntime:{time.time()}\n")

            args_list = []
            unfitness = []
            for H in population:
                args_list.append((H, gaze_points, std_points_1d, semantic_distance_matrix, gaze_para_id_list, init_first_row_penalty_list, init_first_row_std_index_set, text_unit_num_list, text_mapping))
                # unfitness.append(compute_unfitness_in_generic(H, gaze_points, std_points_1d, semantic_distance_matrix, gaze_para_id_list, init_first_row_penalty_list, init_first_row_std_index_set, text_unit_num_list, text_mapping))
            unfitness = p.starmap(compute_unfitness_in_generic, args_list)
            # unfitness = executor.map(compute_unfitness_in_generic, *zip(*args_list))
            selected_population, sorted_unfitness, unfitness_dist_details, selected_args, semantic_dist_list_of_best = select(population, unfitness)
            best_H = selected_population[0]
            print(f"best H: {selected_population[0].tolist()}, best unfitness: {sorted_unfitness[0]}, best args: {selected_args[0]}\n"
                  f"best unfitness_detail: {unfitness_dist_details[0]}")
            if configs.bool_log:
                log_file.write(f"best H: {selected_population[0].tolist()}\n"
                               f"best unfitness: {sorted_unfitness[0]}\n"
                               f"best args: {selected_args[0]}\n"
                               f"best unfitness_detail: {unfitness_dist_details[0]}\n")
            # print(f"unfitness: {sorted_unfitness[:10]}")
            transformed_gaze_points = cv2.perspectiveTransform(gaze_points.reshape(-1, 1, 2), best_H).reshape(-1, 2)

            new_population = []
            while len(new_population) < population_size // 2:
                parent1, parent2 = random.choices(selected_population, k=2)
                if np.isnan(parent1).any() or np.isnan(parent2).any() or np.isinf(parent1).any() or np.isinf(parent2).any():
                    continue
                child_1 = crossover(parent1, parent2)
                new_population.append(child_1)

            # 仅对70%做变异。
            for i in range(len(new_population)):
                if np.random.rand() < 0.7:
                    new_population[i] = mutate(new_population[i])
            population = selected_population + new_population

            # # visualize result.
            # if not configs.bool_log and not configs.bool_save_pic:
            #     max_semantic_dist = max(semantic_dist_list_of_best)
            #     gaze_color_list = [(semantic_dist_list_of_best[i] / max_semantic_dist, 0, 0) for i in range(len(semantic_dist_list_of_best))]
            #     fig, ax = plt.subplots(figsize=(12, 8))
            #     ax.set_xlim(0, 1920)
            #     ax.set_ylim(1200, 0)
            #     ax.set_aspect("equal")
            #     if std_points_color_list is not None:
            #         ax.scatter(std_points_1d[:, 0], std_points_1d[:, 1], label='std point', color=std_points_color_list, marker="x")
            #     else:
            #         ax.scatter(std_points_1d[:, 0], std_points_1d[:, 1], label='std point', color='black')
            #     ax.scatter(gaze_points[:, 0], gaze_points[:, 1], label='original gaze', color='blue', alpha=0.5)
            #     ax.scatter(last_gaze_points[:, 0], last_gaze_points[:, 1], label='original gaze', color='green', alpha=0.5)
            #     ax.scatter(transformed_gaze_points[:, 0], transformed_gaze_points[:, 1], label='transformed gaze', color=gaze_color_list, alpha=0.5)
            #     for pair_index in range(len(src_pts)):
            #         plt.plot([src_pts[pair_index][0], dst_pts[pair_index][0]], [src_pts[pair_index][1], dst_pts[pair_index][1]], color='#DDDDDD', alpha=0.5)
            #     plt.show()

            last_gaze_points = transformed_gaze_points.copy()

        final_args_list = []
        unfitness = []
        for H in population:
            final_args_list.append((H, gaze_points, std_points_1d, semantic_distance_matrix, gaze_para_id_list, init_first_row_penalty_list, init_first_row_std_index_set, text_unit_num_list, text_mapping))
            # unfitness.append(compute_unfitness_in_generic(H, gaze_points, std_points_1d, semantic_distance_matrix, gaze_para_id_list, init_first_row_penalty_list, init_first_row_std_index_set, text_unit_num_list, text_mapping))

        unfitness = p.starmap(compute_unfitness_in_generic, final_args_list)
        selected_population, sorted_unfitness, unfitness_dist_details, selected_args, semantic_dist_list_of_best = select(population, unfitness)
        best_H = selected_population[0]

        print(f"final best H: {selected_population[0].tolist()}, final best unfitness: {sorted_unfitness[0]}, final best args: {selected_args[0]}\n"
              f"final best unfitness_detail: {unfitness_dist_details[0]}")

        return best_H, sorted_unfitness[0], semantic_dist_list_of_best


'''--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------'''
'''MATCHING'''


def min_match(responsibilities):
    col_ind = [-1 for _ in range(len(responsibilities))]
    row_ind = [i for i in range(len(responsibilities))]
    for std_index in range(len(responsibilities)):
        min_index = np.argmin(responsibilities[std_index])
        col_ind[std_index] = min_index
    new_col_ind = []
    new_row_ind = []
    for std_index in range(len(col_ind)):
        if col_ind[std_index] != -1:
            new_col_ind.append(col_ind[std_index])
            new_row_ind.append(std_index)
    return row_ind, col_ind


def min_match_with_weighted_dist(gaze_points, std_points_1d, absolute_distance_matrix, semantic_distance_matrix):
    responsibilities = compute_distance(absolute_distance_matrix, semantic_distance_matrix)
    pairs_indices_of_std = [[] for _ in range(len(responsibilities))]
    gaze_pair_points = [[0, 0] for _ in range(len(responsibilities))]
    std_pair_points = [std_points_1d[i] for i in range(len(std_points_1d))]
    # 对每个gaze point，找到对应的“距离”最近的std point。
    for gaze_index in range(len(responsibilities[0])):
        min_std_index = np.argmin(responsibilities[:, gaze_index])
        pairs_indices_of_std[min_std_index].append(gaze_index)
    # 再遍历每个std point，如果有多个gaze point，就取平均；
    # 如果没有，则看这个std point对应的最小距离是否小于configs.empty_text_unit_punishment，如果是，则取最小的gaze point，如果不是，则跳过这个std point（对应空格、标点或什么都没有）。
    # TODO 这里的最小距离判断得改一下。
    # TODO 另外这里的匹配关系感觉也得修改下。
    for std_index in range(len(pairs_indices_of_std)):
        if len(pairs_indices_of_std[std_index]) > 0:
            # mean_point = np.array([0., 0.])
            target_list = []
            dist_reverse_list = []
            sub_std_index = 0
            while sub_std_index < len(pairs_indices_of_std[std_index]):
                dist = responsibilities[std_index][pairs_indices_of_std[std_index][sub_std_index]]
                # if dist > abs(configs.empty_text_unit_penalty):
                #     pairs_indices_of_std[std_index].pop(sub_std_index)
                # else:
                #     mean_point += gaze_points[pairs_indices_of_std[std_index][sub_std_index]]
                #     sub_std_index += 1
                target_list.append(gaze_points[pairs_indices_of_std[std_index][sub_std_index]])
                dist_reverse_list.append(1 / dist + 1e-10)
                sub_std_index += 1
            # if len(pairs_indices_of_std[std_index]) == 0:
            #     continue
            # else:
            #     mean_point /= len(pairs_indices_of_std[std_index])
            #     gaze_pair_points[std_index] = mean_point
            weight_center = np.array([0.0, 0.0])
            for direction_index in range(len(target_list)):
                weight_center += target_list[direction_index] * dist_reverse_list[direction_index]
            weight_center /= np.sum(dist_reverse_list)
            gaze_pair_points[std_index] = weight_center

        # elif len(pairs_indices_of_std[std_index]) == 0 and responsibilities[std_index].min() < abs(configs.empty_text_unit_penalty):
        elif len(pairs_indices_of_std[std_index]) == 0 and responsibilities[std_index].min() < math.pow(configs.empty_text_unit_penalty, 2):
            min_gaze_index = np.argmin(responsibilities[std_index])
            gaze_pair_points[std_index] = gaze_points[min_gaze_index]
            pairs_indices_of_std[std_index].append(min_gaze_index)

    pair_index = 0
    # 删掉那些[0, 0]的点，因为那些点要么已经被取平均了，要么就是距离大于configs.empty_text_unit_punishment的点。
    while pair_index < len(gaze_pair_points):
        if gaze_pair_points[pair_index][0] == 0 and gaze_pair_points[pair_index][1] == 0:
            del gaze_pair_points[pair_index]
            del std_pair_points[pair_index]
        else:
            pair_index += 1

    return np.array(gaze_pair_points), np.array(std_pair_points)


'''--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------'''
'''EM SOLUTION'''


def em_solution(std_points_1d, df_gaze_data, gaze_density, text_mapping, cali_point_1d, file_name, log_file=None, max_iteration=configs.em_iteration):
    if configs.bool_log and log_file is None:
        raise Exception("log_file is None.")

    H = np.eye(3)

    gaze_x = df_gaze_data["gaze_x"].tolist()
    gaze_y = df_gaze_data["gaze_y"].tolist()
    gaze_points = np.array([[gaze_x[i], gaze_y[i]] for i in range(len(gaze_x))])
    original_gaze_points = gaze_points.copy()
    gaze_para_id_list = df_gaze_data["matrix_x"].tolist()
    # semantic_distance_matrix不随位置变化而变化，因此只计算一次。
    semantic_distance_matrix, (horizontal_weight_matrix, vertical_weight_matrix, first_row_weight_matrix, structural_weight_matrix, gpt_weight_matrix), (horizontal_weight_list, vertical_weight_list, first_row_weight_list, gpt_weight_list)\
        = compute_semantic_distance_matrix(gaze_points, std_points_1d, np.array(gaze_density), text_mapping, gaze_para_id_list)
    print("完成语义距离计算。")

    # 创建根据weight得到的std points的颜色列表。假设存在多篇文章，那么只能呈现其中一篇的结果。
    para_index_for_color = 0
    hybrid_weight_list_of_first_para = 0.1 * horizontal_weight_list[para_index_for_color] + 0.01 * vertical_weight_list[para_index_for_color] + 0.09 * first_row_weight_list[para_index_for_color]
    std_points_color_list = []
    for std_index in range(len(hybrid_weight_list_of_first_para)):
        if hybrid_weight_list_of_first_para[std_index] < 0:
            color = [0.8, 0.8, 0.8]
        else:
            max_hybrid_weight_of_first_para = max(hybrid_weight_list_of_first_para)
            color = [hybrid_weight_list_of_first_para[std_index]/max_hybrid_weight_of_first_para, hybrid_weight_list_of_first_para[std_index]/max_hybrid_weight_of_first_para, 0]
        std_points_color_list.append(color)
    print("完成std颜色准备。")

    # 为first_row_penalty做准备。
    init_first_row_penalty_list, init_first_row_std_index_set, text_unit_num_list = prepare_first_row_penalty(text_mapping)
    print("完成first row penalty准备。")

    for iteration in range(max_iteration):
        print(f"iteration: {iteration}")
        if configs.bool_log:
            log_file.write("*"*200 + "\n" + f"iteration: {iteration}\ntime: {time.time()}\n")

        # E-step: Assign each reading to the Gaussian in std under current H with the highest PDF.
        gaze_points = cv2.perspectiveTransform(original_gaze_points.reshape(-1, 1, 2), H).reshape(-1, 2)
        absolute_distance_matrix = compute_absolute_distance_matrix(gaze_points, std_points_1d)
        reading_pair_points, std_pair_points = min_match_with_weighted_dist(gaze_points, std_points_1d, absolute_distance_matrix, semantic_distance_matrix)
        # M-step: Compute the H and update Gaussian parameters based on the assignments.
        # 使用遗传算法来找到最优的H。
        H, least_unfitness, semantic_dist_list_of_best = generic_algorithm_to_find_best_homography(reading_pair_points, std_pair_points, original_gaze_points, std_points_1d, semantic_distance_matrix, H, gaze_para_id_list, text_mapping, log_file,
                                                                                                   init_first_row_penalty_list, init_first_row_std_index_set, text_unit_num_list, std_points_color_list)
        transformed_gaze_points = cv2.perspectiveTransform(original_gaze_points.reshape(-1, 1, 2), H).reshape(-1, 2)
        bias = analyse_calibration_data.compute_bias_between_cali_centroids_and_std_points(cali_point_1d, std_points_1d, H)
        print(H, least_unfitness)
        print(f"BIAS: {bias}")
        if configs.bool_log:
            log_file.write(f"best H: {H.tolist()}\n"
                           f"best unfitness: {least_unfitness}\n"
                           f"BIAS: {bias}\n\n")
            log_file.flush()

        # init_first_row_penalty_list, init_first_row_std_index_set, text_unit_num_list = prepare_first_row_penalty(text_mapping)
        # unfitness = compute_unfitness_in_generic(H, gaze_points, std_points_1d, semantic_distance_matrix, gaze_para_id_list, init_first_row_penalty_list, init_first_row_std_index_set,
        #                                              text_unit_num_list, text_mapping)

        # visualize data
        if configs.bool_save_pic:
            fig, ax = plt.subplots(figsize=(24, 18))
        else:
            fig, ax = plt.subplots(figsize=(12, 10))
        ax.set_xlim(0, 1920)
        ax.set_ylim(1200, 0)
        ax.set_aspect("equal")
        ax.scatter(std_points_1d[:, 0], std_points_1d[:, 1], label='std point', color=std_points_color_list, marker='x')
        ax.scatter(gaze_points[:, 0], gaze_points[:, 1], label='gaze gaze', color='green', alpha=0.5)
        ax.scatter(original_gaze_points[:, 0], original_gaze_points[:, 1], label='original gaze', color='blue', alpha=0.5)
        ax.scatter(transformed_gaze_points[:, 0], transformed_gaze_points[:, 1], label='transformed gaze', color='red', alpha=0.5)
        for pair_index in range(len(reading_pair_points)):
            plt.plot([reading_pair_points[pair_index][0], std_pair_points[pair_index][0]], [reading_pair_points[pair_index][1], std_pair_points[pair_index][1]], color='#DDDDDD', alpha=0.5)
        if configs.bool_save_pic:
            save_path = f"image/match_with_density/{configs.round}/{file_name}/{configs.test_str}/"
            if not os.path.exists(save_path):
                os.makedirs(save_path)
            plt.savefig(f"{save_path}para_0-4, iter_{iteration}, bias_{bias:.3f}.png")
        else:
            plt.show()
            pass

    log_file.close()

def manual_test(std_points_1d, df_gaze_data, gaze_density, text_mapping, cali_point_1d, file_name, max_iteration=5):
    print(file_name)
    H = np.eye(3)
    # for matrix_x, df_reading_data_matrix_x in df_reading_data_group_by_matrix_x:
    responsibilities = np.zeros((len(std_points_1d), df_gaze_data.shape[0]))
    gaze_x = df_gaze_data["gaze_x"].tolist()
    gaze_y = df_gaze_data["gaze_y"].tolist()
    gaze_points = np.array([[gaze_x[i], gaze_y[i]] for i in range(len(gaze_x))])
    original_gaze_points = gaze_points.copy()
    gaze_para_id_list = df_gaze_data["matrix_x"].tolist()

    # structural_distance_matrix不随位置变化而变化，因此只计算一次。
    semantic_distance_matrix, (horizontal_weight_matrix, vertical_weight_matrix, first_row_weight_matrix, structural_weight_matrix, gpt_weight_matrix), (horizontal_weight_list, vertical_weight_list, first_row_weight_list, gpt_weight_list)\
        = compute_semantic_distance_matrix(gaze_points, std_points_1d, np.array(gaze_density), text_mapping, gaze_para_id_list)

    # 创建根据weight得到的std points的颜色列表。假设存在多篇文章，那么只能呈现其中一篇的结果。
    para_index_for_color = 0
    hybrid_weight_list_of_first_para = 0.9 * horizontal_weight_list[para_index_for_color] + 0.01 * vertical_weight_list[para_index_for_color] + 0.09 * first_row_weight_list[para_index_for_color]
    std_points_color_list = []
    for std_index in range(len(hybrid_weight_list_of_first_para)):
        if hybrid_weight_list_of_first_para[std_index] < 0:
            color = [0.8, 0.8, 0.8]
        else:
            max_hybrid_weight_of_first_para = max(hybrid_weight_list_of_first_para)
            color = [hybrid_weight_list_of_first_para[std_index] / max_hybrid_weight_of_first_para, hybrid_weight_list_of_first_para[std_index] / max_hybrid_weight_of_first_para, 0]
        std_points_color_list.append(color)

    init_first_row_penalty_list, init_first_row_std_index_set, text_unit_num_list = prepare_first_row_penalty(text_mapping)

    for iteration in range(max_iteration):
        print(f"iteration: {iteration}")
        # E-step: Assign each reading to the Gaussian in std under current H with the highest PDF.
        std_left = 0
        std_right = 1920
        std_top = 0
        std_bottom = 1200
        std_points = np.array([[std_left, std_top], [std_right, std_top], [std_right, std_bottom], [std_left, std_bottom]])
        input_text = input()
        # input_left_top, input_right_top, input_right_bottom, input_left_bottom = input_text.split(";")
        # left_top = [int(input_left_top.split(",")[0]), int(input_left_top.split(",")[1])]
        # right_top = [int(input_right_top.split(",")[0]), int(input_right_top.split(",")[1])]
        # right_bottom = [int(input_right_bottom.split(",")[0]), int(input_right_bottom.split(",")[1])]
        # left_bottom = [int(input_left_bottom.split(",")[0]), int(input_left_bottom.split(",")[1])]
        # input_points = np.array([left_top, right_top, right_bottom, left_bottom])
        # H, Mask = cv2.findHomography(std_points.reshape(-1, 1, 2), input_points.reshape(-1, 1, 2))
        if input_text == "NEXT":
            return
        H_text = input_text[1:-1]
        if not "," in H_text:
            H_text = H_text.replace("[ ", "[").replace("  ", " ").replace(" ", ", ")
        H = np.array(ast.literal_eval(H_text))

        # H = np.array([[ 9.35482351e-01 ,-1.08674326e-02 ,-6.62895792e+00],[ 6.78065368e-03 ,1.11921044e+00, 1.84089353e+01],[ 0.00000000e+00 ,0.00000000e+00 ,1.00000000e+00]])
        print(H)
        transformed_gaze_points = cv2.perspectiveTransform(gaze_points.reshape(-1, 1, 2), H).reshape(-1, 2)
        unfitness = compute_unfitness_in_generic(H, gaze_points, std_points_1d, semantic_distance_matrix, gaze_para_id_list, init_first_row_penalty_list, init_first_row_std_index_set, text_unit_num_list, text_mapping)
        print(unfitness[0], unfitness[1], unfitness[2])

        target_dist_list = unfitness[3][1]
        max_semantic_dist = max(target_dist_list)
        target_dist_list = np.array(target_dist_list)
        gaze_color_list = [(target_dist_list[i] / max_semantic_dist, 0, 0) for i in range(len(target_dist_list))]

        bias = analyse_calibration_data.compute_bias_between_cali_centroids_and_std_points(cali_point_1d, std_points_1d, H)
        print(f"BIAS: {bias}")
        fig, ax = plt.subplots(figsize=(16, 12))
        # fig, ax = plt.subplots(figsize=(12, 10))
        ax.set_xlim(0, 1920)
        ax.set_ylim(1200, 0)
        ax.set_aspect("equal")
        check_start = 0
        check_end = None
        ax.scatter(std_points_1d[:, 0], std_points_1d[:, 1], label='std point', color=std_points_color_list, marker="x")
        ax.scatter(gaze_points[check_start:check_end, 0], gaze_points[check_start:check_end, 1], label='original gaze', color='green', alpha=0.5)
        ax.scatter(original_gaze_points[check_start:check_end, 0], original_gaze_points[check_start:check_end, 1], label='original gaze', color='blue', alpha=0.5)
        ax.scatter(transformed_gaze_points[check_start:check_end, 0], transformed_gaze_points[check_start:check_end, 1], label='transformed gaze', color=gaze_color_list[check_start:check_end], alpha=0.5)
        plt.show()


'''--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------'''
'''COMPUTE DENSITY IN FINAL FUNCTION'''


def compute_density(distances, gaze_index):
    density = 0
    for probe_index in range(gaze_index + 1, len(distances[gaze_index])):
        if distances[gaze_index][probe_index] < configs.reading_density_distance_threshold:
            density += 1
        else:
            break
    for probe_index in range(gaze_index - 1, -1, -1):
        if distances[gaze_index][probe_index] < configs.reading_density_distance_threshold:
            density += 1
        else:
            break
    return density


def check_is_outlier(gaze_x, gaze_y, bound_left, bound_right, bound_top, bound_bottom):
    if bound_left <= gaze_x <= bound_right and bound_top <= gaze_y <= bound_bottom:
        return 0
    else:
        return 1


'''--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------'''
'''FINAL FUNCTION'''


def match_with_density():
    std_points = analyse_calibration_data.create_standard_calibration_points()
    std_points_1d = np.array(std_points).reshape(-1, 2)

    reading_data_list = read_files.read_reading_data()
    cali_data_list = read_files.read_calibration_data()
    text_mapping_list = read_files.read_all_modified_reading_text_mapping()
    text_mapping_with_weight = read_files.read_reading_text_mapping_with_weight()

    for file_index in range(len(text_mapping_list)):
        text_mapping_list[file_index]["horizontal_edge_weight"] = text_mapping_with_weight["horizontal_edge_weight"]
        text_mapping_list[file_index]["vertical_edge_weight"] = text_mapping_with_weight["vertical_edge_weight"]
        text_mapping_list[file_index]["first_row_weight"] = text_mapping_with_weight["first_row_weight"]
        text_mapping_list[file_index]["gpt_weight"] = text_mapping_with_weight["gpt_weight"]

    # # 为reading数据和calibration数据添加一个偏移。
    # for file_index in range(len(reading_data_list)):
    #     x_offset = 0
    #     y_offset = 0
    #     for iteration_index in range(len(reading_data_list[file_index])):
    #         reading_df = reading_data_list[file_index][iteration_index]
    #         reading_df = reading_df[reading_df["gaze_x"] != "failed"]
    #         test_pull_gaze_trace.add_distance_to_data(reading_df, x_offset, y_offset)
    #         reading_data_list[file_index][iteration_index] = reading_df
    #     cali_df = cali_data_list[file_index]
    #     cali_df = cali_df[cali_df["gaze_x"] != "failed"]
    #     test_pull_gaze_trace.add_distance_to_data(cali_df, x_offset, y_offset)
    #     cali_data_list[file_index] = cali_df

    # 计算calibration数据的centroid。
    cali_centroid_list = []
    for file_index in range(len(cali_data_list)):
        cali_centroids = np.array(analyse_calibration_data.compute_centroids(cali_data_list[file_index]))
        cali_centroids = cali_centroids.reshape(-1, 2)
        cali_centroid_list.append(cali_centroids)

    # 将reading数据按照matrix_x进行分组。
    combined_reading_data_list = []
    for file_index in range(len(reading_data_list)):
        # df_list = []
        # for iteration_index in range(len(reading_data_list[file_index])):
        #     df_group_by_matrix_x = reading_data_list[file_index][iteration_index].groupby("matrix_x")
        #     for para_id, df_matrix_x in df_group_by_matrix_x:
        #         df_list.append(df_matrix_x)
        # df_list.sort(key=lambda x: x["matrix_x"].iloc[0])
        # combined_reading_data_list.append(pd.concat(df_list, ignore_index=True))
        reading_data_sorted_combined = read_files.sort_reading_data_by_para_id(reading_data_list[file_index])
        combined_reading_data_list.append(reading_data_sorted_combined)

    # 用于保存实验使用样本的序号和quality。
    training_data_index_and_quality_list = []
    # 读取reading quality数据。
    reading_quality_list = read_files.read_quality_prediction()

    if configs.bool_pick_by_quality:
        # 利用reading quality选择数据集。
        training_reading_data = []
        training_text_mapping = []
        for file_index in range(len(combined_reading_data_list)):
            quality_list = configs.target_quality_list
            df_quality = reading_quality_list[file_index]
            data_index_and_quality = []
            for quality_index in range(len(quality_list)):
                para_id_of_target_quality = df_quality[df_quality["prediction"] == quality_list[quality_index]]["para_id"].tolist()
                para_id_of_target_quality = random.sample(para_id_of_target_quality, configs.num_of_sample_of_quality[quality_index])
                for para_index in range(len(para_id_of_target_quality)):
                    data_index_and_quality.append([para_id_of_target_quality[para_index], quality_list[quality_index]])
            data_index_and_quality.sort(key=lambda x: x[0])
            para_id_list = [data_index_and_quality[i][0] for i in range(len(data_index_and_quality))]
            quality_of_para_id_list = [data_index_and_quality[i][1] for i in range(len(data_index_and_quality))]
            print(f"para_id: {para_id_list}\nquality: {quality_of_para_id_list}")
            training_data_index_and_quality_list.append(data_index_and_quality)
            df_reading = combined_reading_data_list[file_index]
            df_text_mapping = text_mapping_list[file_index]
            training_reading_data.append(df_reading[df_reading["matrix_x"].isin(para_id_list)])
            training_text_mapping.append(df_text_mapping[df_text_mapping["para_id"].isin(para_id_list)])
    else:
        # 利用给定的集合作为数据集。
        # 截取部分reading data和text_mapping作为训练集。
        training_data_num_list = configs.training_number_list
        training_reading_data = []
        training_text_mapping = []
        for file_index in range(len(combined_reading_data_list)):
            df_reading = combined_reading_data_list[file_index]
            training_reading_data.append(df_reading[df_reading["matrix_x"].isin(training_data_num_list)])
            df_text_mapping = text_mapping_list[file_index]
            training_text_mapping.append(df_text_mapping[df_text_mapping["para_id"].isin(training_data_num_list)])
            data_index_and_quality = []
            df_reading_quality = reading_quality_list[file_index]
            for training_file_index in range(len(training_data_num_list)):
                index = training_data_num_list[training_file_index]
                data_index_and_quality.append([index, df_reading_quality[df_reading_quality["para_id"] == index]["prediction"].values[0]])
            training_data_index_and_quality_list.append(data_index_and_quality)

    print("完成数据基础处理。")

    # 获取reading数据的label。
    training_reading_labels = event_detection.get_data_label(training_reading_data)
    print("完成label标注。")

    # 计算每个样本点的density。（多进程优化）
    gaze_density_list = []
    gaze_density_args_list = []
    for file_index in range(len(training_reading_data)):
        args_list = []
        df_group_by_matrix_x = training_reading_data[file_index].groupby("matrix_x")
        for para_id, df_matrix_x in df_group_by_matrix_x:
            gaze_x = np.array(df_matrix_x["gaze_x"].values.tolist())
            gaze_y = np.array(df_matrix_x["gaze_y"].values.tolist())
            points = np.array([[gaze_x[i], gaze_y[i]] for i in range(len(gaze_x))], dtype=np.float32)
            distances = np.sqrt(np.sum((points[:, np.newaxis, :] - points[np.newaxis, :, :]) ** 2, axis=-1))
            for gaze_index in range(len(points)):
                args_list.append([distances, gaze_index])
        gaze_density_args_list.append(args_list)
    with Pool(configs.num_of_processes) as p:
        for file_index in range(len(training_reading_data)):
            result = p.starmap(compute_density, gaze_density_args_list[file_index])
            gaze_density_list.append(result)
    print("完成density计算。")

    # 根据labels，挑选出需要的training_reading和gaze_density。去掉那些label为-2, -1, 3的样本点。
    for file_index in range(len(training_reading_data)):
        combined_labels = np.concatenate(training_reading_labels[file_index])
        drop_rows = []
        for gaze_index in range(len(combined_labels)):
            if combined_labels[gaze_index] == -1 or combined_labels[gaze_index] == -2 or combined_labels[gaze_index] == 3:
                drop_rows.append(gaze_index)
        drop_rows = np.array(drop_rows)
        if len(drop_rows) > 0:
            training_reading_data[file_index] = training_reading_data[file_index].reset_index().drop(drop_rows)
            training_reading_data[file_index].reset_index(inplace=True, drop=True)
            gaze_density_list[file_index] = np.delete(np.array(gaze_density_list[file_index]), drop_rows)
        else:
            training_reading_data[file_index].reset_index(inplace=True, drop=True)
    print("完成label筛选。")

    # 去掉那些由于实验不当造成的过于离群的点。
    check_outlier_args_list = []
    for file_index in range(len(training_reading_data)):
        args_list = []
        cali_centroids = cali_centroid_list[file_index]
        H, mask = cv2.findHomography(cali_centroids, std_points_1d, cv2.RANSAC, 5.0)
        gaze_x = np.array(training_reading_data[file_index]["gaze_x"].values.tolist())
        gaze_y = np.array(training_reading_data[file_index]["gaze_y"].values.tolist())
        gaze_points = np.array([[gaze_x[i], gaze_y[i]] for i in range(len(gaze_x))], dtype=np.float32)
        transformed_gaze_points = cv2.perspectiveTransform(gaze_points.reshape(-1, 1, 2), H).reshape(-1, 2)
        bound_left = configs.left - configs.text_width * 1.5
        bound_right = configs.left + configs.col_num * configs.text_width + configs.text_width * 1.5
        bound_top = configs.top - configs.text_height * 1.5
        bound_bottom = configs.top + configs.row_num * configs.text_height + configs.text_height * 1.5
        for gaze_index in range(len(transformed_gaze_points)):
            args_list.append([transformed_gaze_points[gaze_index][0], transformed_gaze_points[gaze_index][1], bound_left, bound_right, bound_top, bound_bottom])
        check_outlier_args_list.append(args_list)
    check_outlier_list = []
    with Pool(configs.num_of_processes) as p:
        for file_index in range(len(check_outlier_args_list)):
            result = p.starmap(check_is_outlier, check_outlier_args_list[file_index])
            check_outlier_list.append(np.array(result))
    for file_index in range(len(training_reading_data)):
        gaze_x0 = training_reading_data[file_index]["gaze_x"].values.tolist()
        gaze_y0 = training_reading_data[file_index]["gaze_y"].values.tolist()
        training_reading_data[file_index] = training_reading_data[file_index][check_outlier_list[file_index] == 0]
        training_reading_data[file_index].reset_index(inplace=True, drop=True)
        gaze_density_list[file_index] = np.array(gaze_density_list[file_index])[check_outlier_list[file_index] == 0]
        gaze_x1 = training_reading_data[file_index]["gaze_x"].values.tolist()
        gaze_y1 = training_reading_data[file_index]["gaze_y"].values.tolist()

        # fig, ax = plt.subplots()
        # ax.set_xlim(0, 1920)
        # ax.set_ylim(1200, 0)
        # ax.scatter(gaze_x0, gaze_y0, s=1, c='r')
        # ax.scatter(gaze_x1, gaze_y1, s=1, c='b')
        # plt.show()
    print("完成离群点去除。")

    file_path = f"data/modified_gaze_data/{configs.round}/{configs.device}/"
    file_name_list = os.listdir(file_path)
    # for file_index in range(len(training_reading_data)):
        # log_path = f"output/alignment/{configs.round}/{configs.device}/{file_name_list[file_index]}/"
        # if not os.path.exists(log_path):
        #     os.makedirs(os.path.dirname(log_path))
        # log_file_name = f"{configs.test_str}.txt"
        # if configs.bool_log:
        #     with open(f"{log_path}{log_file_name}", "w") as log_file:
        #         log_file.write(f"file: {file_name_list[file_index]}\n"
        #                        f"time: {time.time()}\n"
        #                        f"file_index_and_quality: {training_data_index_and_quality_list[file_index]}\n"
        #                        f"coeff_semantic_before_output: {configs.coeff_semantic_before_output}\n"
        #                        f"coeff_gpt_for_non_structural: {configs.coeff_gpt_for_non_structural}\n"
        #                        f"coeff_gpt: {configs.coeff_gpt}\n"
        #                        f"coeff_structural: {configs.coeff_structural}\n"
        #                        f"first_row_text_penalty: {configs.first_row_text_penalty}\n"
        #                        f"empty_text_unit_penalty: {configs.empty_text_unit_penalty}\n"
        #                        f"punctuation_text_unit_penalty: {configs.punctuation_text_unit_penalty}\n"
        #                        f"text_unit_density_threshold_for_empty: {configs.text_unit_density_threshold_for_empty}\n"
        #                        f"far_from_text_unit_penalty: {configs.far_from_text_unit_penalty}\n"
        #                        f"dist_threshold_from_std: {configs.dist_threshold_from_std}\n")
        #         if configs.bool_blur_weight:
        #             log_file.write(f"gaussian_filter: {configs.gaussian_filter}\n")
        #         log_file.write("\n")
        #         em_solution(std_points_1d, training_reading_data[file_index], gaze_density_list[file_index], training_text_mapping[file_index], cali_centroid_list[file_index], file_name_list[file_index], log_file)
        # else:
        #     em_solution(std_points_1d, training_reading_data[file_index], gaze_density_list[file_index], training_text_mapping[file_index], cali_centroid_list[file_index], file_name_list[file_index])
        #     pass

    while True:
        file_name_input = input("please input file name: ")
        if file_name_input == "END":
            break
        elif file_name_input in file_name_list:
            file_index = file_name_list.index(file_name_input)
            manual_test(std_points_1d, training_reading_data[file_index], gaze_density_list[file_index], training_text_mapping[file_index], cali_centroid_list[file_index], file_name_list[file_index])

