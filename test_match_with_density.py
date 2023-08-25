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
import read_files
import test_pull_gaze_trace


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


def min_match_with_weighted_dist(responsibilities, gaze_points, std_points_1d):
    pairs_indices_of_std = [[] for _ in range(len(responsibilities))]
    gaze_pair_points = [[0, 0] for _ in range(len(responsibilities))]
    std_pair_points = [std_points_1d[i] for i in range(len(std_points_1d))]
    # 对每个gaze point，找到对应的“距离”最近的std point。
    for gaze_index in range(len(responsibilities[0])):
        min_std_index = np.argmin(responsibilities[:, gaze_index])
        pairs_indices_of_std[min_std_index].append(gaze_index)
    # 再遍历每个std point，如果有多个gaze point，就取平均；
    # 如果没有，则看这个std point对应的最小距离是否小于configs.empty_text_unit_punishment，如果是，则取最小的gaze point，如果不是，则跳过这个std point（对应空格、标点或什么都没有）。
    for std_index in range(len(pairs_indices_of_std)):
        if len(pairs_indices_of_std[std_index]) > 0:
            mean_point = np.array([0., 0.])
            sub_std_index = 0
            while sub_std_index < len(pairs_indices_of_std[std_index]):
                dist = responsibilities[std_index][pairs_indices_of_std[std_index][sub_std_index]]
                if dist > abs(configs.empty_text_unit_penalty):
                    pairs_indices_of_std[std_index].pop(sub_std_index)
                else:
                    mean_point += gaze_points[pairs_indices_of_std[std_index][sub_std_index]]
                    sub_std_index += 1
            if len(pairs_indices_of_std[std_index]) == 0:
                continue
            else:
                mean_point /= len(pairs_indices_of_std[std_index])
                gaze_pair_points[std_index] = mean_point
        elif len(pairs_indices_of_std[std_index]) == 0 and responsibilities[std_index].min() < abs(configs.empty_text_unit_penalty):
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


def compute_absolute_distance_matrix(gaze_data, std_data):
    std_data_matrix = std_data[:, np.newaxis, :]
    gaze_data_matrix = gaze_data[np.newaxis, :, :]
    distance_vector_matrix = std_data_matrix - gaze_data_matrix
    distance_matrix = np.linalg.norm(distance_vector_matrix, axis=-1)
    return distance_matrix


def compute_structural_distance_matrix(gaze_data, std_data, gaze_density, structural_weight):
    def add_weight_to_blank_weight_list(df, key):
        blank_weight_list = [configs.empty_text_unit_penalty for _ in range(configs.row_num * configs.col_num)]

        for text_unit_index in range(df.shape[0]):
            token = df["word"].iloc[text_unit_index]
            if token.strip() == "":
                continue

            row = df["row"].iloc[text_unit_index]
            col = df["col"].iloc[text_unit_index]
            weight = df[key].iloc[text_unit_index]
            if token in configs.punctuation_list:
                weight = configs.punctuation_text_unit_penalty
            blank_weight_list[row * configs.col_num + col] = weight

        return blank_weight_list
    # 首先根据不同的para_id，把weight分配好。
    para_id_list = structural_weight["para_id"].unique()
    para_id_list = np.sort(para_id_list)

    horizontal_weight_matrix = np.array([[configs.empty_text_unit_penalty for _ in range(len(gaze_data))] for _ in range(len(std_data))])
    vertical_weight_matrix = np.array([[configs.empty_text_unit_penalty for _ in range(len(gaze_data))] for _ in range(len(std_data))])
    first_row_weight_matrix = np.array([[configs.empty_text_unit_penalty for _ in range(len(gaze_data))] for _ in range(len(std_data))])

    for para_id in para_id_list:
        df = structural_weight[structural_weight["para_id"] == para_id]
        horizontal_weight_list = add_weight_to_blank_weight_list(df, "horizontal_edge_weight")
        vertical_weight_list = add_weight_to_blank_weight_list(df, "vertical_edge_weight")
        first_row_weight_list = add_weight_to_blank_weight_list(df, "first_row_weight")

        for gaze_index in range(len(gaze_data)):
            horizontal_weight_matrix[:, gaze_index] = horizontal_weight_list
            vertical_weight_matrix[:, gaze_index] = vertical_weight_list
            first_row_weight_matrix[:, gaze_index] = first_row_weight_list

    structural_weight_matrix = 0.5 * horizontal_weight_matrix + 0.2 * vertical_weight_matrix + 0.3 * first_row_weight_matrix
    gaze_density_exp = 1 + 4 / (1 + np.exp(-1 * (gaze_density - 5)))
    gaze_density_matrix = gaze_density_exp[np.newaxis, :]
    structural_distance_matrix = np.abs(structural_weight_matrix - gaze_density_matrix)
    return structural_distance_matrix


def compute_distance(gaze_data, std_data, structural_distance_matrix):
    absolute_distance_matrix = compute_absolute_distance_matrix(gaze_data, std_data)
    hybrid_distance_matrix = 0.5 * absolute_distance_matrix + structural_distance_matrix
    return hybrid_distance_matrix


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
    perspective_x = H[2, 0]
    perspective_y = H[2, 1]

    penalty = 0
    if abs(theta) > 20:
        penalty += configs.H_rotation_penalty
        # print(f"rotation penalty, theta: {theta}")
    if (not 0.8 <= scale_x <= 1.2) or (not 0.8 <= scale_y <= 1.2):
        penalty += configs.H_scale_penalty
        # print(f"scale penalty, scale_x: {scale_x}, scale_y: {scale_y}")
    if abs(shear_x) > 0.1 or abs(shear_y) > 0.1:
        penalty += configs.H_shear_penalty
        # print(f"shear penalty, shear_x: {shear_x}, shear_y: {shear_y}")
    if abs(perspective_x) > 0.2 or abs(perspective_y) > 0.2:
        penalty += configs.H_projection_penalty
        # print(f"projection penalty, perspective_x: {perspective_x}, perspective_y: {perspective_y}")

    # 假设一个以(0, 0)和(1920, 1200)为对角点的矩形rect_1，现在希望比较经过H变换后的四边形rect_2和rect_1的面积之比。如果面积比过大，则施加惩罚。
    rect_1 = np.array([[0, 0], [1920, 0], [1920, 1200], [0, 1200]], dtype=np.float32)
    rect_2 = cv2.perspectiveTransform(rect_1.reshape(-1, 1, 2), H).reshape(-1, 2)
    space_1 = cv2.contourArea(rect_1)
    space_2 = cv2.contourArea(rect_2)
    space_ratio = space_2 / space_1
    if not 0.9 < space_ratio < 1.1:
        penalty += configs.H_space_ratio_penalty

    return penalty


def compute_unfitness_in_generic(H, gaze_points, std_points_1d, structural_distance_matrix):
    transformed_src_pts = cv2.perspectiveTransform(gaze_points.reshape(-1, 1, 2), H).reshape(-1, 2)
    absolute_distance_matrix = compute_absolute_distance_matrix(transformed_src_pts, std_points_1d)
    # dist_from_hybrid指每个gaze点到最近的std点的距离之和。
    dist_from_hybrid_list = [0 for _ in range(len(gaze_points))]
    for gaze_index in range(len(absolute_distance_matrix[0])):
        min_dist_to_std_index = np.argmin(absolute_distance_matrix[:, gaze_index])
        min_dist_to_std = absolute_distance_matrix[min_dist_to_std_index, gaze_index]
        # 对于离最近的std特别远的点，给一个额外的惩罚。
        # if min_dist_to_std > 150:
        #     min_dist_to_std *= 10
        structural_dist_to_std = structural_distance_matrix[min_dist_to_std_index, gaze_index]
        dist_from_hybrid_list[gaze_index] += min_dist_to_std + structural_dist_to_std
        # dist_from_hybrid_list[gaze_index] += min_dist_to_std
        # if gaze_index == 330:
        #     print(transformed_src_pts[gaze_index], min_dist_to_std_index, min_dist_to_std, structural_dist_to_std)

    dist_from_hybrid = np.mean(dist_from_hybrid_list)
    # dist_from_H指H过大的变化量带来的惩罚。
    dist_from_H = compute_homography_penalty(H)
    # dist_from_H = 0
    # print(dist_from_hybrid, dist_from_H, space_ratio)
    return dist_from_hybrid + dist_from_H


def generic_algorithm_to_find_best_homography(src_pts, dst_pts, gaze_points, std_points_1d, structural_distance_matrix, population_size=100, generations=5):
    def crossover(parent_1, parent_2):
        child = (parent_1 + parent_2) / 2
        return child

    def mutate(H):
        mutation_matrix = np.array([
            [np.random.uniform(-0.1, 0.1), np.random.uniform(-0.05, 0.05), np.random.uniform(-configs.text_width*4, configs.text_width*4)],
            [np.random.uniform(-0.05, 0.05), np.random.uniform(-0.1, 0.1), np.random.uniform(-configs.text_height*4, configs.text_height*4)],
            [np.random.uniform(-0.01, 0.01), np.random.uniform(-0.01, 0.01), 0]])
        H += mutation_matrix
        return H

    def select(population, unfitness):
        # unfitness = np.array(unfitness)
        # unfitness = 1 / (unfitness + 1e-10)
        # unfitness /= unfitness.sum()
        # return np.random.choice(population, size=population_size, replace=True, p=unfitness)
        sorted_indices = np.argsort(unfitness)
        sorted_unfitness = np.sort(unfitness)
        selected_population = [population[i] for i in sorted_indices[:population_size // 2]]
        return selected_population, sorted_unfitness

    population = []
    for population_index in range(population_size):
        random_set = np.random.choice(len(src_pts), size=int(len(src_pts) / 5))
        H, _ = cv2.findHomography(src_pts[random_set].reshape(-1, 1, 2), dst_pts[random_set].reshape(-1, 1, 2))
        population.append(H)

    with Pool(16) as p:
        for generation in range(generations):
            print(f"generation: {generation}")
            args_list = []
            unfitness = []
            for H in population:
                args_list.append((H, gaze_points, std_points_1d, structural_distance_matrix))
                # unfitness.append(compute_unfitness_in_generic(H, gaze_points, std_points_1d, structural_distance_matrix))
            unfitness = p.starmap(compute_unfitness_in_generic, args_list)

            selected_population, sorted_unfitness = select(population, unfitness)
            # for i in range(len(selected_population)):
            #     if selected_population[i][1][2] > 64:
            #         print(selected_population[i], sorted_unfitness[i])
            best_H = selected_population[0]
            print(f"best H: {selected_population[0]}, best unfitness: {sorted_unfitness[0]}")
            transformed_gaze_points = cv2.perspectiveTransform(gaze_points.reshape(-1, 1, 2), best_H).reshape(-1, 2)

            new_population = []
            while len(new_population) < population_size // 2:
                parent1, parent2 = random.choices(selected_population, k=2)
                if np.isnan(parent1).any() or np.isnan(parent2).any() or np.isinf(parent1).any() or np.isinf(parent2).any():
                    continue
                child_1 = crossover(parent1, parent2)
                new_population.append(child_1)

            # 仅对40%做变异。
            for i in range(len(new_population)):
                if np.random.rand() < 0.6:
                    new_population[i] = mutate(new_population[i])
            population = selected_population + new_population


            # fig, ax = plt.subplots(figsize=(12, 8))
            # ax.set_xlim(0, 1920)
            # ax.set_ylim(1200, 0)
            # ax.set_aspect("equal")
            # ax.scatter(std_points_1d[:, 0], std_points_1d[:, 1], label='std point', color='black')
            # ax.scatter(gaze_points[:, 0], gaze_points[:, 1], label='original gaze', color='blue', alpha=0.8)
            # ax.scatter(transformed_gaze_points[:, 0], transformed_gaze_points[:, 1], label='transformed gaze', color='red', alpha=0.5)
            # for pair_index in range(len(src_pts)):
            #     plt.plot([src_pts[pair_index][0], dst_pts[pair_index][0]], [src_pts[pair_index][1], dst_pts[pair_index][1]], color='#DDDDDD', alpha=0.5)
            # plt.show()

    final_unfitness = []
    for H in population:
        final_unfitness.append(compute_unfitness_in_generic(H, gaze_points, std_points_1d, structural_distance_matrix))
    best_H = population[np.argmin(final_unfitness)]
    # 这里的best_H是只有最后一步的H，而不是最好的H。
    return best_H, min(final_unfitness)


def em_solution(std_points_1d, df_gaze_data, gaze_density, text_mapping, max_iteration=50):
    H = np.eye(3)
    cov_init = np.array([[12, 0], [0, 12]]) * 2
    gaussian_std_points = [{'mean': std_points_1d[i], 'cov': cov_init} for i in range(len(std_points_1d))]

    df_reading_data_group_by_matrix_x = df_gaze_data.groupby("matrix_x")
    # for matrix_x, df_reading_data_matrix_x in df_reading_data_group_by_matrix_x:

    responsibilities = np.zeros((len(std_points_1d), df_gaze_data.shape[0]))
    gaze_x = df_gaze_data["gaze_x"].tolist()
    gaze_y = df_gaze_data["gaze_y"].tolist()
    gaze_points = np.array([[gaze_x[i], gaze_y[i]] for i in range(len(gaze_x))])
    original_gaze_points = gaze_points.copy()

    # structural_distance_matrix不随位置变化而变化，因此只计算一次。
    structural_distance_matrix = compute_structural_distance_matrix(gaze_points, std_points_1d, np.array(gaze_density), text_mapping)

    for iteration in range(max_iteration):
        print(f"iteration: {iteration}")
        # E-step: Assign each reading to the Gaussian in std under current H with highest PDF.
        gaze_points = cv2.perspectiveTransform(gaze_points.reshape(-1, 1, 2), H).reshape(-1, 2)

        responsibilities = compute_distance(gaze_points, std_points_1d, structural_distance_matrix)

        # args_list = []
        # for std_point_index, gaussian in enumerate(gaussian_std_points):
        #     for gaze_index in range(len(gaze_points)):
        #         args_list.append((gaze_points[gaze_index], gaussian['mean'], gaussian['cov']))
        # with Pool(16) as pool:
        #     results = pool.starmap(compute_distance_simple, args_list)
        # results = np.array(results).reshape(len(gaussian_std_points), len(gaze_points))
        # responsibilities = results

        # # linear_sum_assignment的匹配效果太差，不如自己写。
        # # row_ind, col_ind = linear_sum_assignment(responsibilities)
        # row_ind, col_ind = min_match(responsibilities)
        #
        # # M-step: Compute the H and update Gaussian parameters based on the assignments.
        # reading_pair_points = []
        # std_pair_points = []
        # for pair_index in range(len(row_ind)):
        #     reading_pair_points.append(gaze_points[col_ind[pair_index]])
        #     std_pair_points.append(std_points_1d[row_ind[pair_index]])
        # reading_pair_points = np.array(reading_pair_points)
        # std_pair_points = np.array(std_pair_points)

        reading_pair_points, std_pair_points = min_match_with_weighted_dist(responsibilities, gaze_points, std_points_1d)
        # 使用遗传算法来找到最优的H。
        H, least_unfitness = generic_algorithm_to_find_best_homography(reading_pair_points, std_pair_points, gaze_points, std_points_1d, structural_distance_matrix)
        # H, mask = cv2.findHomography(reading_pair_points.reshape(-1, 1, 2), std_pair_points.reshape(-1, 1, 2), cv2.LMEDS, 10.0)
        transformed_gaze_points = cv2.perspectiveTransform(gaze_points.reshape(-1, 1, 2), H).reshape(-1, 2)
        print(H, least_unfitness)

        # for pair_index in range(len(reading_pair_points)):
        #     print(reading_pair_points[pair_index], std_pair_points[pair_index])

        # for std_point_index, gaussian in enumerate(gaussian_std_points):
        #     assigned_points = transformed_gaze_points[col_ind[row_ind == std_point_index]]
        #     if len(assigned_points) > 0:
        #         gaussian["mean"] = np.mean(assigned_points, axis=0)
        #         if len(assigned_points) > 1:
        #             gaussian["cov"] = np.cov(assigned_points, rowvar=False)

        fig, ax = plt.subplots(figsize=(12, 8))
        ax.set_xlim(0, 1920)
        ax.set_ylim(1200, 0)
        ax.set_aspect("equal")
        ax.scatter(std_points_1d[:, 0], std_points_1d[:, 1], label='std point', color='black')
        ax.scatter(gaze_points[:, 0], gaze_points[:, 1], label='original gaze', color='green', alpha=0.8)
        ax.scatter(original_gaze_points[:, 0], original_gaze_points[:, 1], label='original gaze', color='blue', alpha=0.75)
        ax.scatter(transformed_gaze_points[:, 0], transformed_gaze_points[:, 1], label='transformed gaze', color='red', alpha=0.5)
        for pair_index in range(len(reading_pair_points)):
            plt.plot([reading_pair_points[pair_index][0], std_pair_points[pair_index][0]], [reading_pair_points[pair_index][1], std_pair_points[pair_index][1]], color='#DDDDDD', alpha=0.5)
        plt.show()


def match_with_density():
    std_points = analyse_calibration_data.create_standard_calibration_points()
    std_points_1d = np.array(std_points).reshape(-1, 2)
    std_point_x_min = std_points_1d[:, 0].min() - configs.text_width / 2
    std_point_x_max = std_points_1d[:, 0].max() + configs.text_width / 2
    std_point_y_min = std_points_1d[:, 1].min() - configs.text_height / 2
    std_point_y_max = std_points_1d[:, 1].max() + configs.text_height / 2
    std_point_box = (std_point_x_min, std_point_x_max, std_point_y_min, std_point_y_max)

    reading_data_list = read_files.read_reading_data()
    cali_data_list = read_files.read_calibration_data()
    text_mapping_list = read_files.read_all_modified_reading_text_mapping()
    sorted_text_mapping = read_files.read_sorted_reading_text_mapping()

    for file_index in range(len(text_mapping_list)):
        text_mapping_list[file_index]["horizontal_edge_weight"] = sorted_text_mapping["horizontal_edge_weight"]
        text_mapping_list[file_index]["vertical_edge_weight"] = sorted_text_mapping["vertical_edge_weight"]
        text_mapping_list[file_index]["first_row_weight"] = sorted_text_mapping["first_row_weight"]

    # 为reading数据和calibration数据添加一个偏移。
    for file_index in range(len(reading_data_list)):
        x_offset = 0
        y_offset = -64
        for iteration_index in range(len(reading_data_list[file_index])):
            reading_df = reading_data_list[file_index][iteration_index]
            reading_df = reading_df[reading_df["gaze_x"] != "failed"]
            test_pull_gaze_trace.add_distance_to_data(reading_df, x_offset, y_offset)
            reading_data_list[file_index][iteration_index] = reading_df
        cali_df = cali_data_list[file_index]
        cali_df = cali_df[cali_df["gaze_x"] != "failed"]
        test_pull_gaze_trace.add_distance_to_data(cali_df, x_offset, y_offset)
        cali_data_list[file_index] = cali_df

    # 计算calibration数据的centroid。
    cali_centroid_list = []
    for file_index in range(len(cali_data_list)):
        cali_centroid_list.append(analyse_calibration_data.compute_centroids(cali_data_list[file_index]))

    # 将reading数据按照matrix_x进行分组。
    combined_reading_data_list = []
    for file_index in range(len(reading_data_list)):
        df_list = []
        for iteration_index in range(len(reading_data_list[file_index])):
            df_group_by_matrix_x = reading_data_list[file_index][iteration_index].groupby("matrix_x")
            for para_id, df_matrix_x in df_group_by_matrix_x:
                df_list.append(df_matrix_x)
        df_list.sort(key=lambda x: x["matrix_x"].iloc[0])
        combined_reading_data_list.append(pd.concat(df_list, ignore_index=True))

    # 截取部分reading data作为训练集。
    training_data_num_list = [2]
    training_reading_data = []
    for file_index in range(len(combined_reading_data_list)):
        df = combined_reading_data_list[file_index]
        training_reading_data.append(df[df["matrix_x"].isin(training_data_num_list)])

    # 截取部分text_mapping作为训练集。
    training_text_mapping = []
    for file_index in range(len(combined_reading_data_list)):
        df = text_mapping_list[file_index]
        training_text_mapping.append(df[df["para_id"].isin(training_data_num_list)])


    # 计算每个样本点的density。
    gaze_density_list = []
    for file_index in range(len(training_reading_data)):
        gaze_x = np.array(training_reading_data[file_index]["gaze_x"].values.tolist())
        gaze_y = np.array(training_reading_data[file_index]["gaze_y"].values.tolist())
        points = np.array([[gaze_x[i], gaze_y[i]] for i in range(len(gaze_x))], dtype=np.float32)
        distances = np.sqrt(np.sum((points[:, np.newaxis, :] - points[np.newaxis, :, :]) ** 2, axis=-1))

        density_list = [0 for _ in range(len(distances))]
        for gaze_index in range(len(distances)):
            for probe_index in range(gaze_index + 1, len(distances[gaze_index])):
                if distances[gaze_index][probe_index] < configs.reading_density_distance_threshold:
                    density_list[gaze_index] += 1
                else:
                    break
            for probe_index in range(gaze_index - 1, -1, -1):
                if distances[gaze_index][probe_index] < configs.reading_density_distance_threshold:
                    density_list[gaze_index] += 1
                else:
                    break
        gaze_density_list.append(density_list)

    # 找到每个样本的外包矩形，对齐两个矩形的左上角，然后将reading数据平移。然后通过x_min和y_min计算相对位置。 # TODO debug完了取消注释。
    # for file_index in range(len(training_reading_data)):
    for file_index in range(1):
        print(file_index)
        gaze_x = np.array(training_reading_data[file_index]["gaze_x"].values.tolist())
        gaze_y = np.array(training_reading_data[file_index]["gaze_y"].values.tolist())
        para_id = np.array(training_reading_data[file_index]["matrix_x"].values.tolist())
        points = np.array([[gaze_x[i], gaze_y[i]] for i in range(len(gaze_x))], dtype=np.float32)

        # 基于统计的方法。
        # Q1_x = np.percentile(gaze_x, 25)
        # Q3_x = np.percentile(gaze_x, 75)
        # IQR_x = Q3_x - Q1_x
        #
        # Q1_y = np.percentile(gaze_y, 25)
        # Q3_y = np.percentile(gaze_y, 75)
        # IQR_y = Q3_y - Q1_y
        #
        # bound_coeff_x = 1
        # bound_coeff_y = 0.5
        # # 使用IQR确定离群点的界限
        # lower_bound_x = Q1_x - bound_coeff_x * IQR_x
        # upper_bound_x = Q3_x + bound_coeff_x * IQR_x
        # lower_bound_y = Q1_y - 0.25 * IQR_y
        # upper_bound_y = Q3_y + 0.75 * IQR_y
        #
        # # 过滤离群点
        # filtered_data = points[(points[:, 0] >= lower_bound_x) & (points[:, 0] <= upper_bound_x) & (points[:, 1] >= lower_bound_y) & (points[:, 1] <= upper_bound_y)]
        # x_min = filtered_data[:, 0].min()
        # x_max = filtered_data[:, 0].max()
        # y_min = filtered_data[:, 1].min()
        # y_max = filtered_data[:, 1].max()

        # # 基于IsolationForest的方法。
        # clf = IsolationForest(contamination=0.15, random_state=42)
        # outliers = clf.fit_predict(points)
        #
        # # 将正常点和离群点分开
        # filtered_data = points[outliers == 1]
        # left_out_data = points[outliers == -1]
        #
        # x_min = filtered_data[:, 0].min()
        # x_max = filtered_data[:, 0].max()
        # y_min = filtered_data[:, 1].min()
        # y_max = filtered_data[:, 1].max()

        # 基于领域的方法。
        sampled_data = np.array(random.sample(points.tolist(), int(len(points) * 0.3)))
        # 为采样的数据计算距离
        distances_sampled = distance.cdist(sampled_data, sampled_data, 'euclidean')
        # 对于每个点，获取其到其他点的距离，并取最近的k个点的平均距离
        avg_distances_sampled = np.sort(distances_sampled)[:, 1:25].mean(axis=1)
        # 使用距离的75th百分位数作为阈值
        threshold_sampled = np.percentile(avg_distances_sampled, 65)
        # 标识离群点
        outliers_sampled = avg_distances_sampled > threshold_sampled
        # 将正常点和离群点分开
        filtered_data = sampled_data[~outliers_sampled]
        left_out_data = sampled_data[outliers_sampled]
        x_min = filtered_data[:, 0].min()
        x_max = filtered_data[:, 0].max()
        y_min = filtered_data[:, 1].min()
        y_max = filtered_data[:, 1].max()

        # 计算以x_min，y_min为原点的相对密度。
        # relative_gaze_x_list = []
        # relative_gaze_y_list = []
        # for gaze_index in range(training_reading_data[file_index].shape[0]):
        #     gaze_x = training_reading_data[file_index]["gaze_x"]
        #     gaze_y = training_reading_data[file_index]["gaze_y"]
        #     relative_gaze_x = (gaze_x - x_min) / (x_max - x_min)
        #     relative_gaze_y = (gaze_y - y_min) / (y_max - y_min)
        #     relative_gaze_x_list.append(relative_gaze_x)
        #     relative_gaze_y_list.append(relative_gaze_y)
        # training_reading_data[file_index]["relative_gaze_x"] = relative_gaze_x_list
        # training_reading_data[file_index]["relative_gaze_y"] = relative_gaze_y_list

        # fig, ax = plt.subplots()
        # ax.set_xlim(0, 1920)
        # ax.set_ylim(1200, 0)
        # ax.scatter(gaze_x, gaze_y, s=1, c='k')
        # rect = plt.Rectangle((x_min, y_min), x_max - x_min, y_max - y_min, fill=False, edgecolor='r', linewidth=1)
        # ax.add_patch(rect)
        # plt.show()

        # 对齐两个矩形的左上角。
        # move_between_boxes = [std_point_box[0] - x_min, std_point_box[2] - y_min]
        # print(move_between_boxes)
        #
        # for gaze_index in range(training_reading_data[file_index].shape[0]):
        #     training_reading_data[file_index]["gaze_x"].iloc[gaze_index] += move_between_boxes[0]
        #     training_reading_data[file_index]["gaze_y"].iloc[gaze_index] += move_between_boxes[1]

    for file_index in range(len(training_reading_data)):
        em_solution(std_points_1d, training_reading_data[file_index], gaze_density_list[file_index], training_text_mapping[file_index])

