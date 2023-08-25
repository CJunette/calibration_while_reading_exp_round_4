import numpy as np
import pandas as pd
import dtw
from matplotlib import pyplot as plt

import analyse_calibration_data
import read_files


def add_distance_to_data(df, x_offset, y_offset):
    gaze_x_list = df["gaze_x"].tolist()
    gaze_y_list = df["gaze_y"].tolist()
    gaze_x_list = [float(i) + x_offset for i in gaze_x_list]
    gaze_y_list = [float(i) + y_offset for i in gaze_y_list]
    df["gaze_x"] = gaze_x_list
    df["gaze_y"] = gaze_y_list


def compute_density_in_pull_test(reading_position_list, std_points_1d):
    reading_position_list = np.array(reading_position_list)
    std_points_1d = np.array(std_points_1d)
    distance_matrix = np.sqrt(np.sum((std_points_1d[:, np.newaxis, :] - reading_position_list[np.newaxis, :, :]) ** 2, axis=-1))
    std_density = [0 for _ in range(len(std_points_1d))]
    reading_pairing = [0 for _ in range(len(reading_position_list))]
    distance_threshold = 40
    for gaze_index in range(len(reading_position_list)):
        min_distance = np.min(distance_matrix[:, gaze_index])
        if min_distance < distance_threshold:
            min_distance_index = np.argmin(distance_matrix[:, gaze_index])
            std_density[min_distance_index] += 1
            reading_pairing[gaze_index] += 1
    # print(std_density)
    return std_density, reading_pairing


def compute_force_in_pull_test(std_pnt_density_list, reading_pairing_list, reading_pnt_list, std_density_list):
    force_list = []
    for gaze_index in range(len(reading_pnt_list)):
        if reading_pairing_list[gaze_index] > 0:
            continue
        effective_std_points = []
        for std_index in range(len(std_density_list)):
            if std_pnt_density_list[std_index] == 0:
                effective_std_points.append(std_density_list[std_index])
        # compute the Force vector between reading_pnt_list[gaze_index] and effective_std_points using np broadcast
        force_list_of_gaze_index = np.array(effective_std_points) - np.array(reading_pnt_list[gaze_index])
        for force in force_list_of_gaze_index:
            force_list.append(force)
    if len(force_list) == 0:
        avg_force = [0, 0]
    else:
        avg_force = np.mean(np.array(force_list), axis=0) * 0.25
    # print(avg_force)
    for gaze_index in range(len(reading_pnt_list)):
        reading_pnt_list[gaze_index] += avg_force
    return avg_force


def iteration_loop_in_pull_test(reading_position_list, std_points_1d, cali_points_1d, max_iteration=50):
    def save_reading_position_of_iteration(iteration_history_of_reading):
        # 深拷贝复制reading_position_list
        reading_position_list_of_iteration = []
        for reading_position in reading_position_list:
            reading_position_list_of_iteration.append(reading_position.copy())
        iteration_history_of_reading.append(reading_position_list_of_iteration)

    def compute_bias_of_calibration(force):
        new_cali_points_1d = []
        for cali_index in range(len(cali_points_1d)):
            new_cali_points_1d.append(np.array(cali_points_1d[cali_index]) + np.array(force))

        bias = np.array(new_cali_points_1d) - np.array(std_points_1d)
        bias = np.mean(bias, axis=0)
        bias = np.linalg.norm(bias)
        return bias

    loop_count = 0
    iteration_history_of_reading = []
    save_reading_position_of_iteration(iteration_history_of_reading)
    compute_bias_of_calibration([0, 0])
    force_history = [[0, 0]]
    bias_history = []
    bias = compute_bias_of_calibration([0, 0])
    bias_history.append(bias)

    while True:
        if loop_count == 8:
            print()
        res = dtw.dtw(x=reading_position_list, y=std_points_1d)
        std_pnt_density_list, reading_pairing_list = compute_density_in_pull_test(reading_position_list, std_points_1d)
        gaze_color_list = []
        for gaze_index in range(len(reading_pairing_list)):
            if reading_pairing_list[gaze_index] == 0:
                gaze_color_list.append('r')
            else:
                gaze_color_list.append('b')
        std_color_list = []
        for std_index in range(len(std_pnt_density_list)):
            if std_pnt_density_list[std_index] == 0:
                std_color_list.append('k')
            else:
                std_color_list.append('g')

        # visualize.
        # fig, ax = plt.subplots(figsize=(16, 10))
        # ax.set_xlim(0, 1920)
        # ax.set_ylim(1200, 0)
        # ax.set_aspect("equal")
        # ax.scatter(np.array(reading_position_list)[:, 0], np.array(reading_position_list)[:, 1], c=gaze_color_list, marker='o')
        # ax.scatter(np.array(std_points_1d)[:, 0], np.array(std_points_1d)[:, 1], c=std_color_list, marker='o')
        # for step_index in range(len(res.index1)):
        #     reading_position = reading_position_list[res.index1[step_index]]
        #     std_position = std_points_1d[res.index2[step_index]]
        #     # print(reading_position, std_position)
        #     plt.plot([reading_position[0], std_position[0]], [reading_position[1], std_position[1]], c='#DDDDDD', linewidth=0.5)
        #     pass
        #
        # plt.show()

        loop_count += 1
        save_reading_position_of_iteration(iteration_history_of_reading)

        force = compute_force_in_pull_test(std_pnt_density_list, reading_pairing_list, reading_position_list, std_points_1d)
        force_history.append(force)

        avg_movement = np.mean(np.array(iteration_history_of_reading[-1]) - np.array(iteration_history_of_reading[0]), axis=0)
        bias = compute_bias_of_calibration(avg_movement)
        bias_history.append(bias)

        distance_between_two_step = None
        if loop_count > 1:
            distance_between_two_step = np.linalg.norm(np.mean(np.array(iteration_history_of_reading[-1]) - np.array(iteration_history_of_reading[-2])))

        # 设置终止条件。
        if np.linalg.norm(force) < 5:
            print("break, force < limit, force: ", force)
            break
        elif loop_count > max_iteration:
            print("break, iteration > max, iteration: ", loop_count)
            break
        # elif np.isnan(force.any()):
        #     print("break, all points are paired")

    min_force_index = np.argmin(np.linalg.norm(np.array(force_history[1:]), axis=1))
    print(np.linalg.norm(force_history[min_force_index + 1]), bias_history[min_force_index + 1])
    min_force_reading = iteration_history_of_reading[min_force_index + 1]

    # min_bias_index = np.argmin(bias_history)
    # print(bias_history[min_bias_index])
    # min_bias_reading = iteration_history_of_reading[min_bias_index]

    return min_force_reading


def pull_test():
    std_points = analyse_calibration_data.create_standard_calibration_points()
    reading_data_list = read_files.read_reading_data()
    cali_data_list = read_files.read_calibration_data()
    text_mapping_list = read_files.read_all_modified_reading_text_mapping()

    # 为reading数据和calibration数据添加一个偏移。
    for file_index in range(len(reading_data_list)):
        x_offset = 400
        y_offset = 400
        for iteration_index in range(len(reading_data_list[file_index])):
            reading_df = reading_data_list[file_index][iteration_index]
            reading_df = reading_df[reading_df["gaze_x"] != "failed"]
            add_distance_to_data(reading_df, x_offset, y_offset)
            reading_data_list[file_index][iteration_index] = reading_df
        cali_df = cali_data_list[file_index]
        cali_df = cali_df[cali_df["gaze_x"] != "failed"]
        add_distance_to_data(cali_df, x_offset, y_offset)
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

    for file_index in range(len(combined_reading_data_list)):
        reading_df_group_by_matrix_x = combined_reading_data_list[file_index].groupby("matrix_x")
        for para_id, reading_df_matrix_x in reading_df_group_by_matrix_x:
            # if file_index != 0 or para_id != 4:
            #     continue
            df_text_mapping = text_mapping_list[file_index]
            df_text_mapping = df_text_mapping[df_text_mapping["para_id"] == para_id]
            row_list = df_text_mapping["row"].unique().tolist()
            std_points_1d = []
            cali_points_1d = []
            for row_id in row_list:
                col_list = df_text_mapping[df_text_mapping["row"] == row_id]["col"].unique().tolist()
                for col_id in col_list:
                    std_points_1d.append(std_points[row_id][col_id])
                    cali_points_1d.append(cali_centroid_list[file_index][row_id][col_id])
            reading_position_list = reading_df_matrix_x[["gaze_x", "gaze_y"]].values.tolist()

            reading_position_list = iteration_loop_in_pull_test(reading_position_list, std_points_1d, cali_points_1d)

            # visualize result.
            fig, ax = plt.subplots(figsize=(16, 10))
            ax.set_xlim(0, 1920)
            ax.set_ylim(1200, 0)
            ax.set_aspect("equal")
            ax.scatter(np.array(reading_position_list)[:, 0], np.array(reading_position_list)[:, 1], c='b', marker='o')
            ax.scatter(np.array(std_points_1d)[:, 0], np.array(std_points_1d)[:, 1], c='k', marker='o')

            plt.show()

