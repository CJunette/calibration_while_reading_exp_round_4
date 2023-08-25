import math
import os
from multiprocessing import Pool

import cv2
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt, patches

import analyse_calibration_data
import configs
import read_files


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


def get_point_density_single_pool(reading_df, gaze_index):
    # print(gaze_index)
    density = 0
    gaze_x = reading_df.iloc[gaze_index]["gaze_x"]
    gaze_y = reading_df.iloc[gaze_index]["gaze_y"]
    for probe_index in range(gaze_index - 1, -1, -1):
        distance = np.linalg.norm([gaze_x - reading_df.iloc[probe_index]["gaze_x"], gaze_y - reading_df.iloc[probe_index]["gaze_y"]])
        if distance <= configs.reading_density_distance_threshold:
            density += 1
        else:
            break
    for probe_index in range(gaze_index + 1, reading_df.shape[0]):
        distance = np.linalg.norm([gaze_x - reading_df.iloc[probe_index]["gaze_x"], gaze_y - reading_df.iloc[probe_index]["gaze_y"]])
        if distance <= configs.reading_density_distance_threshold:
            density += 1
        else:
            break
    return density


def render_point_density_hist():
    file_path = f"data/modified_gaze_data/{configs.round}/{configs.device}/"
    file_list = os.listdir(file_path)

    for file_index in range(len(file_list)):
        reading_file_path = f"{file_path}/{file_list[file_index]}/reading/"
        reading_file_list = os.listdir(reading_file_path)

        # for debug.
        # density_list = []
        # for reading_file_index in range(len(reading_file_list)):
        #     reading_df = pd.read_csv(f"{reading_file_path}{reading_file_list[reading_file_index]}", encoding="utf-8_sig")
        #     for gaze_index in range(reading_df.shape[0]):
        #         density = 0
        #         gaze_x = reading_df.iloc[gaze_index]["gaze_x"]
        #         gaze_y = reading_df.iloc[gaze_index]["gaze_y"]
        #         for probe_index in range(gaze_index - 1, -1, -1):
        #             distance = np.linalg.norm([gaze_x - reading_df.iloc[probe_index]["gaze_x"], gaze_y - reading_df.iloc[probe_index]["gaze_y"]])
        #             if distance <= configs.reading_density_distance_threshold:
        #                 density += 1
        #             else:
        #                 break
        #         for probe_index in range(gaze_index + 1, reading_df.shape[0]):
        #             distance = np.linalg.norm([gaze_x - reading_df.iloc[probe_index]["gaze_x"], gaze_y - reading_df.iloc[probe_index]["gaze_y"]])
        #             if distance <= configs.reading_density_distance_threshold:
        #                 density += 1
        #             else:
        #                 break
        #         density_list.append(density)

        args_list = []
        for reading_file_index in range(len(reading_file_list)):
            reading_df = pd.read_csv(f"{reading_file_path}{reading_file_list[reading_file_index]}", encoding="utf-8_sig")
            for gaze_index in range(reading_df.shape[0]):
                args_list.append((reading_df, gaze_index))

        with Pool(16) as p:
            density_list = p.starmap(get_point_density_single_pool, args_list)

        # with open(f"temp.txt", "a") as f:
        #     f.write(f"{density_list}\n")
        #     f.write("-----\n")

        plt.cla()
        # 设置图片大小，设置dpi为100，图像大小为1920*1200
        plt.figure(figsize=(1920 / 100, 1200 / 100), dpi=100)
        # plt.xlim(0, 55)
        # plt.ylim(0, 850)
        plt.hist(density_list, bins=100)
        # 在右上角显示density_list的长度
        plt.text(0.95, 0.95, f"total point num: {len(density_list)}", horizontalalignment='right', verticalalignment='top', transform=plt.gca().transAxes)
        # 保存图片
        file_name = file_list[file_index]
        save_path = f"image/point_density/{configs.round}/{file_name}.png"
        # 如果没有路径则创建。
        if not os.path.exists(os.path.dirname(save_path)):
            os.makedirs(os.path.dirname(save_path))
        # plt.savefig(save_path)
        plt.show()


def match_manual_weight_and_gaze_density():
    '''
    字的颜色越红，代表此处gaze density越大；底色越蓝，达标此处manual weight越大。
    :return:
    '''
    def visual_density(ax, df_actual_density, df_manual_density, df_reading):
        actual_density_list = df_actual_density["text_unit_density"].tolist()
        max_actual_density = max(actual_density_list)
        manual_density_list = df_manual_density["weight"].tolist()
        max_manual_weight = max(manual_density_list)

        for text_unit_index in range(df_actual_density.shape[0]):
            text = df_actual_density.iloc[text_unit_index]["word"]
            center_x = df_actual_density.iloc[text_unit_index]["x"]
            center_y = df_actual_density.iloc[text_unit_index]["y"]
            width = configs.text_width
            height = configs.text_height
            manual_color = (0, 0, manual_density_list[text_unit_index] / max_manual_weight)
            rect = patches.Rectangle((center_x - width / 2, center_y - height / 2), width, height, linewidth=0.5, edgecolor='none', facecolor=manual_color)
            ax.add_patch(rect)
            actual_color = (math.sqrt(actual_density_list[text_unit_index] / max_actual_density), 0, 0)
            ax.text(center_x, center_y, text, fontsize=10, horizontalalignment='center', verticalalignment='center', color=actual_color)

        for gaze_index in range(df_reading.shape[0]):
            gaze_x = df_reading.iloc[gaze_index]["gaze_x"]
            gaze_y = df_reading.iloc[gaze_index]["gaze_y"]
            ax.scatter(gaze_x, gaze_y, s=1, c='green')

    df_manual_weight_all = pd.read_csv(f"data/text/{configs.round}/weight/8_20_fine_relevance_gpt_weighted_90-94.csv", encoding="utf-8_sig")
    df_manual_weight_all.drop(columns=["Unnamed: 0"], axis=1, inplace=True)
    manual_weight_index = [90, 91, 92, 93, 94]

    def compute_distance_between_manual_and_actual(df_actual_density, df_manual_density):
        actual_density_list = df_actual_density["text_unit_density"].tolist()
        max_actual_density = max(actual_density_list)
        manual_density_list = df_manual_density["weight"].tolist()
        max_manual_weight = max(manual_density_list)

        relative_density_list = [actual_density_list[i] / max_actual_density for i in range(len(actual_density_list))]
        relative_manual_weight_list = [manual_density_list[i] / max_manual_weight for i in range(len(manual_density_list))]

        distance_list = []
        for i in range(len(relative_density_list)):
            distance_list.append(abs(relative_density_list[i] - relative_manual_weight_list[i]))

        print(np.mean(distance_list))

    manual_density_list = []
    for weight_index in manual_weight_index:
        df = df_manual_weight_all[df_manual_weight_all["para_id"] == weight_index]
        manual_density_list.append(df)

    text_unit_density_file_path_prefix = f"data/text_density/{configs.round}/{configs.device}/"
    text_unit_density_file_list = os.listdir(text_unit_density_file_path_prefix)
    df_text_density_list = []
    for file_index in range(len(text_unit_density_file_list)):
        df_text_density = pd.read_csv(f"{text_unit_density_file_path_prefix}{text_unit_density_file_list[file_index]}/text_unit_density.csv")
        df_text_density_list.append(df_text_density)

    actual_density_list = []
    for file_index in range(len(df_text_density_list)):
        df_list = []
        for text_unit_index in manual_weight_index:
            df = df_text_density_list[file_index]
            if df[df["para_id"] == text_unit_index].shape[0] > 0:
                df_list.append(df[df["para_id"] == text_unit_index])
        df_list.sort(key=lambda x: x.iloc[0]["para_id"])
        actual_density_list.append(df_list)

    reading_data_df_list = read_files.read_all_modified_reading_files()
    reading_data_list = []
    for file_index in range(len(reading_data_df_list)):
        df_list = []
        for iteration_index in range(len(reading_data_df_list[file_index])):
            for weight_index in manual_weight_index:
                df = reading_data_df_list[file_index][iteration_index]
                if df[df["matrix_x"] == weight_index].shape[0] > 0:
                    df_list.append(df[df["matrix_x"] == weight_index])
        df_list.sort(key=lambda x: x.iloc[0]["matrix_x"])
        reading_data_list.append(df_list)

    file_path = f"data/modified_gaze_data/{configs.round}/{configs.device}/"
    file_list = os.listdir(file_path)

    # visualize
    for file_index in range(len(actual_density_list)):
        if file_list[file_index] != "20230725_164316":
            continue
        # plt.cla()
        fig, axes = plt.subplots(2, 3)
        plt.subplots_adjust(left=0.05, bottom=0.05, right=0.95, top=0.95, wspace=0.1, hspace=0.1)
        plt.rcParams['font.sans-serif'] = ['SimSun']  # 指定默认字体

        for text_unit_index in range(len(actual_density_list[file_index])):
            axes[int(text_unit_index / 3)][text_unit_index % 3].set_xlim(250, 1750)
            axes[int(text_unit_index / 3)][text_unit_index % 3].set_ylim(1000, 0)
            axes[int(text_unit_index / 3)][text_unit_index % 3].set_aspect("equal")
            axes[int(text_unit_index / 3)][text_unit_index % 3].set_title(f"第{manual_weight_index[text_unit_index]}段")
            compute_distance_between_manual_and_actual(actual_density_list[file_index][text_unit_index], manual_density_list[text_unit_index])
            visual_density(axes[int(text_unit_index / 3)][text_unit_index % 3], actual_density_list[file_index][text_unit_index], manual_density_list[text_unit_index], reading_data_list[file_index][text_unit_index])

        print(file_list[file_index])
        manager = plt.get_current_fig_manager()
        manager.window.showMaximized()
        plt.show()


def down_sample_reading():
    reading_data_list = read_files.read_reading_data()
    std_points_list = analyse_calibration_data.create_standard_calibration_points()
    std_points_list = np.array(std_points_list).reshape(-1, 2)
    std_points_x_list = [std_points_list[i][0] for i in range(len(std_points_list))]
    std_points_y_list = [std_points_list[i][1] for i in range(len(std_points_list))]
    reading_file_list = os.listdir(f"data/modified_gaze_data/{configs.round}/{configs.device}/")

    for file_index in range(len(reading_data_list)):
        df_reading = pd.concat(reading_data_list[file_index])
        para_num = int(df_reading["matrix_x"].max() + 1)
        # gaze_x_down_sample_list = []
        # gaze_y_down_sample_list = []
        if reading_file_list[file_index] != "20230725_164316":
            continue

        for para_id in range(para_num):
            if para_id != 90:
                continue
            df = df_reading[df_reading["matrix_x"] == para_id]
            gaze_x = df["gaze_x"].tolist()
            gaze_y = df["gaze_y"].tolist()
            time = df["time"].tolist()
            gaze_x_down_sample = [gaze_x[0]]
            gaze_y_down_sample = [gaze_y[0]]
            time_down_sample = [time[0]]
            # gaze_x_down_sample_list.append(gaze_x[0])
            # gaze_y_down_sample_list.append(gaze_y[0])

            for i in range(1, len(gaze_x)):
                if time[i] - time_down_sample[-1] > 0.1:
                    gaze_x_down_sample.append(gaze_x[i])
                    gaze_y_down_sample.append(gaze_y[i])
                    time_down_sample.append(time[i])
                    # gaze_x_down_sample_list.append(gaze_x[i])
                    # gaze_y_down_sample_list.append(gaze_y[i])

            plt.xlim(0, 1920)
            plt.ylim(1200, 0)
            _, ax = plt.subplots()
            ax.set_aspect("equal")
            ax.scatter(std_points_x_list, std_points_y_list, s=5, c="k")
            ax.scatter(gaze_x, gaze_y, s=1, c="b")
            ax.scatter(gaze_x_down_sample, gaze_y_down_sample, s=5, c="r")
            plt.show()


def add_all_reading(weight_file_name):
    file_candidate_list = [0, 1, 2, 4, 5, 6]
    para_candidate_list = [90, 91, 92, 93, 94]

    text_unit_density_list = read_files.read_text_unit_density()
    manual_density_file_path = f"data/text/{configs.round}/weight/{weight_file_name}"
    # df_manual_density = pd.read_csv(manual_density_file_path, encoding="utf-8_sig", index_col=False)

    for para_index in range(len(para_candidate_list)):
        df_list = []
        for file_index in range(len(file_candidate_list)):
            df = text_unit_density_list[file_index]
            df = df[df["para_id"] == para_candidate_list[para_index]]
            df_list.append(df)

        df_all = df_list[0].copy(deep=True)
        new_relative_density_list = [0 for _ in range(df_all.shape[0])]
        df_all["relative_text_unit_density"] = new_relative_density_list

        for file_index in range(len(file_candidate_list)):
            df_all["relative_text_unit_density"] += df_list[file_index]["relative_text_unit_density"]

        relative_density_list = df_all["relative_text_unit_density"].tolist()

        for token_index in range(df_all.shape[0]):
            print("{" + f"'token_index': {token_index}, 'token': {df_all.iloc[token_index]['word']}, 'relative_density': {relative_density_list[token_index]:.6f}" + "}", end=", ")

        relative_density_list = [math.pow(i, 1/1) for i in relative_density_list]
        max_relative_density = max(relative_density_list)
        # manual_density_list = df_manual_density[df_manual_density["para_id"] == para_candidate_list[para_index]]["weight"].tolist()
        # max_manual_weight = max(manual_density_list)

        # visualize
        fig, ax = plt.subplots()
        ax.set_aspect('equal')
        ax.set_xlim(0, 1920)
        ax.set_ylim(1200, 0)
        plt.rcParams['font.sans-serif'] = ['SimSun']  # 指定默认字体

        for text_unit_index in range(df_all.shape[0]):
            text = df_all.iloc[text_unit_index]["word"]
            center_x = df_all.iloc[text_unit_index]["x"]
            center_y = df_all.iloc[text_unit_index]["y"]
            width = configs.text_width
            height = configs.text_height
            actual_color = (relative_density_list[text_unit_index] / max_relative_density, 0, 0)
            # manual_color = (0, 0, manual_density_list[text_unit_index] / max_manual_weight)
            ax.text(center_x, center_y, text, fontsize=15, horizontalalignment='center', verticalalignment='center', color=actual_color)
            # rect = patches.Rectangle((center_x - width / 2, center_y - height / 2), width, height, linewidth=0.5, edgecolor='none', facecolor=manual_color)
            rect = patches.Rectangle((center_x - width / 2, center_y - height / 2), width, height, linewidth=0.5, edgecolor='#DDDDDD', facecolor='none')
            ax.add_patch(rect)

        plt.show()







