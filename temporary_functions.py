import math
import os
import numpy as np
import pandas as pd

import analyse_calibration_data
import analyse_reading_data
import configs


def combine_error_data_in_round_4():
    '''
    用于将出错的实验中的有效阅读数据与后续补救实验中的数据合到一起。
    :return:
    '''

    dir_target = "D:\\Work\\2023.7.25_visualization_of_gaze_pilot_experiment_3_ver_0\\calibrationdataanalyzer\\packed_data\\round_4\\tobii\\20230810_111446_20230809_1.5_fxz\\reading\\"
    dir_source = "D:\\Work\\2023.7.25_visualization_of_gaze_pilot_experiment_3_ver_0\\calibrationdataanalyzer\\packed_data\\round_4\\tobii\\20230810_102118_20230810_1_fxz\\reading\\"

    target_file_list = os.listdir(dir_target)
    target_df_list = []
    for file_name in target_file_list:
        df = pd.read_csv(dir_target + file_name, encoding="utf-8")
        target_df_list.append(df)

    source_file_list = os.listdir(dir_source)
    source_df_list = []
    for file_name in source_file_list:
        df = pd.read_csv(dir_source + file_name, encoding="utf-8")
        source_df_list.append(df)

    source_para_id_list_1 = []
    source_df_of_para_id_list_1 = []
    for source_index in range(len(source_df_list)):
        source_para_id_list_2 = source_df_list[source_index]["matrix_x"].unique()
        source_para_id_list_1.append(source_para_id_list_2)
        source_df_of_para_id_list_2 = []
        for para_id in source_para_id_list_2:
            df = source_df_list[source_index]
            df_para_id = df[df["matrix_x"] == para_id]
            source_df_of_para_id_list_2.append(df_para_id)
        source_df_of_para_id_list_1.append(source_df_of_para_id_list_2)

    target_para_id_list_1 = []
    target_df_of_para_id_list_1 = []
    for target_index in range(len(target_df_list)):
        target_para_id_list_2 = target_df_list[target_index]["matrix_x"].unique()
        target_para_id_list_1.append(target_para_id_list_2)
        target_df_of_para_id_list_2 = []
        for para_id in target_para_id_list_2:
            df = target_df_list[target_index]
            df_para_id = df[df["matrix_x"] == para_id]
            target_df_of_para_id_list_2.append(df_para_id)
        target_df_of_para_id_list_1.append(target_df_of_para_id_list_2)

    for source_index in range(len(source_df_of_para_id_list_1)):
        for sub_source_index in range(len(source_para_id_list_1[source_index])):
            actual_source_index = source_para_id_list_1[source_index][sub_source_index]
            source_df = source_df_of_para_id_list_1[source_index][sub_source_index]
            for target_index in range(len(target_df_of_para_id_list_1)):
                if actual_source_index in target_para_id_list_1[target_index]:
                    target_df_index = np.where(target_para_id_list_1[target_index] == actual_source_index)[0][0]
                    target_df = target_df_of_para_id_list_1[target_index][target_df_index]

                    phase = source_df["phase"].tolist()
                    source_length = len(phase)
                    username = [target_df["username"].tolist()[0] for _ in range(source_length)]
                    iteration = [target_df["iteration"].tolist()[0] for _ in range(source_length)]
                    time = source_df["time"].tolist()
                    gaze_x = source_df["gaze_x"].tolist()
                    gaze_y = source_df["gaze_y"].tolist()
                    speed = source_df["speed"].tolist()
                    matrix_x = [target_df["matrix_x"].tolist()[0] for _ in range(source_length)]
                    matrix_y = [target_df["matrix_y"].tolist()[0] for _ in range(source_length)]

                    new_df = pd.DataFrame({
                        "username": username,
                        "iteration": iteration,
                        "phase": phase,
                        "time": time,
                        "gaze_x": gaze_x,
                        "gaze_y": gaze_y,
                        "speed": speed,
                        "matrix_x": matrix_x,
                        "matrix_y": matrix_y
                    })

                    target_df_of_para_id_list_1[target_index][target_df_index] = new_df
                    break

    for target_index in range(len(target_df_of_para_id_list_1)):
        con_df = pd.concat(target_df_of_para_id_list_1[target_index])
        con_df.reset_index()

        con_df_group_by_matrix_x = con_df.groupby("matrix_x")
        start_time_list = []
        matrix_x_list = []
        for matrix_x, df in con_df_group_by_matrix_x:
            start_time_list.append(df["time"].tolist())
            matrix_x_list.append(matrix_x)
        start_time_difference = abs(start_time_list[0][-1] - start_time_list[1][0])
        if start_time_difference > 100: # 这里的100是个经验值，指的是同一个iteration下，前一个df的最后一个时间和后一个df的第一个时间之间的差。
            second_df_start_time = start_time_list[1][0]
            new_start_time = start_time_list[0][-1] + 20 # 这里的20也是经验值，是同一个iteration下，给前一个df的最后一个时间加上20后作为后一个df的第一个时间。
            for gaze_index in range(con_df.shape[0]):
                if con_df.iloc[gaze_index]["matrix_x"] == matrix_x_list[1]:
                    second_df_original_time = con_df.iloc[gaze_index]["time"]
                    con_df.iloc[gaze_index, con_df.columns.get_loc("time")] = second_df_original_time - second_df_start_time + new_start_time

        save_prefix = dir_target
        if not os.path.exists(os.path.dirname(save_prefix)):
            os.makedirs(os.path.dirname(save_prefix))

        save_name = target_file_list[target_index][:-4]
        save_path = save_prefix + save_name + ".csv"

        con_df.to_csv(save_path, encoding="utf-8_sig")


def split_data_in_round_1():
    exp_text_file = "data/back_up_gaze_data/round_1/exp_text.txt"
    with open(exp_text_file, "r", encoding="utf-8_sig") as f:
        exp_text = f.read()
    exp_text = exp_text.replace("\n\n\n", "\n-----\n")

    round_1_original_data_dir = "data/back_up_gaze_data/round_1/original_data/"
    file_list = os.listdir(round_1_original_data_dir)
    file_list = [file for file in file_list if file.endswith(".csv")]
    df_list = []
    for file_index in range(len(file_list)):
        file_name = round_1_original_data_dir + file_list[file_index]
        df = pd.read_csv(file_name, encoding="utf-8_sig")
        df_list.append(df)

    file_prefix = "data/back_up_gaze_data/round_1/modified_data/"
    for file_index in range(len(df_list)):
        file_path = file_prefix + file_list[file_index][:-4] + "/"
        reading_file_path = file_path + "reading/"
        if not os.path.exists(reading_file_path):
            os.makedirs(reading_file_path)

        # save reading of each iteration as different files.
        df = df_list[file_index]
        df_reading = df[df["phase"] == "word"]
        df_reading_group_by_iteration = df_reading.groupby("iteration")
        for iteration, df_iteration in df_reading_group_by_iteration:
            reading_save_path = f"{reading_file_path}{int(float(iteration))}.csv"
            df_iteration.to_csv(reading_save_path, encoding="utf-8_sig")

        # save calibration
        df_calibration = df[df["phase"] == "cali"]
        cali_save_path = f"{file_path}calibration.csv"
        df_calibration.to_csv(cali_save_path, encoding="utf-8_sig")

        # save exp_text
        exp_text_save_path = f"{file_path}exp_text.txt"
        with open(exp_text_save_path, "w", encoding="utf-8_sig") as f:
            f.write(exp_text)


def split_seeso_data():
    file_path = "data/modified_gaze_data/round_4/seeso/"
    file_list = os.listdir(file_path)
    file_list = [file for file in file_list if not file.startswith("seeso")]

    for file_index in range(len(file_list)):
        file_name = f"{file_path}{file_list[file_index]}"
        df = pd.read_csv(file_name, encoding="utf-8_sig", index_col=False)
        # TODO 这里交换列的地方可能可以去掉，之后的数据应该会被处理好。
        df_wrong_time = df["time"].tolist()
        # 将df的第6列到第10列数据左移一列。第6列数据覆盖第5列数据。
        df.iloc[:, 5:10] = df.iloc[:, 5:10].shift(-1, axis=1)
        # 将df的第10列数据替换为df_wrong_time
        df["TrackingState"] = df_wrong_time

        df_reading = df[df["phase"] == "word"]
        df_cali = df[df["phase"] == "cali"]

        save_path = f"{file_path}/{file_list[file_index][:-4]}/"
        df_reading_grouped_by_iteration = df_reading.groupby("iteration")
        for iteration, df_iteration in df_reading_grouped_by_iteration:
            df_reading_grouped_by_matrix_x = df_iteration.groupby("matrix_x")
            speed_list = []
            for matrix_x, df_matrix_x in df_reading_grouped_by_matrix_x:
                speed = [0]
                for i in range(1, df_matrix_x.shape[0]):
                    gaze_x = df_matrix_x.iloc[i]["gaze_x"]
                    gaze_y = df_matrix_x.iloc[i]["gaze_y"]
                    last_x = df_matrix_x.iloc[i - 1]["gaze_x"]
                    last_y = df_matrix_x.iloc[i - 1]["gaze_y"]
                    distance = math.sqrt((gaze_x - last_x) ** 2 + (gaze_y - last_y) ** 2)
                    speed.append(distance)
                speed_list.extend(speed)
            df_iteration["speed"] = speed_list
            df_iteration.dropna(inplace=True)
            save_name = f"{save_path}/seeso_reading/{int(float(iteration))}.csv"
            if not os.path.exists(os.path.dirname(save_name)):
                os.makedirs(os.path.dirname(save_name))
            df_iteration.to_csv(save_name, encoding="utf-8_sig")

        df_cali.to_csv(f"{save_path}/seeso_calibration.csv", encoding="utf-8_sig")


def modify_round_1_reading_data_using_calibration():
    file_path_prefix = f"data/back_up_gaze_data/round_1/reformat_data/"
    file_list = os.listdir(file_path_prefix)

    for file_index in range(len(file_list)):
        calibration_file_name = f"{file_path_prefix}/{file_list[file_index]}/calibration.csv"
        df_calibration = pd.read_csv(calibration_file_name, encoding="utf-8_sig")
        homography_matrix = analyse_calibration_data.get_homography_matrix_for_calibration(df_calibration)

        reading_file_path = f"{file_path_prefix}/{file_list[file_index]}/reading"
        reading_file_list = os.listdir(reading_file_path)
        reading_file_list = [int(i[:-4]) for i in reading_file_list]
        reading_file_list.sort()

        for reading_file_index in range(len(reading_file_list)):
            df = pd.read_csv(f"{reading_file_path}/{reading_file_list[reading_file_index]}.csv", encoding="utf-8_sig", index_col=False)
            df.drop(df.columns[0], axis=1, inplace=True)
            transformed_df = analyse_reading_data.apply_homography_to_reading(df, homography_matrix)

            save_name = f"data/modified_gaze_data/round_1/tobii/{file_list[file_index]}/reading/{reading_file_index}.csv"
            transformed_df.to_csv(save_name, encoding="utf-8_sig", index=False)


def generate_reading_data_90_to_94():
    '''
        单独生成90-94的reading数据。
    '''
    file_path_prefix = f"data/back_up_gaze_data/round_1/reformat_data/"
    file_list = os.listdir(file_path_prefix)

    reading_df_list_1 = []
    calibration_df_list_1 = []
    for file_index in range(len(file_list)):
        calibration_file_name = f"{file_path_prefix}/{file_list[file_index]}/calibration.csv"
        df_calibration = pd.read_csv(calibration_file_name, encoding="utf-8_sig")
        calibration_df_list_1.append(df_calibration)

        reading_file_path = f"{file_path_prefix}/{file_list[file_index]}/reading"
        reading_file_list = os.listdir(reading_file_path)
        reading_file_list = [int(i[:-4]) for i in reading_file_list]
        reading_file_list.sort()
        df_90_94_list = []
        index_list = [90, 91, 92, 93, 94]
        for reading_file_index in range(len(reading_file_list)):
            df = pd.read_csv(f"{reading_file_path}/{reading_file_list[reading_file_index]}.csv", encoding="utf-8_sig")
            para_id_list = df["matrix_x"].unique()
            para_id_list = [int(para_id_list[i]) for i in range(len(para_id_list))]
            for index in index_list:
                if index in para_id_list:
                    df_90_94_list.append(df[df["matrix_x"] == index])
        df_new = pd.concat(df_90_94_list)
        save_path = f"data/back_up_gaze_data/round_1/temp/90-94/"
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        df_new.to_csv(save_path + f"{file_list[file_index]}_90-94.csv", encoding="utf-8_sig", index=False)