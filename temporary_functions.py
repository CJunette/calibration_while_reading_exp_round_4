import math
import os
import numpy as np
import openai
import pandas as pd

import analyse_calibration_data
import analyse_reading_data
import configs
import talk_with_GPT


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


def get_tokens_of_certain_para(token_type, para_list):
    for para_index in para_list:
        file_name = f"data/text/{configs.round}/tokens/{token_type}_tokens/{para_index}.csv"
        df = pd.read_csv(file_name, encoding="utf-8_sig", index_col=False)
        token_list = []
        for token_index in range(df.shape[0]):
            token_list.append((token_index, df.iloc[token_index]["tokens"]))
        print(token_list)


def combine_temp_csv(file_prefix):
    file_path = f"data/text/{configs.round}/weight/temp/"
    file_list = os.listdir(file_path)

    df_list = []
    index_list = []
    for file_index in range(len(file_list)):
        if file_list[file_index].startswith(file_prefix):
            index_str = file_list[file_index].replace(file_prefix, "").replace(".csv", "").replace("_", "")
            index_list.append(index_str)
            file_name = file_path + file_list[file_index]
            df = pd.read_csv(file_name, encoding="utf-8_sig", index_col=False)
            df_list.append(df)

    index_list.sort()
    df_new = pd.concat(df_list)
    df_new.to_csv(f"data/text/{configs.round}/weight/{file_prefix}_{index_list[0]}-{index_list[-1]}.csv", encoding="utf-8_sig", index=False)


def ask_gpt_about_density():
    talk_with_GPT.set_openai()
    talk_with_GPT.start_using_IDE()

    response = openai.ChatCompletion.create(
        model="gpt-4-0613",
        messages=[
            {"role": "system", "content": f"你是一个分析文本与眼动行为的专家。"},
            {"role": "user", "content": "我会给你一系列文本文字和该文字对应的眼动停留时间（用相对密度来表示），将这些文字组合起来就是完整的文章。"
                                        "我会给你传递一个字典构成的列表，字典中的'token'代表文字，'token_density'代表相对的眼动停留时间。"
                                        "请告诉我这些眼动停留时间较长的文字的特点，为什么人在阅读它时会发生停留；以及我应该设计什么样的prompt才能让你能够预测出这样的眼动停留行为。\n"
                                        "文字及停留时间如下：\n"
                                        "[{'token_index': 0, 'token': 很, 'relative_density': 0.000204}, {'token_index': 1, 'token': 努, 'relative_density': 0.000126}, {'token_index': 2, 'token': 力, 'relative_density': 0.001339}, {'token_index': 3, 'token': 的, 'relative_density': 0.001728}, {'token_index': 4, 'token': 班, 'relative_density': 0.000794}, {'token_index': 5, 'token': 主, 'relative_density': 0.000749}, {'token_index': 6, 'token': 任, 'relative_density': 0.000094}, {'token_index': 7, 'token': :, 'relative_density': 0.000095}, {'token_index': 8, 'token':   , 'relative_density': 0.000000}, {'token_index': 9, 'token':   , 'relative_density': 0.000135}, {'token_index': 10, 'token': 有, 'relative_density': 0.000314}, {'token_index': 11, 'token': 没, 'relative_density': 0.001710}, {'token_index': 12, 'token': 有, 'relative_density': 0.001799}, {'token_index': 13, 'token': 发, 'relative_density': 0.000764}, {'token_index': 14, 'token': 现, 'relative_density': 0.000625}, {'token_index': 15, 'token': 现, 'relative_density': 0.000772}, {'token_index': 16, 'token': 在, 'relative_density': 0.000853}, {'token_index': 17, 'token': 小, 'relative_density': 0.000912}, {'token_index': 18, 'token': 孩, 'relative_density': 0.000866}, {'token_index': 19, 'token': 子, 'relative_density': 0.000777}, {'token_index': 20, 'token': 的, 'relative_density': 0.000750}, {'token_index': 21, 'token': 表, 'relative_density': 0.000780}, {'token_index': 22, 'token': 达, 'relative_density': 0.000645}, {'token_index': 23, 'token': 能, 'relative_density': 0.000791}, {'token_index': 24, 'token': 力, 'relative_density': 0.000805}, {'token_index': 25, 'token': 下, 'relative_density': 0.000497}, {'token_index': 26, 'token': 降, 'relative_density': 0.000338}, {'token_index': 27, 'token': 了, 'relative_density': 0.000198}, {'token_index': 28, 'token': ？, 'relative_density': 0.000115}, {'token_index': 29, 'token': 比, 'relative_density': 0.000145}, {'token_index': 30, 'token': 如, 'relative_density': 0.000285}, {'token_index': 31, 'token': 遇, 'relative_density': 0.000456}, {'token_index': 32, 'token': 到, 'relative_density': 0.000214}, {'token_index': 33, 'token': 任, 'relative_density': 0.000401}, {'token_index': 34, 'token': 何, 'relative_density': 0.000394}, {'token_index': 35, 'token': 事, 'relative_density': 0.000160}, {'token_index': 36, 'token': 情, 'relative_density': 0.000205}, {'token_index': 37, 'token': 都, 'relative_density': 0.000076}, {'token_index': 38, 'token': 可, 'relative_density': 0.000023}, {'token_index': 39, 'token': 以, 'relative_density': 0.000038}, {'token_index': 40, 'token': 用, 'relative_density': 0.000422}, {'token_index': 41, 'token': “, 'relative_density': 0.000690}, {'token_index': 42, 'token': 6, 'relative_density': 0.000541}, {'token_index': 43, 'token': ”, 'relative_density': 0.000561}, {'token_index': 44, 'token': 字, 'relative_density': 0.000614}, {'token_index': 45, 'token': 回, 'relative_density': 0.000873}, {'token_index': 46, 'token': 答, 'relative_density': 0.000456}, {'token_index': 47, 'token': 。, 'relative_density': 0.000306}, {'token_index': 48, 'token': 网, 'relative_density': 0.000142}, {'token_index': 49, 'token': 络, 'relative_density': 0.000634}, {'token_index': 50, 'token': 用, 'relative_density': 0.000779}, {'token_index': 51, 'token': 语, 'relative_density': 0.000430}, {'token_index': 52, 'token': 替, 'relative_density': 0.000763}, {'token_index': 53, 'token': 代, 'relative_density': 0.000337}, {'token_index': 54, 'token': 了, 'relative_density': 0.000275}, {'token_index': 55, 'token': 常, 'relative_density': 0.000334}, {'token_index': 56, 'token': 规, 'relative_density': 0.000401}, {'token_index': 57, 'token': 表, 'relative_density': 0.000244}, {'token_index': 58, 'token': 达, 'relative_density': 0.000132}, {'token_index': 59, 'token': ，, 'relative_density': 0.000165}, {'token_index': 60, 'token': 很, 'relative_density': 0.000272}, {'token_index': 61, 'token': 多, 'relative_density': 0.000704}, {'token_index': 62, 'token': 孩, 'relative_density': 0.000818}, {'token_index': 63, 'token': 子, 'relative_density': 0.000345}, {'token_index': 64, 'token': 作, 'relative_density': 0.000039}, {'token_index': 65, 'token': 业, 'relative_density': 0.000019}, {'token_index': 66, 'token': 、, 'relative_density': 0.000000}, {'token_index': 67, 'token': 日, 'relative_density': 0.000000}, {'token_index': 68, 'token': 记, 'relative_density': 0.000000}, {'token_index': 69, 'token': 、, 'relative_density': 0.000057}, {'token_index': 70, 'token': 作, 'relative_density': 0.000102}, {'token_index': 71, 'token': 文, 'relative_density': 0.000125}, {'token_index': 72, 'token': 都, 'relative_density': 0.000392}, {'token_index': 73, 'token': 有, 'relative_density': 0.000256}, {'token_index': 74, 'token': 网, 'relative_density': 0.000321}, {'token_index': 75, 'token': 络, 'relative_density': 0.000463}, {'token_index': 76, 'token': 用, 'relative_density': 0.000388}, {'token_index': 77, 'token': 语, 'relative_density': 0.000267}, {'token_index': 78, 'token': 的, 'relative_density': 0.000239}, {'token_index': 79, 'token': 影, 'relative_density': 0.000084}, {'token_index': 80, 'token': 子, 'relative_density': 0.000038}, {'token_index': 81, 'token': 。, 'relative_density': 0.000145}, {'token_index': 82, 'token':   , 'relative_density': 0.000000}, {'token_index': 83, 'token':   , 'relative_density': 0.000000}, {'token_index': 84, 'token': 记, 'relative_density': 0.000039}, {'token_index': 85, 'token': 得, 'relative_density': 0.000211}, {'token_index': 86, 'token': 有, 'relative_density': 0.000394}, {'token_index': 87, 'token': 一, 'relative_density': 0.000774}, {'token_index': 88, 'token': 阵, 'relative_density': 0.000523}, {'token_index': 89, 'token': 子, 'relative_density': 0.000810}, {'token_index': 90, 'token': 我, 'relative_density': 0.000488}, {'token_index': 91, 'token': 自, 'relative_density': 0.000638}, {'token_index': 92, 'token': 己, 'relative_density': 0.000894}, {'token_index': 93, 'token': 也, 'relative_density': 0.000658}, {'token_index': 94, 'token': 玩, 'relative_density': 0.000186}, {'token_index': 95, 'token': 梗, 'relative_density': 0.000362}, {'token_index': 96, 'token': ，, 'relative_density': 0.000294}, {'token_index': 97, 'token': “, 'relative_density': 0.000519}, {'token_index': 98, 'token': 信, 'relative_density': 0.000822}, {'token_index': 99, 'token': XX, 'relative_density': 0.001051}, {'token_index': 100, 'token': ，, 'relative_density': 0.001002}, {'token_index': 101, 'token': 得, 'relative_density': 0.000969}, {'token_index': 102, 'token': 永, 'relative_density': 0.000746}, {'token_index': 103, 'token': 生, 'relative_density': 0.000158}, {'token_index': 104, 'token': ”, 'relative_density': 0.000151}, {'token_index': 105, 'token': ，, 'relative_density': 0.000336}, {'token_index': 106, 'token': 时, 'relative_density': 0.000499}, {'token_index': 107, 'token': 间, 'relative_density': 0.001198}, {'token_index': 108, 'token': 长, 'relative_density': 0.000542}, {'token_index': 109, 'token': 了, 'relative_density': 0.000026}, {'token_index': 110, 'token': 导, 'relative_density': 0.000000}, {'token_index': 111, 'token': 致, 'relative_density': 0.000000}, {'token_index': 112, 'token': 一, 'relative_density': 0.000000}, {'token_index': 113, 'token': 些, 'relative_density': 0.000000}, {'token_index': 114, 'token': 本, 'relative_density': 0.000019}, {'token_index': 115, 'token': 会, 'relative_density': 0.000355}, {'token_index': 116, 'token': 组, 'relative_density': 0.000337}, {'token_index': 117, 'token': 织, 'relative_density': 0.000237}, {'token_index': 118, 'token': 的, 'relative_density': 0.000406}, {'token_index': 119, 'token': 语, 'relative_density': 0.000243}, {'token_index': 120, 'token': 言, 'relative_density': 0.000157}, {'token_index': 121, 'token': 都, 'relative_density': 0.000218}, {'token_index': 122, 'token': 不, 'relative_density': 0.000537}, {'token_index': 123, 'token': 知, 'relative_density': 0.000636}, {'token_index': 124, 'token': 怎, 'relative_density': 0.000093}, {'token_index': 125, 'token': 么, 'relative_density': 0.000084}, {'token_index': 126, 'token': 去, 'relative_density': 0.000081}, {'token_index': 127, 'token': 表, 'relative_density': 0.000221}, {'token_index': 128, 'token': 达, 'relative_density': 0.000065}, {'token_index': 129, 'token': 。, 'relative_density': 0.000180}, {'token_index': 130, 'token': 我, 'relative_density': 0.000405}, {'token_index': 131, 'token': 是, 'relative_density': 0.000412}, {'token_index': 132, 'token': 成, 'relative_density': 0.000124}, {'token_index': 133, 'token': 年, 'relative_density': 0.000069}, {'token_index': 134, 'token': 人, 'relative_density': 0.000054}, {'token_index': 135, 'token': ，, 'relative_density': 0.000073}, {'token_index': 136, 'token': 我, 'relative_density': 0.000152}, {'token_index': 137, 'token': 能, 'relative_density': 0.000451}, {'token_index': 138, 'token': 意, 'relative_density': 0.000149}, {'token_index': 139, 'token': 识, 'relative_density': 0.000077}, {'token_index': 140, 'token': 到, 'relative_density': 0.000000}, {'token_index': 141, 'token': 问, 'relative_density': 0.000000}, {'token_index': 142, 'token': 题, 'relative_density': 0.000000}, {'token_index': 143, 'token': 的, 'relative_density': 0.000091}, {'token_index': 144, 'token': 严, 'relative_density': 0.000181}, {'token_index': 145, 'token': 重, 'relative_density': 0.000120}, {'token_index': 146, 'token': 性, 'relative_density': 0.000105}, {'token_index': 147, 'token': ，, 'relative_density': 0.000111}, {'token_index': 148, 'token': 所, 'relative_density': 0.000147}, {'token_index': 149, 'token': 以, 'relative_density': 0.000243}, {'token_index': 150, 'token': 立, 'relative_density': 0.000144}, {'token_index': 151, 'token': 刻, 'relative_density': 0.000095}, {'token_index': 152, 'token': 克, 'relative_density': 0.000375}, {'token_index': 153, 'token': 制, 'relative_density': 0.000567}, {'token_index': 154, 'token': 网, 'relative_density': 0.000430}, {'token_index': 155, 'token': 络, 'relative_density': 0.000462}, {'token_index': 156, 'token': 表, 'relative_density': 0.000166}, {'token_index': 157, 'token': 达, 'relative_density': 0.000213}, {'token_index': 158, 'token': ，, 'relative_density': 0.000164}, {'token_index': 159, 'token': 多, 'relative_density': 0.000332}, {'token_index': 160, 'token': 看, 'relative_density': 0.000237}, {'token_index': 161, 'token': 名, 'relative_density': 0.000608}, {'token_index': 162, 'token': 著, 'relative_density': 0.000596}, {'token_index': 163, 'token': 和, 'relative_density': 0.000222}, {'token_index': 164, 'token': 经, 'relative_density': 0.000316}, {'token_index': 165, 'token': 典, 'relative_density': 0.000358}, {'token_index': 166, 'token': ，, 'relative_density': 0.000405}, {'token_index': 167, 'token': 才, 'relative_density': 0.000354}, {'token_index': 168, 'token': 把, 'relative_density': 0.000586}, {'token_index': 169, 'token': 思, 'relative_density': 0.000241}, {'token_index': 170, 'token': 维, 'relative_density': 0.000000}, {'token_index': 171, 'token': 带, 'relative_density': 0.000000}, {'token_index': 172, 'token': 回, 'relative_density': 0.000000}, {'token_index': 173, 'token': 来, 'relative_density': 0.000072}, {'token_index': 174, 'token': 。, 'relative_density': 0.000123}, {'token_index': 175, 'token': 成, 'relative_density': 0.000151}, {'token_index': 176, 'token': 年, 'relative_density': 0.001365}, {'token_index': 177, 'token': 人, 'relative_density': 0.000345}, {'token_index': 178, 'token': 都, 'relative_density': 0.000340}, {'token_index': 179, 'token': 是, 'relative_density': 0.000331}, {'token_index': 180, 'token': 如, 'relative_density': 0.000101}, {'token_index': 181, 'token': 此, 'relative_density': 0.000200}, {'token_index': 182, 'token': ，, 'relative_density': 0.000118}, {'token_index': 183, 'token': 小, 'relative_density': 0.000163}, {'token_index': 184, 'token': 孩, 'relative_density': 0.000235}, {'token_index': 185, 'token': 子, 'relative_density': 0.000707}, {'token_index': 186, 'token': 三, 'relative_density': 0.000888}, {'token_index': 187, 'token': 观, 'relative_density': 0.000626}, {'token_index': 188, 'token': 没, 'relative_density': 0.000407}, {'token_index': 189, 'token': 有, 'relative_density': 0.000630}, {'token_index': 190, 'token': 形, 'relative_density': 0.000206}, {'token_index': 191, 'token': 成, 'relative_density': 0.000159}, {'token_index': 192, 'token': ，, 'relative_density': 0.000104}, {'token_index': 193, 'token': 更, 'relative_density': 0.000222}, {'token_index': 194, 'token': 容, 'relative_density': 0.000425}, {'token_index': 195, 'token': 易, 'relative_density': 0.000372}, {'token_index': 196, 'token': 被, 'relative_density': 0.000250}, {'token_index': 197, 'token': 影, 'relative_density': 0.000086}, {'token_index': 198, 'token': 响, 'relative_density': 0.000096}, {'token_index': 199, 'token': 。, 'relative_density': 0.000000}, {'token_index': 200, 'token': 在, 'relative_density': 0.000000}, {'token_index': 201, 'token': 班, 'relative_density': 0.000000}, {'token_index': 202, 'token': 级, 'relative_density': 0.000019}, {'token_index': 203, 'token': ，, 'relative_density': 0.000019}, {'token_index': 204, 'token': 对, 'relative_density': 0.000000}, {'token_index': 205, 'token': 于, 'relative_density': 0.000140}, {'token_index': 206, 'token': 学, 'relative_density': 0.000151}, {'token_index': 207, 'token': 生, 'relative_density': 0.000229}, {'token_index': 208, 'token': 玩, 'relative_density': 0.000328}, {'token_index': 209, 'token': 梗, 'relative_density': 0.000093}, {'token_index': 210, 'token': 一, 'relative_density': 0.000073}, {'token_index': 211, 'token': 直, 'relative_density': 0.000146}, {'token_index': 212, 'token': 以, 'relative_density': 0.000163}, {'token_index': 213, 'token': 来, 'relative_density': 0.000316}, {'token_index': 214, 'token': 我, 'relative_density': 0.000306}, {'token_index': 215, 'token': 都, 'relative_density': 0.000195}, {'token_index': 216, 'token': 没, 'relative_density': 0.000166}, {'token_index': 217, 'token': 有, 'relative_density': 0.000107}, {'token_index': 218, 'token': 一, 'relative_density': 0.000165}, {'token_index': 219, 'token': 棒, 'relative_density': 0.000132}, {'token_index': 220, 'token': 子, 'relative_density': 0.000296}, {'token_index': 221, 'token': 打, 'relative_density': 0.000019}, {'token_index': 222, 'token': 死, 'relative_density': 0.000019}, {'token_index': 223, 'token': ，, 'relative_density': 0.000077}, {'token_index': 224, 'token': 但, 'relative_density': 0.000090}, {'token_index': 225, 'token': 是, 'relative_density': 0.000052}, {'token_index': 226, 'token': 希, 'relative_density': 0.000058}, {'token_index': 227, 'token': 望, 'relative_density': 0.000163}, {'token_index': 228, 'token': 未, 'relative_density': 0.000039}, {'token_index': 229, 'token': 来, 'relative_density': 0.000000}, {'token_index': 230, 'token': 孩, 'relative_density': 0.000000}, {'token_index': 231, 'token': 子, 'relative_density': 0.000000}, {'token_index': 232, 'token': 们, 'relative_density': 0.000019}, {'token_index': 233, 'token': 可, 'relative_density': 0.000000}, {'token_index': 234, 'token': 以, 'relative_density': 0.000000}, {'token_index': 235, 'token': 早, 'relative_density': 0.000140}, {'token_index': 236, 'token': 日, 'relative_density': 0.000019}, {'token_index': 237, 'token': 有, 'relative_density': 0.000000}, {'token_index': 238, 'token': 自, 'relative_density': 0.000000}, {'token_index': 239, 'token': 己, 'relative_density': 0.000073}, {'token_index': 240, 'token': 的, 'relative_density': 0.000097}, {'token_index': 241, 'token': 判, 'relative_density': 0.000148}, {'token_index': 242, 'token': 断, 'relative_density': 0.000229}, {'token_index': 243, 'token': ，, 'relative_density': 0.000297}, {'token_index': 244, 'token': 家, 'relative_density': 0.000286}, {'token_index': 245, 'token': 长, 'relative_density': 0.000174}, {'token_index': 246, 'token': 、, 'relative_density': 0.000210}, {'token_index': 247, 'token': 学, 'relative_density': 0.000277}, {'token_index': 248, 'token': 校, 'relative_density': 0.000329}, {'token_index': 249, 'token': 、, 'relative_density': 0.000222}, {'token_index': 250, 'token': 社, 'relative_density': 0.000185}, {'token_index': 251, 'token': 会, 'relative_density': 0.000362}, {'token_index': 252, 'token': 共, 'relative_density': 0.000196}, {'token_index': 253, 'token': 同, 'relative_density': 0.000438}, {'token_index': 254, 'token': 引, 'relative_density': 0.000258}, {'token_index': 255, 'token': 导, 'relative_density': 0.000237}, {'token_index': 256, 'token': ，, 'relative_density': 0.000200}, {'token_index': 257, 'token': 千, 'relative_density': 0.000318}, {'token_index': 258, 'token': 万, 'relative_density': 0.000441}, {'token_index': 259, 'token': 不, 'relative_density': 0.000116}, {'token_index': 260, 'token': 可, 'relative_density': 0.000023}, {'token_index': 261, 'token': 让, 'relative_density': 0.000000}, {'token_index': 262, 'token': 孩, 'relative_density': 0.000058}, {'token_index': 263, 'token': 子, 'relative_density': 0.000000}, {'token_index': 264, 'token': 们, 'relative_density': 0.000019}, {'token_index': 265, 'token': 沉, 'relative_density': 0.000085}, {'token_index': 266, 'token': 迷, 'relative_density': 0.000252}, {'token_index': 267, 'token': 网, 'relative_density': 0.000826}, {'token_index': 268, 'token': 络, 'relative_density': 0.000591}, {'token_index': 269, 'token': 梗, 'relative_density': 0.000246}, {'token_index': 270, 'token': ！, 'relative_density': 0.000267}]"}]
    )

    response_str = response["choices"][0]["message"]["content"].strip()
    print(response_str)


def ask_gpt_about_article():
    talk_with_GPT.set_openai()
    talk_with_GPT.start_using_IDE()

    response = openai.ChatCompletion.create(
        model="gpt-4-0613",
        messages=[
            {"role": "system", "content": f"你是一个分析文本阅读行为的专家。你需要找出用户在快速阅读时，最可能关注的那些能够“吸引眼球”的词语。注意，由于用户是在做快速阅读，所以对于首尾内容关注会更多，对于中间的内容则会更有选择性。"},
            {"role": "user", "content": "我会给你一系列文本，你需要告诉我这段文本的类型是什么（新闻、推荐、微博、评论还是其他类型），可能是在哪个平台发布的。"
                                        "以及对于这个平台的用户而言，这些类型的文本应该着重关注哪些内容（what_to_focus，用自然语言描述）；这篇文章中主要关注的文字（分词）是哪些（focus_words）；"
                                        "这类文本不该关注的内容是哪些（what_to_ignore，用自然语言描述）；这篇文章中没有关注的文字（分词）是哪些（ignored_words）。"
                                        "请注意，值得关注的内容与不值得关注的内容间不应该存在冲突（即对于2个相似的概念，不应该一个出现在值得关注，另一个出现在不值得关注）。你可以尽可能多地找出需要关注和不需要关注的内容。\n"
                                        "你需要用字典返回这五个内容，如"
                                        "{'type': '这段文本可以被归类为一种“评论/社交媒体评论”类型。这是一篇关于青旅拒绝35岁以上顾客的政策的讨论，显然是发生在一个类似知乎的在线问答平台。', "
                                        "'what_to_focus': '讨论或评论的主题，评论者的态度或立场，关键论点，逻辑推理，以及可能引发进一步讨论或争议的点。', "
                                        "'what_to_ignore': '过于冗长的背景说明，一些无关的插入语或过度的个人情绪。'"
                                        "'focus_words': '在这段文本中，主要的关键词或者说是分词可能包括“北京”（地点）、“青旅”（主题/对象），“35岁以上顾客”（讨论的焦点），“生活习惯不同，不好管理”（拒绝的原因），“不能认定违规”（法律意见），以及“歧视”、“收入低”、“素质差性格有问题”、“因果关系”（讨论或争辩的关键点）。'"
                                        "'ignored_words: '“知乎”，以及一个具体的引用（很多支持歧视的回答依据基本是“一个人如果到35岁还住青旅，说明这个人。。。（收入低素质差性格有问题等等）”）。这个引用合并在了我关注的文字中，因为我会更关注该引用中的主要观点，而不是引用本身。'}\n"
                                        "文本如下：\n"
                                        """
    “饭圈”产业价值千亿却走向癫狂，谁该负责
    “饭圈”文化愈演愈烈，粉丝与网络水军混杂，因各种立场、观点、利益冲突，而引发各类网上互撕互黑等风波。6月15日，中央网信办宣布开展为期两个月的“饭圈”乱象整治专项行动，聚焦明星榜单、热门话题、粉丝社群、互动评论等环节。突如其来的强监管，显示近年狂飙突进的“粉丝经济”走到了十字路口。
    官方文件直接点名“饭圈”，前所未见。“饭圈”是一个近年走红的网络用语，主要指娱乐明星粉丝（Fans）组成的圈子。不同于过去所谓“追星族”，“饭圈”更多是基于社群网络的半职业化组织，一些娱乐明星的粉丝业已形成职业分工运作模式，包括“粉头”“数据女工”等新型角色，深度参与明星日常活动，为明星造热度，维持形象和商业价值。
    """}]
    )

    # TODO 之后先将文本给GPT让它给出3段分析，然后把这3段分析给GPT，让它对每个分词进行打分。
    response_str = response["choices"][0]["message"]["content"].strip()
    print(response_str)
