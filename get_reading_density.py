import math
import os
from multiprocessing import Pool

import pandas as pd

import configs
import read_files


def get_text_unit_density_single_pool(text_mapping_list, file_index, matrix_x, df_reading_matrix_x):
    print(file_index, matrix_x)
    para_id = int(matrix_x)
    df_text_mapping = text_mapping_list[file_index]
    df_text_mapping_of_para_id = df_text_mapping[df_text_mapping["para_id"] == para_id]

    # TODO modified_gaze_data下的reading data都被修改过了，因此这里不需要再做额外的矫正。
    text_unit_density = [0 for _ in range(df_text_mapping_of_para_id.shape[0])]
    for text_unit_index in range(df_reading_matrix_x.shape[0]):
        gaze_x = df_reading_matrix_x.iloc[text_unit_index]["gaze_x"]
        gaze_y = df_reading_matrix_x.iloc[text_unit_index]["gaze_y"]
        distance_to_text_unit_list = []
        for text_mapping_index in range(df_text_mapping_of_para_id.shape[0]):
            text_mapping_x = df_text_mapping_of_para_id.iloc[text_mapping_index]["x"]
            text_mapping_y = df_text_mapping_of_para_id.iloc[text_mapping_index]["y"]
            distance_to_gaze = math.sqrt((gaze_x - text_mapping_x) ** 2 + (gaze_y - text_mapping_y) ** 2)
            distance_to_text_unit_list.append(distance_to_gaze)
        # 添加额外的判断，如果距离太远，就不属于任何一个text unit。
        min_distance = min(distance_to_text_unit_list)
        if min_distance < configs.text_unit_density_threshold:
            min_index = distance_to_text_unit_list.index(min(distance_to_text_unit_list))
            text_unit_density[min_index] += 1
    df_text_mapping_of_para_id["text_unit_density"] = text_unit_density
    return df_text_mapping_of_para_id, file_index, para_id


def get_text_unit_density():
    text_mapping_list = read_files.read_all_modified_reading_text_mapping()
    df_reading_list = read_files.read_all_modified_reading_files()

    # for debug
    # df_text_density_list = []
    # for file_index in range(len(df_reading_list)):
    #     for iteration_index in range(len(df_reading_list[file_index])):
    #         df_reading = df_reading_list[file_index][iteration_index]
    #         df_reading_grouped_by_matrix_x = df_reading.groupby("matrix_x")
    #         for matrix_x, df_reading_matrix_x in df_reading_grouped_by_matrix_x:
    #             text_density_result = get_text_unit_density_single_pool(text_mapping_list, file_index, matrix_x, df_reading_matrix_x)
    #             df_text_density_list.append(text_density_result)

    file_list = os.listdir(f"data/modified_gaze_data/{configs.round}/{configs.device}")
    # for multi pool
    for file_index in range(len(df_reading_list)):
        args_list = []

        for iteration_index in range(len(df_reading_list[file_index])):
            df_reading = df_reading_list[file_index][iteration_index]
            df_reading_grouped_by_matrix_x = df_reading.groupby("matrix_x")
            for matrix_x, df_reading_matrix_x in df_reading_grouped_by_matrix_x:
                args = [text_mapping_list, file_index, matrix_x, df_reading_matrix_x]
                args_list.append(args)

        with Pool(16) as p:
            text_mapping_result = p.starmap(get_text_unit_density_single_pool, args_list)

    # for file_index in range(len(df_reading_list)):
        text_mapping_result_of_file_index = []
        for text_mapping_result_item in text_mapping_result:
            if text_mapping_result_item[1] == file_index:
                text_mapping_result_of_file_index.append(text_mapping_result_item)

        text_mapping_result_of_file_index.sort(key=lambda x: x[2])
        df_text_mapping_result_of_file_index = pd.concat([item[0] for item in text_mapping_result_of_file_index])

        # compute the relative density here
        total_gaze_count = df_text_mapping_result_of_file_index["text_unit_density"].sum()
        df_text_mapping_result_of_file_index["relative_text_unit_density"] = df_text_mapping_result_of_file_index["text_unit_density"] / total_gaze_count

        save_path = f"data/text_density/{configs.round}/{configs.device}/{file_list[file_index]}/text_unit_density.csv"
        if not os.path.exists(os.path.dirname(save_path)):
            os.makedirs(os.path.dirname(save_path))
        df_text_mapping_result_of_file_index.to_csv(save_path, encoding="utf-8_sig", index=False)


def get_token_density_single_pool(df_token_list, df_text_density_combined, file_index, para_index):
    print(file_index, para_index)
    df_token = df_token_list[para_index]
    df_text_density = df_text_density_combined[df_text_density_combined["para_id"] == para_index]

    density_list = []
    relative_density_list = []
    row_list = []
    para_id_list = []
    row_start = df_text_density["row"].iloc[0]
    for token_index in range(df_token.shape[0]):
        row = df_token["row"].iloc[token_index][0] + row_start
        text_unit_components = df_token["text_unit_component"].iloc[token_index][0]
        density = 0
        relative_density = 0
        for text_unit_component in text_unit_components:
            density += df_text_density[df_text_density["row"] == row][df_text_density["col"] == text_unit_component]["text_unit_density"].iloc[0]
            relative_density += df_text_density[df_text_density["row"] == row][df_text_density["col"] == text_unit_component]["relative_text_unit_density"].iloc[0]
        density_list.append(density)
        relative_density_list.append(relative_density)
        row_list.append([row])
        para_id_list.append(para_index)

    df_token["density"] = density_list
    df_token["relative_density"] = relative_density_list
    df_token["row"] = row_list
    df_token["para_id"] = para_id_list
    return df_token, file_index, para_index


def get_token_density():
    def create_args_list(df_token_list, df_text_density_list):
        args_list = []
        for file_index in range(len(df_text_density_list)):
            df_text_density_combined = df_text_density_list[file_index]

            for para_index in range(len(df_token_list)):
                args = [df_token_list, df_text_density_combined, file_index, para_index]
                args_list.append(args)

        return args_list

    def combine_df_token_and_save(token_density_result, token_type, text_unit_density_file_list):
        token_density_grouped_by_file_index = [[] for _ in range(len(text_unit_density_file_list))]
        for index in range(len(token_density_result)):
            file_index = token_density_result[index][1]
            token_density_grouped_by_file_index[file_index].append(token_density_result[index])

        for file_index in range(len(token_density_grouped_by_file_index)):
            token_density_grouped_by_file_index[file_index].sort(key=lambda x: x[2])
            df_token_density_grouped_by_file_index = pd.concat([item[0] for item in token_density_grouped_by_file_index[file_index]])

            save_path = f"data/text_density/{configs.round}/{configs.device}/{text_unit_density_file_list[file_index]}/{token_type}_token_density.csv"
            if not os.path.exists(os.path.dirname(save_path)):
                os.makedirs(os.path.dirname(save_path))
            df_token_density_grouped_by_file_index.to_csv(save_path, encoding="utf-8_sig", index=False)

    text_unit_density_file_path_prefix = f"data/text_density/{configs.round}/{configs.device}/"
    text_unit_density_file_list = os.listdir(text_unit_density_file_path_prefix)
    df_text_density_list = []
    for file_index in range(len(text_unit_density_file_list)):
        df_text_density = pd.read_csv(f"{text_unit_density_file_path_prefix}{text_unit_density_file_list[file_index]}/text_unit_density.csv")
        df_text_density_list.append(df_text_density)

    # for debug
    # df_fine_token_list = read_files.read_all_token_data("fine")
    # for file_index in range(len(df_text_density_list)):
    #     df_text_density_combined = df_text_density_list[file_index]
    #
    #     df_fine_token_density_list = []
    #     for para_index in range(len(df_fine_token_list)):
    #         df_fine_token_density, _, _ = get_token_density_single_pool(df_fine_token_list, df_text_density_combined, file_index, para_index)
    #         df_fine_token_density_list.append(df_fine_token_density)
    #     df_fine_token_density = pd.concat(df_fine_token_density_list, ignore_index=True)
    #
    #     # fine_save_path = f"data/text_density/{configs.round}/{configs.device}/{text_unit_density_file_list[file_index]}/fine_token_density.csv"
    #     # df_fine_token_density.to_csv(fine_save_path, encoding="utf-8_sig", index=False)

    df_fine_token_list = read_files.read_all_token_data("fine")
    fine_args_list = create_args_list(df_fine_token_list, df_text_density_list)
    df_coarse_token_list = read_files.read_all_token_data("coarse")
    coarse_args_list = create_args_list(df_coarse_token_list, df_text_density_list)

    with Pool(16) as p:
        fine_token_density_result = p.starmap(get_token_density_single_pool, fine_args_list)
        coarse_token_density_result = p.starmap(get_token_density_single_pool, coarse_args_list)

    combine_df_token_and_save(fine_token_density_result, "fine", text_unit_density_file_list)
    combine_df_token_and_save(coarse_token_density_result, "coarse", text_unit_density_file_list)
