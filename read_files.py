import json
import os

import pandas as pd

import configs


def read_sorted_text():
    '''
    由于round4的实验文本都是统一的，唯一的区别是文本出现的位置和顺序。因此在做分词时，只需要分词一次即可。
    这里统一读取data/text/下的exp_text_sorted.txt。
    :return:
    '''

    # 读取数据
    file_path = f"data/text/{configs.round}/exp_text_sorted.txt"
    with open(file_path, 'r', encoding='utf-8_sig') as f:
        text = f.read()

    text_str_list = text.split("-----\n")
    text_list = []
    for index in range(len(text_str_list)):
        text_str = text_str_list[index]
        text_index = text_str.split(" jjxnb\n")[0]
        text = text_str.split(" jjxnb\n")[1]
        text_list.append({"text_index": text_index, "text": text})

    return text_list


def read_text_mapping_of_sorted_data():
    '''
    这里根据sorted_data处理得到了一个text_mapping。
    这里返回的是将text_unit按照para_id和row_id分组的结果。
    :return: 返回值是一个三维数组，第一维代表不同para_id，第二维代表不同row_id，第三维代表不同text_unit。
    '''

    df_text_mapping = pd.read_csv(f"data/text/{configs.round}/text_sorted_mapping.csv", encoding="utf-8_sig")

    df_text_mapping_grouped_by_para_id = df_text_mapping.groupby("para_id")
    text_unit_list_1 = []

    for para_id, df_grouped_by_para_id in df_text_mapping_grouped_by_para_id:
        text_unit_list_2 = [[] for _ in range(configs.row_num)]
        df_grouped_by_row = df_grouped_by_para_id.groupby("row")
        for row_id, df_grouped_by_row in df_grouped_by_row:
            text_unit_list_3 = []
            for text_unit_index in range(df_grouped_by_row.shape[0]):
                text_unit_list_3.append(df_grouped_by_row.iloc[text_unit_index]["word"])
            text_unit_list_2[row_id] = text_unit_list_3
        text_unit_list_1.append(text_unit_list_2)

    return text_unit_list_1


def read_token_file_names():
    fine_tokens_path_prefix = f"data/text/{configs.round}/tokens/fine_tokens/"
    coarse_tokens_path_prefix = f"data/text/{configs.round}/tokens/coarse_tokens/"

    fine_tokens_file_list = os.listdir(fine_tokens_path_prefix)
    coarse_tokens_file_list = os.listdir(coarse_tokens_path_prefix)
    fine_token_file_index_list = [int(fine_tokens_file_list[i][:-4]) for i in range(len(fine_tokens_file_list))]
    coarse_token_file_index_list = [int(coarse_tokens_file_list[i][:-4]) for i in range(len(coarse_tokens_file_list))]
    fine_token_file_index_list.sort()
    coarse_token_file_index_list.sort()

    file_num = len(fine_tokens_file_list)

    return fine_token_file_index_list, coarse_token_file_index_list, file_num


def json_load_for_df_columns(df, column_list):
    # iterate through all df columns
    for column in column_list:
        df[column] = df[column].apply(json.loads)


def read_all_modified_reading_files():
    file_path_prefix = f"data/modified_gaze_data/{configs.round}/{configs.device}"
    file_list = os.listdir(file_path_prefix)

    df_list_1 = []
    for file_index in range(len(file_list)):
        reading_file_path = f"{file_path_prefix}/{file_list[file_index]}/reading"
        reading_file_list = os.listdir(reading_file_path)
        reading_file_list = [int(i[:-4]) for i in reading_file_list]
        reading_file_list.sort()
        df_list_2 = []
        for reading_file_index in range(len(reading_file_list)):
            df = pd.read_csv(f"{reading_file_path}/{reading_file_list[reading_file_index]}.csv", encoding="utf-8_sig")
            df_list_2.append(df)
        df_list_1.append(df_list_2)

    return df_list_1


def read_all_modified_reading_text_mapping():
    file_path_prefix = f"data/modified_gaze_data/{configs.round}/{configs.device}"
    file_list = os.listdir(file_path_prefix)

    df_list_1 = []
    for file_index in range(len(file_list)):
        text_mapping_file_name = f"{file_path_prefix}/{file_list[file_index]}/text_mapping.csv"
        df = pd.read_csv(text_mapping_file_name, encoding="utf-8_sig")
        df_list_1.append(df)
    return df_list_1


def read_reading_text_mapping_with_weight():
    file_name = f"data/text/{configs.round}/text_sorted_mapping_with_weight.csv"
    df = pd.read_csv(file_name, encoding="utf-8_sig")

    return df


def read_all_token_data(token_type):
    file_path_prefix = f"data/text/{configs.round}/tokens/{token_type}_tokens"
    file_list = os.listdir(file_path_prefix)
    file_list = [int(i[:-4]) for i in file_list]
    file_list.sort()

    df_list_1 = []
    for file_index in range(len(file_list)):
        df = pd.read_csv(f"{file_path_prefix}/{file_list[file_index]}.csv", encoding="utf-8_sig")
        json_load_for_df_columns(df, ["text_unit_component", "row"])
        df_list_1.append(df)
    return df_list_1


def read_token_density_of_token_type(token_type):
    file_path_prefix = f"data/text_density/{configs.round}/{configs.device}/"
    file_list = os.listdir(file_path_prefix)

    df_token_density_list = []
    for file_index in range(len(file_list)):
        df = pd.read_csv(f"{file_path_prefix}/{file_list[file_index]}/{token_type}_token_density.csv", encoding="utf-8_sig")
        json_load_for_df_columns(df, ["text_unit_component", "row", "backward", "forward", "anterior_passage"])
        df_token_density_list.append(df)
    return df_token_density_list


def change_single_quotation_to_double_quotation(x):
    '''
    用于处理部分csv文件在读取后冒号是单引号，导致无法使用json.load()的情况。
    :param x:
    :return:
    '''
    x_changed = x.replace("\'", "\"")
    return x_changed


def read_reading_data():
    file_path = f"data/original_gaze_data/{configs.round}/{configs.device}/"
    file_list = os.listdir(file_path)

    reading_data_list_1 = []
    for file_index in range(len(file_list)):
        reading_file_path = f"{file_path}/{file_list[file_index]}/reading/"
        reading_file_list = os.listdir(reading_file_path)
        reading_data_list_2 = []
        for reading_file_index in range(len(reading_file_list)):
            reading_df = pd.read_csv(f"{reading_file_path}{reading_file_list[reading_file_index]}", encoding="utf-8_sig")
            reading_data_list_2.append(reading_df)
        reading_data_list_1.append(reading_data_list_2)

    return reading_data_list_1


def sort_reading_data_by_para_id(reading_data):
    df_list = []
    for iteration_index in range(len(reading_data)):
        df_group_by_matrix_x = reading_data[iteration_index].groupby("matrix_x")
        for para_id, df_matrix_x in df_group_by_matrix_x:
            df_list.append(df_matrix_x)
    df_list.sort(key=lambda x: x["matrix_x"].iloc[0])
    new_df = pd.concat(df_list, ignore_index=True)
    return new_df


def read_tokens(token_type="fine"):
    file_path = f"data/text/{configs.round}/tokens/{token_type}_tokens/"
    file_list = os.listdir(file_path)
    file_list = [int(i[:-4]) for i in file_list]
    file_list.sort()

    tokens_list = []
    for file_index in range(len(file_list)):
        tokens_df = pd.read_csv(f"{file_path}{file_list[file_index]}.csv", encoding="utf-8_sig")
        tokens_df["row"] = tokens_df["row"].apply(json.loads)
        tokens_df["text_unit_component"] = tokens_df["text_unit_component"].apply(json.loads)
        tokens_df["backward"] = tokens_df["backward"].apply(json.loads)
        tokens_df["forward"] = tokens_df["forward"].apply(json.loads)
        tokens_df["anterior_passage"] = tokens_df["anterior_passage"].apply(json.loads)
        tokens_list.append(tokens_df)

    return tokens_list


def read_text_unit_density():
    file_path = f"data/text_density/{configs.round}/{configs.device}/"
    file_list = os.listdir(file_path)

    text_unit_density_list = []
    for file_index in range(len(file_list)):
        file_name = f"{file_path}{file_list[file_index]}/text_unit_density.csv"
        text_unit_density_df = pd.read_csv(file_name, encoding="utf-8_sig")
        text_unit_density_list.append(text_unit_density_df)

    return text_unit_density_list


def read_calibration_data():
    file_path_prefix = f"data/original_gaze_data/{configs.round}/{configs.device}/"
    file_list = os.listdir(file_path_prefix)

    cali_list = []
    for file_index in range(len(file_list)):
        calibration_file_name = f"{file_path_prefix}/{file_list[file_index]}/calibration.csv"
        df_cali = pd.read_csv(calibration_file_name, encoding="utf-8_sig")
        cali_list.append(df_cali)
    return cali_list
