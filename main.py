# 用于处理实验round 4中的数据。

import os
import numpy as np
import pandas as pd

import analyse_calibration_data
import analyse_reading_data
import configs
import get_reading_density
import read_files
import talk_with_GPT
import temporary_functions
import text_process


def get_token_info():
    # text_process.tokenize_exp_text_sorted() # 用于分词。
    # text_process.add_text_unit_index_to_tokens() # 用于获取分词对应的text_unit的index。
    text_process.add_text_context_to_tokens() # 用于获取每个分词的上下文。


def get_density():
    # get_reading_density.get_text_unit_density() # 获取每个text_unit的密度。
    get_reading_density.get_token_density() # 将text_unit的密度组合成为token的密度。经过这个函数后，token的行会被调整为其所在的真实行（之前都是1-3之间，现在会根据text_unit的位置被调整为1-3或4-6）。


if __name__ == '__main__':
    # temporary_functions.combine_data() # 将round4中出错的两轮实验的数据合并在一个file里。
    # temporary_functions.split_data_in_round_1() # 用来将round1中合并的数据拆分成其他round那样的格式。
    # temporary_functions.split_seeso_data() # 用来将seeso数据拆分成其他round那样的格式。
    # temporary_functions.modify_round_1_reading_data_using_calibration() # 用于将round_1的数据通过calibration估算出的单应性矩阵调整到正确的位置。
    # temporary_functions.get_tokens_of_certain_para("fine", [90, 91, 92, 93, 94]) # 用于获取某个段落的token。
    # temporary_functions.combine_temp_csv("8_21_fine_duration_fourth_from_gpt") # 将text/round_1/weight/temp文件夹中的csv文件合并成一个文件。
    # temporary_functions.ask_gpt_about_density() # 直接把density结果交给GPT，让GPT给出什么prompt是合适的。
    # temporary_functions.ask_gpt_about_article() # 直接把文本文字交给GPT，让GPT对这个文本进行类别分析，并给出哪些内容是最关键的。
    # temporary_functions.ask_gpt_to_subsume_article() # 让gpt把文本进行分析，同时给出5个关键词。
    temporary_functions.read_article_category() # 将刚才的分类结果导入，并用gpt对这些文本类型进行分类。
    # temporary_functions.compute_edge_point_distance() # 计算对于std points，边界上的点到每个std point的距离。

    # analyse_reading_data.render_point_density_hist()
    # analyse_reading_data.match_manual_weight_and_gaze_density()
    # analyse_reading_data.down_sample_reading()
    # analyse_reading_data.add_all_reading("8_21_fine_duration_fourth_from_gpt_90-94.csv")

    # analyse_calibration_data.visualize_original_cali_centroids(f"data/back_up_gaze_data/{configs.round}/reformat_data/20230724_101920/calibration.csv")

    # get_token_info()
    # get_density()
    # talk_with_GPT.save_fine_tune_data()
    # talk_with_GPT.test_gpt_fine_tune_prediction()

    # talk_with_GPT.get_gpt_prediction()
