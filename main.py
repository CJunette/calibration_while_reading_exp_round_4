# 用于处理实验round 4中的数据。

import os
import numpy as np
import pandas as pd

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

    # get_token_info()
    # get_density()

    talk_with_GPT.save_fine_tune_data()
    # talk_with_GPT.test_gpt_prediction()


