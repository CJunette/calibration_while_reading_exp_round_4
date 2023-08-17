import json
import os.path
from multiprocessing import Pool

from matplotlib.patches import Rectangle, Ellipse, Circle
import numpy as np
import openai
import pandas as pd
from matplotlib import pyplot as plt

import configs
import read_files


def start_using_IDE():
    """
    When start python file using IDE, we just need to set proxy with openai.
    :return:
    """
    openai.proxy = 'http://127.0.0.1:10809'


def set_openai():
    # openai.organization = "org-veTDIexYdGbOKcYt8GW4SNOH"
    key_file = open("data/key/OpenAI.txt", "r")
    key = key_file.read()
    openai.api_key = key


def get_gpt_embedding(target_str):
    response = openai.Embedding.create(
        model="text-embedding-ada-002",
        input=[target_str])
    # print("raw response", response)
    response_value = response["data"][0]["embedding"]
    return np.array(response_value)


def prepare_data(token_type):
    token_density_list = read_files.read_token_density_of_token_type(token_type)
    # TODO 之后这里的training和testing的数量可能会发生调整，后续代码也需要修改。
    df_density_for_training = token_density_list[:5]
    df_density_for_testing = token_density_list[5]

    df_training_static = df_density_for_training[0].copy(deep=True)
    temp_density = [[] for _ in range(df_training_static.shape[0])]
    temp_relative_density = [[] for _ in range(df_training_static.shape[0])]
    df_training_static["density"] = temp_density
    df_training_static["relative_density"] = temp_relative_density
    for token_index in range(df_training_static.shape[0]):
        for file_index in range(len(df_density_for_training)):
            df_training_static.iloc[token_index]["density"].append(df_density_for_training[file_index].iloc[token_index]["density"])
            df_training_static.iloc[token_index]["relative_density"].append(df_density_for_training[file_index].iloc[token_index]["relative_density"])

    return df_density_for_training, df_density_for_testing, df_training_static


def prepare_training_text_for_gpt(static_training_df):
    sorted_text = read_files.read_sorted_text()
    df_grouped_by_para_id = static_training_df.groupby("para_id")

    str_list = []

    for para_id, df_para_id in df_grouped_by_para_id:
        full_text = sorted_text[para_id]["text"]
        # 部分full_text最后可能会是一个\n，需要将它去掉
        if full_text[-1] == "\n":
            full_text = full_text[:-1]
        token_str_list = []
        for token_index in range(df_para_id.shape[0]):
            token = df_para_id.iloc[token_index]["tokens"]
            forward = df_para_id.iloc[token_index]["forward"]
            backward = df_para_id.iloc[token_index]["backward"]
            anterior_passage = df_para_id.iloc[token_index]["anterior_passage"]
            density = df_para_id.iloc[token_index]["density"]
            relative_density = df_para_id.iloc[token_index]["relative_density"]
            # 将relative_density转为str，每个数据保留5位有效数字
            relative_density = [int(density * 1e7)/1e7 for density in relative_density]
            # relative_density_str = "[" + ", ".join([f"{float(x): .7f}" for x in relative_density]) + "]"
            # token_str = "{" + f"'token': '{token}', 'token_index': {token_index}, 'backward': {backward}, 'forward': {forward}, 'anterior_passage': '{anterior_passage}', 'relative_density': {relative_density}" + "}"
            token_str = "{" + f"'token': '{token}', 'token_index': {token_index}, 'backward': {backward}, 'forward': {forward}, 'relative_density': {relative_density}" + "}"
            token_str_list.append(token_str)

        token_list_str = "[" + ", \n".join(token_str_list) + "]"
        para_str = "{\n" + f"'full_text': \n'{full_text}',\n 'tokens_and_info': \n'{token_list_str}'" + "\n}"
        str_list.append(para_str)
    return str_list


def prepare_text_for_gpt(df, bool_training=False):
    sorted_text = read_files.read_sorted_text()
    df_grouped_by_para_id = df.groupby("para_id")

    str_list = []

    for para_id, df_para_id in df_grouped_by_para_id:
        full_text = sorted_text[para_id]["text"]
        # 部分full_text最后可能会是一个\n，需要将它去掉
        if full_text[-1] == "\n":
            full_text = full_text[:-1]
        token_str_list = []
        for token_index in range(df_para_id.shape[0]):
            token = df_para_id.iloc[token_index]["tokens"]
            forward = df_para_id.iloc[token_index]["forward"]
            backward = df_para_id.iloc[token_index]["backward"]
            anterior_passage = df_para_id.iloc[token_index]["anterior_passage"]
            density = df_para_id.iloc[token_index]["density"]
            row_position = df_para_id.iloc[token_index]["row_position"]
            distance_from_row_start = df_para_id.iloc[token_index]["start_dist"]
            distance_from_row_end = df_para_id.iloc[token_index]["end_dist"]
            split = df_para_id.iloc[token_index]["split"]

            if bool_training:
                relative_density = df_para_id.iloc[token_index]["relative_density"]
                # 将relative_density转为str，每个数据保留3位有效数字
                relative_density = [int(density * 1e5) / 1e3 for density in relative_density]
                token_str = "{" + f"'token': '{token}', 'token_index': {token_index}, 'distance_to_start': {distance_from_row_start}, 'distance_to_end': {distance_from_row_end}, 'split': {split}, 'relative_density': {relative_density}" + "}"
            else:
                # token_str = "{" + f"'token': '{token}', 'token_index': {token_index}, 'backward': {backward}, 'forward': {forward}, 'anterior_passage': '{anterior_passage}'" + "}"
                token_str = "{" + f"'token': '{token}', 'token_index': {token_index}, 'distance_to_start': {distance_from_row_start}, 'distance_to_end': {distance_from_row_end}, 'split': {split}" + "}"
            token_str_list.append(token_str)

        token_list_str = "[" + ", \n".join(token_str_list) + "]"
        para_str = "{\n" + f"'full_text': \n'{full_text}',\n 'tokens_and_info': \n'{token_list_str}'" + "\n}"
        str_list.append(para_str)
    return str_list


def test_gpt_prediction(token_type="fine"):
    start_using_IDE()
    set_openai()

    # TODO 这里的df_density_for_testing目前只有一个文件，之后可能也会变成多个文件，需要重新修改一下这里相关的代码。
    df_density_for_training, df_density_for_testing, df_training_static = prepare_data(token_type)
    training_str_list = prepare_text_for_gpt(df_training_static, bool_training=True)
    testing_str_list = prepare_text_for_gpt(df_density_for_testing, bool_training=False)

    training_text_num = 2
    response = openai.ChatCompletion.create(
        model="gpt-4-0613",
        messages=[
            {"role": "system", "content": "你是一个分析文本分词与阅读眼动行为的专家。我会为你提供一系列文本的分词，及人在阅读时视线在上面停留的相对时间。"
                                          "你需要学习分词与阅读时间之间存在的关系。之后我会向你提供一些分词，你需要根据你学到的知识给出你对不同分词的停留时间的预测。"
                                          "你需要主要从3个方面理解文本与阅读时间之间的关联：1. 该分词本身的特征（如难度、生僻程度）。2. 该分词与整篇文章所传达的核心意思之间的关联。3. 该分词与上下文之间的关联（即确定上文时，该分词是否容易预测）。"},
            {"role": "user", "content": "我会解释一下我将提供的数据格式及其含义。\n"
                                        "我会向你提供一文本，以及该文本的分词（token）、分词序号（token_index）、到行首的距离（distance_to_start）、到行尾的距离（distance_to_end）、当前词是否被换行分割（split）、相对密度（relative_density）。\n"
                                        # "我会向你提供一文本，以及该文本的分词（token）、前方的3个分词（backward）、后方的3个分词（forward）、相对密度（relative_density）。\n"
                                        "其中分词代表文本全文中的一个分词。分词序号代表该序号在所有分词中的位置。"
                                        "到行首的距离代表该分词到当前行的第一个分词的距离，如果该距离为0，则说明当前分词是改行的第一个分词。到行尾的距离代表该分词到当前行的最后一个分词的距离，如果该距离为0，则说明当前分词是改行的最后一个分词。"
                                        "当前词是否被换行分割代表该分词是否被换行分割，导致一部分在行尾，另一部分在行首。如果该值为1，则说明该分词是被换行分割的。"
                                        "相对密度（relative_density）是一个数组，代表的是不同用户的实现在该分词上停留的时间，相对密度越大代表停留时间越长。同一个文本中，同一用户的相对密度之和为100\n"
                                        # "backward和forward中出现的<SOR>代表一行的开始位置，<EOR>代表一行的结束位置。anterior_passage中出现的<SOP>代表文章的开始位置。\n"
                                        "你需要学习这些文本的分词与相对密度之间的关系，并将这个知识用于之后的任务。\n"
                                        "具体的数据格式如下（我在示例中使用省略号代表数据未完全显示，你在之后的回复后不能这样做）：\n"
                                        "{'full_text': '【领英中国今日停服，包括移动端 App、桌面端网站和微信小程序等】    据IT之家消息，今年5月9日，职场社交平台领英宣布，求职App领英职场将于2023年8月9日起正式停服。'"
                                        "'tokens_and_info':"
                                        "[{'token': '【', 'token_index': 0, 'distance_to_start': 0, 'distance_to_end': 17, 'split': 0, 'density': [0, 0, 0, 0, 0]}, "
                                        "{'token': '领英', 'token_index': 1, 'distance_to_start': 1, 'distance_to_end': 16, 'split': 0, 'density': [0.09, 0.378, 0.595, 0.201, 0.0]}, ...]"},
            {"role": "assistant", "content": "好的，请给我用于学习的文本。"},
            {"role": "user", "content": "\n".join(training_str_list[:training_text_num])},
            {"role": "assistant", "content": "好的，我已经学习到了其中的联系。接下来你向我提供具体的文本及分词时，我会直接向你反馈你对于每个分词的相对停留时间的预测。\n"
                                             "举例。\n"
                                             "如果你提供："
                                             "{'full_text': '【领英'"
                                             "'tokens_and_info': "
                                             "[{'token': '【', 'token_index': 0, 'distance_to_start': 0, 'distance_to_end': 17, 'split': 0}, "
                                             "{'token': '领英', 'token_index': 1, 'distance_to_start': 1, 'distance_to_end': 16, 'split': 0}]"
                                             "我会返回："
                                             "[{'token': '【', 'token_index': 0, density: 0}, "
                                             "{'token': '领英', 'token_index': 1, density: 0.02}]\n"
                                             "你可以直接向我提供数据了。"},
             {"role": "user", "content": f"{testing_str_list[training_text_num]}"}
        ])
    response_value = response["choices"][0]["message"]["content"].strip()
    print(response_value)


def prepare_text_for_fine_tune(df, bool_training=False):
    sorted_text = read_files.read_sorted_text()
    df_grouped_by_para_id = df.groupby("para_id")

    prompt_list = []
    completion_list = []

    for para_id, df_para_id in df_grouped_by_para_id:
        if bool_training:
            if para_id >= configs.fine_tune_training_num:
                break
        else:
            if para_id < configs.fine_tune_training_num:
                continue

        full_text = sorted_text[para_id]["text"]
        # 部分full_text最后可能会是一个\n，需要将它去掉
        if full_text[-1] == "\n":
            full_text = full_text[:-1]

        for token_index in range(df_para_id.shape[0]):
            token = df_para_id.iloc[token_index]["tokens"]
            forward = df_para_id.iloc[token_index]["forward"]
            backward = df_para_id.iloc[token_index]["backward"]
            anterior_passage = df_para_id.iloc[token_index]["anterior_passage"]
            density = df_para_id.iloc[token_index]["density"]
            row_position = df_para_id.iloc[token_index]["row_position"]
            distance_from_row_start = df_para_id.iloc[token_index]["start_dist"]
            distance_from_row_end = df_para_id.iloc[token_index]["end_dist"]
            split = df_para_id.iloc[token_index]["split"]
            row = df_para_id.iloc[token_index]["row"][0]

            prompt_dict = {
                "token": token,
                "token_index": int(token_index),
                "backward": backward,
                "forward": forward,
                "row": int(row),
                "split": int(split),
                "anterior_passage": anterior_passage,
                "distance_from_row_end": int(distance_from_row_end),
                "distance_from_row_start": int(distance_from_row_start)
            }
            prompt_list.append(prompt_dict)

            relative_density = df_para_id.iloc[token_index]["relative_density"]
            # 将relative_density转为str，每个数据保留3位有效数字
            relative_density_mean = np.mean(relative_density)
            relative_density_std = np.std(relative_density)
            density_mean = np.mean(density)
            density_std = np.std(density)
            completion_dict = {
                "relative_density_mean": relative_density_mean,
                "relative_density_std": relative_density_std,
                "density_mean": density_mean,
                "density_std": density_std
            }
            completion_list.append(completion_dict)

    return prompt_list, completion_list


def save_fine_tune_data(token_type="fine"):
    def save_data(data_path, prompt_list, completion_list):
        with open(data_path, "w") as f:
            for i in range(len(prompt_list)):
                prompt_str = json.dumps(prompt_list[i])
                completion_str = json.dumps(completion_list[i])
                data = {
                    "prompt": prompt_str + "\n\n###\n\n",
                    "completion": " " + completion_str + "\n\n###\n\n"
                }

                f.write(json.dumps(data) + "\n")

    df_density_for_training, df_density_for_testing, df_training_static = prepare_data(token_type)
    training_prompt_list, training_completion_list = prepare_text_for_fine_tune(df_training_static, bool_training=True)
    testing_prompt_list, test_completion_list = prepare_text_for_fine_tune(df_density_for_testing, bool_training=False)
    validation_prompt_list, validation_completion_list = prepare_text_for_fine_tune(df_training_static, bool_training=False)

    training_data_path = "data/fine_tune/training_data/"
    if not os.path.exists(training_data_path):
        os.makedirs(os.path.dirname(training_data_path))
    testing_data_path = "data/fine_tune/testing_data/"
    if not os.path.exists(testing_data_path):
        os.makedirs(os.path.dirname(testing_data_path))
    validation_data_path = "data/fine_tune/validation_data/"
    if not os.path.exists(validation_data_path):
        os.makedirs(os.path.dirname(validation_data_path))

    training_data_name = f"{training_data_path}training_data_ver_{configs.fine_tune_ver}.jsonl"
    testing_data_name = f"{testing_data_path}testing_data_ver_{configs.fine_tune_ver}.jsonl"
    validation_data_name = f"{validation_data_path}validation_data_ver_{configs.fine_tune_ver}.jsonl"

    save_data(training_data_name, training_prompt_list, training_completion_list)
    save_data(testing_data_name, testing_prompt_list, test_completion_list)
    save_data(validation_data_name, validation_prompt_list, validation_completion_list)
    # f.write(data_str + "\n")
    # 执行完成后，需要在terminal中执行以下命令，用openai自带的检验算法检验文档是否符合训练数据的要求。
    # openai tools fine_tunes.prepare_data -f <training_file_path>
    # 接下来需要配置环境变量。
    # $Env:OPENAI_API_KEY="your_api_key"
    # 然后设置代理。
    # $proxy='http://127.0.0.1:10809';$ENV:HTTP_PROXY=$proxy;$ENV:HTTPS_PROXY=$proxy
    # 然后按openai官网的提示继续创建模型即可。
    # openai api fine_tunes.create -t <TRAIN_FILE_ID_OR_PATH> -m <BASE_MODEL>
    # 以上内容的网页参考：https://platform.openai.com/docs/guides/fine-tuning。


def get_gpt_prediction(test_data_list, validation_data_list, test_data_index):
    set_openai()
    start_using_IDE()

    while True:
        test_prompt_str = test_data_list[test_data_index]["prompt"]
        test_prompt = json.loads(test_prompt_str[:-7])
        test_completion_str = test_data_list[test_data_index]["completion"][:-7]
        test_completion = json.loads(test_completion_str)

        models = openai.Model.list()

        response = openai.Completion.create(
            model=configs.fine_tune_model_name,
            prompt=test_prompt_str,
            max_tokens=100,
        )

        print(test_data_index, len(test_data_list))

        validation_completion_str = validation_data_list[test_data_index]["completion"][:-7]
        validation_completion = json.loads(validation_completion_str)

        gpt_completion_str = response["choices"][0]["text"].split("\n\n###\n\n")[0]

        try:
            gpt_completion = json.loads(gpt_completion_str)
            # 保证所需数值都在返回的结果中。
            if "relative_density_mean" in gpt_completion and "relative_density_std" in gpt_completion and "density_mean" in gpt_completion and "density_std" in gpt_completion:
                # 保证不会出现过于夸张的数值。
                if gpt_completion["density_mean"] < 200 and gpt_completion["density_std"] < 200:
                    return test_prompt, test_completion, gpt_completion, validation_completion
                else:
                    print(f"error in {test_data_index}, {'wrong mean or std'}, {gpt_completion_str}")
            else:
                print(f"error in {test_data_index}, {'information loss'}, {gpt_completion_str}")
        except Exception as e:
            print(f"error in {test_data_index}, {e}, {gpt_completion_str}")


def save_gpt_fine_tune_prediction():
    test_data_file_path = f"data/fine_tune/testing_data/testing_data_ver_{configs.fine_tune_ver}.jsonl"
    test_data_list = []
    with open(test_data_file_path, "r") as f:
        for line in f:
            entry = json.loads(line)
            test_data_list.append(entry)

    validation_data_file_path = f"data/fine_tune/validation_data/validation_data_ver_{configs.fine_tune_ver}.jsonl"
    validation_data_list = []
    with open(validation_data_file_path, "r") as f:
        for line in f:
            entry = json.loads(line)
            validation_data_list.append(entry)

    prompt_list = []
    test_completion_list = []
    gpt_completion_list = []
    validation_completion_list = []
    test_data_index = 0

    args_list = []
    while test_data_index < len(test_data_list):
        args_list.append((test_data_list, validation_data_list, test_data_index))
        test_data_index += 1

    with Pool(8) as p:
        result_list = p.starmap(get_gpt_prediction, args_list)

    for result_index in range(len(result_list)):
        test_prompt, test_completion, gpt_completion, validation_completion = result_list[result_index]
        prompt_list.append(test_prompt)
        test_completion_list.append(test_completion)
        gpt_completion_list.append(gpt_completion)
        validation_completion_list.append(validation_completion)

    df = pd.DataFrame({
        "prompt": prompt_list,
        "test_completion": test_completion_list,
        "gpt_completion": gpt_completion_list,
        "validation_completion": validation_completion_list
    })

    df["prompt"].apply(json.dumps)
    df["test_completion"].apply(json.dumps)
    df["gpt_completion"].apply(json.dumps)
    df["validation_completion"].apply(json.dumps)

    save_path = f"data/fine_tune/{configs.fine_tune_model_name.replace(':', '_')}/"
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    save_name = f"{save_path}result_005.csv"
    df.to_csv(save_name, index=False, encoding="utf-8_sig")


def change_quotation_mark(df):
    df["prompt"] = df["prompt"].apply(read_files.change_single_quotation_to_double_quotation).apply(json.loads)
    df["test_completion"] = df["test_completion"].apply(read_files.change_single_quotation_to_double_quotation).apply(json.loads)
    df["gpt_completion"] = df["gpt_completion"].apply(read_files.change_single_quotation_to_double_quotation).apply(json.loads)
    df["validation_completion"] = df["validation_completion"].apply(read_files.change_single_quotation_to_double_quotation).apply(json.loads)


def read_and_visualize_gpt_prediction():
    df = pd.read_csv(f"data/fine_tune/{configs.fine_tune_model_name.replace(':', '_')}/result_001.csv", encoding="utf-8_sig")
    change_quotation_mark(df)

    fig, axes = plt.subplots(2, 1)
    axes[0].set_xlim(-1, df.shape[0] + 1)
    axes[0].set_ylim(-0.0075, 0.015)
    axes[1].set_xlim(-1, df.shape[0] + 1)
    axes[1].set_ylim(-1, 100)
    plt.rcParams['font.sans-serif'] = ['SimSun']
    plt.subplots_adjust(left=0.05, right=0.95, bottom=0.05, top=0.95, wspace=0.1, hspace=0.1)
    for token_index in range(df.shape[0]):
        token = df["prompt"][token_index]["token"]
        test_relative_density_mean = df["test_completion"][token_index]["relative_density_mean"]
        test_relative_density_std = df["test_completion"][token_index]["relative_density_std"]
        gpt_relative_density_mean = df["gpt_completion"][token_index]["relative_density_mean"]
        gpt_relative_density_std = df["gpt_completion"][token_index]["relative_density_std"]
        validation_relative_density_mean = df["validation_completion"][token_index]["relative_density_mean"]
        validation_relative_density_std = df["validation_completion"][token_index]["relative_density_std"]
        gpt_relative_density_rect = Rectangle((token_index, gpt_relative_density_mean - gpt_relative_density_std), 0.03, gpt_relative_density_std * 2, color="red", alpha=0.5)
        validation_relative_density_rect = Rectangle((token_index, validation_relative_density_mean - validation_relative_density_std), 0.02, validation_relative_density_std * 2, color="green", alpha=0.5)
        axes[0].scatter(token_index, test_relative_density_mean, s=5, color="blue")
        axes[0].add_patch(gpt_relative_density_rect)
        axes[0].add_patch(validation_relative_density_rect)
        axes[0].text(token_index, -0.005, token, ha="center", va="center", fontsize=8)

        test_density_mean = df["test_completion"][token_index]["density_mean"]
        test_density_std = df["test_completion"][token_index]["density_std"]
        gpt_density_mean = df["gpt_completion"][token_index]["density_mean"]
        gpt_density_std = df["gpt_completion"][token_index]["density_std"]
        validation_density_mean = df["validation_completion"][token_index]["density_mean"]
        validation_density_std = df["validation_completion"][token_index]["density_std"]
        gpt_density_rect = Rectangle((token_index, gpt_density_mean - gpt_density_std), 0.03, gpt_density_std * 2, color="red", alpha=0.5)
        validation_density_rect = Rectangle((token_index, validation_density_mean - validation_density_std), 0.02, validation_density_std * 2, color="green", alpha=0.5)
        axes[1].scatter(token_index, test_density_mean, s=5, color="blue")
        axes[1].add_patch(gpt_density_rect)
        axes[1].add_patch(validation_density_rect)
        axes[1].text(token_index, -0.5, token, ha="center", va="center", fontsize=8)

    plt.show()


def check_gpt_fine_tune_prediction_stability():
    df_list = []
    result_file_path = f"data/fine_tune/{configs.fine_tune_model_name.replace(':', '_')}/"
    file_list = os.listdir(result_file_path)
    for file_index in range(len(file_list)):
        file_name = f"{result_file_path}{file_list[file_index]}"
        if file_name.endswith(".csv"):
            df = pd.read_csv(f"{file_name}", encoding="utf-8_sig")
            change_quotation_mark(df)
            df_list.append(df)

    color_list = ["red", "blue", "green", "yellow", "black", "purple", "orange", "pink", "gray", "brown", "cyan", "magenta"]

    relative_density_mean_std_list = []
    relative_density_std_std_list = []
    density_mean_std_list = []
    density_std_std_list = []
    for token_index in range(df_list[0].shape[0]):
        relative_density_mean_list = []
        relative_density_std_list = []
        density_mean_list = []
        density_std_list = []
        for file_index in range(len(df_list)):
            gpt_relative_density_mean = df_list[file_index]["gpt_completion"][token_index]["relative_density_mean"]
            gpt_relative_density_std = df_list[file_index]["gpt_completion"][token_index]["relative_density_std"]
            gpt_density_mean = df_list[file_index]["gpt_completion"][token_index]["density_mean"]
            gpt_density_std = df_list[file_index]["gpt_completion"][token_index]["density_std"]

            relative_density_mean_list.append(gpt_relative_density_mean)
            relative_density_std_list.append(gpt_relative_density_std)
            density_mean_list.append(gpt_density_mean)
            density_std_list.append(gpt_density_std)

        relative_density_mean_std_list.append(np.std(relative_density_mean_list))
        relative_density_std_std_list.append(np.std(relative_density_std_list))
        density_mean_std_list.append(np.std(density_mean_list))
        density_std_std_list.append(np.std(density_std_list))

    print("std of relative density mean", np.mean(relative_density_mean_std_list))
    print("std of relative density std", np.mean(relative_density_std_std_list))
    print("std of density mean", np.mean(density_mean_std_list))
    print("std of density std", np.mean(density_std_std_list))

    # 修改上述代码，创建左右2个图像，一边显示relative_density，一边显示density。
    fig, axes = plt.subplots(2, 1)
    axes[0].set_xlim(-1, df_list[0].shape[0] + 1)
    axes[0].set_ylim(-0.0075, 0.015)
    axes[1].set_xlim(-1, df_list[0].shape[0] + 1)
    axes[1].set_ylim(-1, 100)
    plt.rcParams['font.sans-serif'] = ['SimSun']
    plt.subplots_adjust(left=0.05, right=0.95, bottom=0.05, top=0.95, wspace=0.1, hspace=0.1)
    for df_index in range(len(df_list)):
        df = df_list[df_index]
        for token_index in range(df.shape[0]):
            token = df["prompt"][token_index]["token"]
            gpt_relative_density_mean = df["gpt_completion"][token_index]["relative_density_mean"]
            gpt_relative_density_std = df["gpt_completion"][token_index]["relative_density_std"]
            get_density_mean = df["gpt_completion"][token_index]["density_mean"]
            get_density_std = df["gpt_completion"][token_index]["density_std"]
            gpt_relative_density_rect = Rectangle((token_index, gpt_relative_density_mean - gpt_relative_density_std), 0.03, gpt_relative_density_std * 2, color=color_list[df_index], alpha=0.1)
            gpt_density_rect = Rectangle((token_index, get_density_mean - get_density_std), 0.02, get_density_std * 2, color=color_list[df_index], alpha=0.1)
            axes[0].add_patch(gpt_relative_density_rect)
            axes[1].add_patch(gpt_density_rect)
            axes[0].text(token_index, -0.005, token, ha="center", va="center", fontsize=8)
            axes[1].text(token_index, -0.1, token, ha="center", va="center", fontsize=8)

    plt.show()


def test_gpt_fine_tune_prediction():
    # save_gpt_fine_tune_prediction() # 保存gpt预测结果。
    # check_gpt_fine_tune_prediction_stability() # 检查多次返回的预测结果是否稳定。
    read_and_visualize_gpt_prediction() # 根据某次返回结果，检查其与实际结果是否接近。


def main():
    test_gpt_fine_tune_prediction()


if __name__ == "__main__":
    main()
