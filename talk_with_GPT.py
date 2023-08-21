import json
import os.path
import re
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


'''--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------'''
'''GPT API相关'''


def create_df_statistic_single_pool(token_index, df_list):
    density_list = []
    relative_density_list = []
    for file_index in range(len(df_list)):
        print(token_index, file_index)
        density_list.append(df_list[file_index].iloc[token_index]["density"])
        relative_density_list.append(df_list[file_index].iloc[token_index]["relative_density"])

    return density_list, relative_density_list


def create_df_statistic(df_list):
    df_statistic = df_list[0].copy(deep=True)
    temp_density = [[] for _ in range(df_statistic.shape[0])]
    temp_relative_density = [[] for _ in range(df_statistic.shape[0])]
    df_statistic["density"] = temp_density
    df_statistic["relative_density"] = temp_relative_density
    # for debug.
    # for token_index in range(df_statistic.shape[0]):
    #     density_list, relative_density_list = create_df_statistic_single_pool(token_index, df_list)
    #     df_statistic.at[token_index, "density"] = density_list
    #     df_statistic.at[token_index, "relative_density"] = relative_density_list

    # 多线程。
    args_list = []
    for token_index in range(df_statistic.shape[0]):
        args = [token_index, df_list]
        args_list.append(args)
    with Pool(16) as p:
        results = p.starmap(create_df_statistic_single_pool, args_list)
    for token_index in range(df_statistic.shape[0]):
        df_statistic.at[token_index, "density"] = results[token_index][0]
        df_statistic.at[token_index, "relative_density"] = results[token_index][1]

    return df_statistic


def prepare_data(token_type):
    # 调整为多进程处理。
    token_density_list = read_files.read_token_density_of_token_type(token_type)
    # TODO 之后这里的training和testing的数量可能会发生调整，后续代码也需要修改。
    df_density_for_training = token_density_list[:configs.fine_tune_training_file_num]
    df_density_for_testing = token_density_list[configs.fine_tune_training_file_num:]

    df_training_statistic = create_df_statistic(df_density_for_training)
    df_testing_statistic = create_df_statistic(df_density_for_testing)

    return df_density_for_training, df_density_for_testing, df_training_statistic, df_testing_statistic
    # return df_density_for_training, df_density_for_testing


def prepare_text_for_gpt(df_list, bool_training=False):
    str_list = []

    for file_index in range(len(df_list)):
        df = df_list[file_index]
        sorted_text = read_files.read_sorted_text()
        df_grouped_by_para_id = df.groupby("para_id")

        for para_id, df_para_id in df_grouped_by_para_id:
            full_text = sorted_text[para_id]["text"]
            full_text = full_text.replace("\n", "")

            token_list = []
            token_index_list = []
            distance_start_list = []
            distance_end_list = []
            split_list = []
            relative_density_mean_list = []
            relative_density_std_list = []

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
                relative_density = df_para_id.iloc[token_index]["relative_density"]

                token_list.append(token)
                token_index_list.append(token_index)
                distance_start_list.append(distance_from_row_start)
                distance_end_list.append(distance_from_row_end)
                split_list.append(split)

                relative_density_mean = int(np.mean(relative_density) * 1e6) / 1e4
                relative_density_std = int(np.std(relative_density) * 1e6) / 1e4
                relative_density_mean_list.append(relative_density_mean)
                relative_density_std_list.append(relative_density_std)

            token_str = "[" + ",".join(token_list) + "]"
            token_index_str = "[" + ",".join([str(i) for i in token_index_list]) + "]"
            distance_start_str = "[" + ",".join([str(i) for i in distance_start_list]) + "]"
            distance_end_str = "[" + ",".join([str(i) for i in distance_end_list]) + "]"
            split_str = "[" + ",".join([str(i) for i in split_list]) + "]"
            relative_density_mean_str = "[" + ",".join([str(i) for i in relative_density_mean_list]) + "]"
            relative_density_std_str = "[" + ",".join([str(i) for i in relative_density_std_list]) + "]"

            if bool_training:
                para_str = "{" + f"'full_text': {full_text}, " \
                                 f"'token_list': {token_str}, " \
                                 f"'token_index_list': {token_index_str}, " \
                                 f"'dist_start_list': {distance_start_str}, " \
                                 f"'dist_end_list': {distance_end_str}, " \
                                 f"'split_list': {split_str}, " \
                                 f"'rela_den_mean_list': {relative_density_mean_str}, " \
                                 f"'rela_den_std_list': {relative_density_std_str}" + "}"
            else:
                para_str = "{" + f"'full_text': {full_text}, " \
                                 f"'token_list': {token_str}, " \
                                 f"'token_index_list': {token_index_str}, " \
                                 f"'dist_start_list': {distance_start_str}, " \
                                 f"'dist_end_list': {distance_end_str}, " \
                                 f"'split_list': {split_str}" + "}"

            str_list.append(para_str)
    return str_list


def _save_gpt_prediction(token_type):
    start_using_IDE()
    set_openai()

    # TODO 这里的df_density_for_testing目前只有一个文件，之后可能也会变成多个文件，需要重新修改一下这里相关的代码。
    # df_density_for_training, df_density_for_testing, df_training_statistic, df_testing_statistic = prepare_data(token_type)
    # training_str_list = prepare_text_for_gpt(df_density_for_training, bool_training=True)
    # testing_str_list = prepare_text_for_gpt(df_density_for_testing, bool_training=False)

    token_list = read_files.read_tokens()
    target_para_index = [90, 91, 92, 93, 94]
    target_token_list = []
    for para_index in target_para_index:
        target_token_list.append(token_list[para_index])

    for target_index in range(len(target_token_list)):
        df = target_token_list[target_index]
        full_text = ""
        for token_index in range(df.shape[0]):
            full_text += df.iloc[token_index]["tokens"]
            if token_index + 1 < df.shape[0] - 1 and df.iloc[token_index]["row"] != df.iloc[token_index + 1]["row"]:
                cur_row = df.iloc[token_index]["row"][0]
                next_row = df.iloc[token_index + 1]["row"][0]
                for row_index in range(next_row - cur_row):
                    full_text += "\n"
                if df.iloc[token_index]["split"] == 1:
                    next_token = df.iloc[token_index + 1]["tokens"]
                    # take away the (.*) in next token
                    next_token = re.sub(r"\(.+\)", "", next_token)
                    df["tokens"][token_index + 1] = next_token
            token_list = df["tokens"].tolist()
            token_list_str = "[" + ",".join([f"({i}, '{token_list[i]}')" for i in range(len(token_list))]) + "]"
            prompt_str = "{" + f"'full_text': '{full_text}', 'token_list': {token_list_str}" + "}"

            response = openai.ChatCompletion.create(
                model="gpt-4-0613",
                messages=[
                    {"role": "system", "content": "你是一个分析文本与分词的专家。我会为你提供一段文本与它对应分词，你需要告诉我三个指标，一是每个分词与文章的关联程度，二是每个分词本身的阅读难度，三是这个分词在前后文中的阅读难度。"},
                    {"role": "user", "content": "我会解释一下我将提供的数据及其含义。\n"
                                                "我会向你提供一句话的完整文本与其对应的分词及每个分词序号。如{'full_text': '今天天气很好', 'token_list': [(0, '今天'), (1, '天气'), (2, '很好')]}\n"
                                                "full_text中的'\n'代表换行。\n"
                                                "你需要给我的返回是一个列表，其中包含8个成分：分词序号，分词，关联程度打分（rele），关联程度打分确定度（rele_cred），分词本身阅读难度打分（diff），分词本身阅读难度打分确定度（diff_cred），分词上下文阅读难度打分（diff_context），分词上下文阅读难度打分确定度（diff_context_cred）。"
                                                "关联程度打分、分词本身阅读难度打分、分词上下文阅读难度打分的分数都是从1到5，分数低代表关联度低或难度低，分数高代表关联度高或难度高。确信度打分为0到1，分数低代表不确定，分数高代表确定。\n"
                                                "关联程度代表一个分词与这篇文章想要传达的核心意思之间的关联有多大；分词本身难度，代表仅看这个分词，它有多难理解，如一些专业名词的难度就很高（5分），而一些助词、标点，难度就很低（1分）；分词上下文难度，指放在上下文中，一个分词是否容易理解，如叠词单看时容易理解，但放在上下文中则会引发与上下文内容相关的额外的思考。"
                                                "3个关于打分的确定度，可以被理解为不同人是否都会认可你给出的打分，当确定度高时，代表大部分人都会认可你的打分；当确定度低时，代表只有少部分人会认可你的打分。\n"                                                
                                                "注意，打分过程中，不可增加、删除或修改token。你需要返回所有token的打分结果，打分时间可以较长，但绝对不可以只返回部分结果。"},
                    {"role": "user", "content": "接下来我们模拟一下输入和返回的结果。本次模拟中打分的结果不具有参考性。\n"},
                    {"role": "user", "content": "。{'full_text': '今天天气很好', 'token_list': [(0, '今天'), (1, '天气'), (2, '很好')]}"},
                    {"role": "assistant", "content": "{'return_list': [{'index': 0, 'token': '今天', 'rele': 4, 'rele_cred': 0.8, 'diff': 2, 'diff_cred': 0.9, 'diff_context': '3', 'diff_context_cred': 0.7}], "
                                                     "[{'index': 1, 'token': '天气', 'rele': 5, 'rele_cred': 0.8, 'diff': 2, 'diff_cred': 0.9, 'diff_context': '3', 'diff_context_cred': 0.7}], "
                                                     "[{'index': 2, 'token': '很好', 'rele': 4, 'rele_cred': 0.8, 'diff': 1, 'diff_cred': 0.9, 'diff_context': '2', 'diff_context_cred': 0.7}]}"},
                    {"role": "user", "content": "你做得很好，接下来请按相同的格式给我返回，但之后的打分需要是正确的打分。\n"},
                    {"role": "assistant", "content": "好的，我已经理解你的任务了。你可以给我你需要打分的文本了。\n"},
                    {"role": "user", "content": f"{prompt_str}"}],
            )

            response_str = response["choices"][0]["message"]["content"].strip()
            print(response_str)
            print()


'''--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------'''
'''FINE TUNE保存训练集相关'''


def prepare_text_for_fine_tune_single_pool(para_id, token_index, sorted_text, df_para_id):
    print(para_id, token_index)
    full_text = sorted_text[para_id]["text"]
    # 部分full_text最后可能会是一个\n，需要将它去掉
    if full_text[-1] == "\n":
        full_text = full_text[:-1]
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
    return prompt_dict, completion_dict


def prepare_text_for_fine_tune(df, bool_training=False):
    sorted_text = read_files.read_sorted_text()

    prompt_list = []
    completion_list = []
    df_grouped_by_para_id = df.groupby("para_id")

    # for debug.
    # for para_id, df_para_id in df_grouped_by_para_id:
    #     print(para_id)
    #     if bool_training:
    #         if para_id >= configs.fine_tune_training_para_num:
    #             break
    #     else:
    #         if para_id < configs.fine_tune_training_para_num:
    #             continue
    #
    #     full_text = sorted_text[para_id]["text"]
    #     # 部分full_text最后可能会是一个\n，需要将它去掉
    #     if full_text[-1] == "\n":
    #         full_text = full_text[:-1]
    #
    #     for token_index in range(df_para_id.shape[0]):
    #         token = df_para_id.iloc[token_index]["tokens"]
    #         forward = df_para_id.iloc[token_index]["forward"]
    #         backward = df_para_id.iloc[token_index]["backward"]
    #         anterior_passage = df_para_id.iloc[token_index]["anterior_passage"]
    #         density = df_para_id.iloc[token_index]["density"]
    #         row_position = df_para_id.iloc[token_index]["row_position"]
    #         distance_from_row_start = df_para_id.iloc[token_index]["start_dist"]
    #         distance_from_row_end = df_para_id.iloc[token_index]["end_dist"]
    #         split = df_para_id.iloc[token_index]["split"]
    #         row = df_para_id.iloc[token_index]["row"][0]
    #
    #         prompt_dict = {
    #             "token": token,
    #             "token_index": int(token_index),
    #             "backward": backward,
    #             "forward": forward,
    #             "row": int(row),
    #             "split": int(split),
    #             "anterior_passage": anterior_passage,
    #             "distance_from_row_end": int(distance_from_row_end),
    #             "distance_from_row_start": int(distance_from_row_start)
    #         }
    #         prompt_list.append(prompt_dict)
    #
    #         relative_density = df_para_id.iloc[token_index]["relative_density"]
    #         # 将relative_density转为str，每个数据保留3位有效数字
    #         relative_density_mean = np.mean(relative_density)
    #         relative_density_std = np.std(relative_density)
    #         density_mean = np.mean(density)
    #         density_std = np.std(density)
    #         completion_dict = {
    #             "relative_density_mean": relative_density_mean,
    #             "relative_density_std": relative_density_std,
    #             "density_mean": density_mean,
    #             "density_std": density_std
    #         }
    #         completion_list.append(completion_dict)

    args_list = []
    for para_id, df_para_id in df_grouped_by_para_id:
        if bool_training:
            if para_id >= configs.fine_tune_training_para_num:
                break
        else:
            if para_id < configs.fine_tune_training_para_num:
                continue
        for token_index in range(df_para_id.shape[0]):
            args_list.append([para_id, token_index, sorted_text, df_para_id])
    with Pool(16) as p:
        result = p.starmap(prepare_text_for_fine_tune_single_pool, args_list)

    for result_index in range(len(result)):
        prompt_list.append(result[result_index][0])
        completion_list.append(result[result_index][1])

    return prompt_list, completion_list


def _save_fine_tune_data(token_type):
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

    df_density_for_training, df_density_for_testing, df_training_statistic, df_testing_statistic = prepare_data(token_type)
    # df_density_for_training, df_density_for_testing = prepare_data(token_type)
    training_prompt_list, training_completion_list = prepare_text_for_fine_tune(df_training_statistic, bool_training=True)
    testing_prompt_list, test_completion_list = prepare_text_for_fine_tune(df_testing_statistic, bool_training=False)
    validation_prompt_list, validation_completion_list = prepare_text_for_fine_tune(df_training_statistic, bool_training=False)

    training_data_path = "data/fine_tune/training_data/"
    if not os.path.exists(training_data_path):
        os.makedirs(os.path.dirname(training_data_path))
    testing_data_path = "data/fine_tune/testing_data/"
    if not os.path.exists(testing_data_path):
        os.makedirs(os.path.dirname(testing_data_path))
    validation_data_path = "data/fine_tune/validation_data/"
    if not os.path.exists(validation_data_path):
        os.makedirs(os.path.dirname(validation_data_path))

    training_data_name = f"{training_data_path}{configs.round}_{token_type}_training_data_ver_{configs.fine_tune_ver}.jsonl"
    testing_data_name = f"{testing_data_path}{configs.round}_{token_type}_testing_data_ver_{configs.fine_tune_ver}.jsonl"
    validation_data_name = f"{validation_data_path}{configs.round}_{token_type}_validation_data_ver_{configs.fine_tune_ver}.jsonl"

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


'''--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------'''
'''FINE TUNE结果验证相关'''


def get_fine_tune_prediction(test_data_list, validation_data_list, test_data_index):
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


def save_gpt_fine_tune_prediction(token_type, save_file_index=None):
    test_data_file_path = f"data/fine_tune/testing_data/{configs.round}_{token_type}_testing_data_ver_{configs.fine_tune_ver}.jsonl"
    test_data_list = []
    with open(test_data_file_path, "r") as f:
        for line in f:
            entry = json.loads(line)
            test_data_list.append(entry)

    validation_data_file_path = f"data/fine_tune/validation_data/{configs.round}_{token_type}_validation_data_ver_{configs.fine_tune_ver}.jsonl"
    validation_data_list = []
    with open(validation_data_file_path, "r") as f:
        for line in f:
            entry = json.loads(line)
            validation_data_list.append(entry)

    # 保存的数据中没有包含para_id这类的信息，因此需要通过token_density的数据来获取。
    df_token_density = read_files.read_token_density_of_token_type(token_type)[0]
    df_token_density_for_test = df_token_density[df_token_density["para_id"] >= configs.fine_tune_training_para_num]
    para_id_list = df_token_density_for_test["para_id"].tolist()

    prompt_list = []
    test_completion_list = []
    gpt_completion_list = []
    validation_completion_list = []
    test_data_index = 0

    args_list = []
    while test_data_index < len(test_data_list):
        if para_id_list[test_data_index] not in [90, 91, 92, 93, 94]:
            test_data_index += 1
            continue # 只选择部分数据结果保存，以加快效率。# FIXME 不用时可以注释掉。
        args_list.append((test_data_list, validation_data_list, test_data_index))
        test_data_index += 1

    with Pool(16) as p:
        result_list = p.starmap(get_fine_tune_prediction, args_list)

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
    if save_file_index is None:
        save_name = f"{save_path}{configs.round}_{token_type}_result_001.csv"
    else:
        save_name = f"{save_path}{configs.round}_{token_type}_result_{str(save_file_index).zfill(3)}.csv"

    df.to_csv(save_name, index=False, encoding="utf-8_sig")


def change_quotation_mark(df):
    df["prompt"] = df["prompt"].apply(read_files.change_single_quotation_to_double_quotation).apply(json.loads)
    df["test_completion"] = df["test_completion"].apply(read_files.change_single_quotation_to_double_quotation).apply(json.loads)
    df["gpt_completion"] = df["gpt_completion"].apply(read_files.change_single_quotation_to_double_quotation).apply(json.loads)
    df["validation_completion"] = df["validation_completion"].apply(read_files.change_single_quotation_to_double_quotation).apply(json.loads)


def read_and_visualize_gpt_prediction(token_type):
    df = pd.read_csv(f"data/fine_tune/{configs.fine_tune_model_name.replace(':', '_')}/{configs.round}_{token_type}_result_001.csv", encoding="utf-8_sig")
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


'''--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------'''
'''外界调用函数'''


def save_fine_tune_data(token_type="fine"):
    _save_fine_tune_data(token_type)


def test_gpt_fine_tune_prediction(token_type="fine"):
    # for index in range(1, 11):
    #     save_gpt_fine_tune_prediction(token_type, index) # 保存gpt预测结果。
    check_gpt_fine_tune_prediction_stability() # 检查多次返回的预测结果是否稳定。
    # read_and_visualize_gpt_prediction(token_type) # 根据某次返回结果，检查其与实际结果是否接近。


def get_gpt_prediction(token_type="fine"):
    _save_gpt_prediction(token_type)

