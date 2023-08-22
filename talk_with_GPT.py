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


def send_single_rate_request(tokens, reading_analyse_dict, full_text,  prompt_str, index_str, index_credibility_str, index_explain_str, index_quote_explain_str):
    start_using_IDE()
    set_openai()

    while True:
        response = openai.ChatCompletion.create(
            model="gpt-4-0613",
            messages=[
                {"role": "system", "content": f"你是一个分析文本与分词的专家。我会为你提供一段文本与它对应分词，你需要告诉我一个指标：{index_str}。"},
                {"role": "user", "content": "你需要打分的文本如下（文本中的\\n代表换行，你可以忽略）：\n"
                                            f"{full_text}\n"
                                            f"在之前的交流中，你告诉我了一些关于这篇文本的信息：\n"
                                            f"其类型为：{reading_analyse_dict['type']}\n"
                                            f"其关键词为：{reading_analyse_dict['keywords']}\n"
                                            f"应该关注的内容为：{reading_analyse_dict['what_to_focus']}"
                                            f"具体该关注的分词为：{reading_analyse_dict['focus_words']}"
                                            f"不该关注的内容为：{reading_analyse_dict['what_to_ignore']}"
                                            f"具体不该关注的分词为：{reading_analyse_dict['ignored_words']}"},
                {"role": "assistant", "content": "好的，我理解你对这篇文章的描述了。接下来请告诉我你需要我执行的任务细节与我们间通信的格式。"},
                {"role": "user", "content": "我会解释一下我将提供的数据及其含义。\n"
                                            "我会向你提供之前文本的分词及每个分词序号。如{'token_list': [(0, '今天'), (1, '天气'), (2, '很好')]}\n"
                                            f"你需要给我的返回是一个列表，其中包含4个成分：分词序号，分词，{index_str}，{index_credibility_str}。"
                                            f"{index_explain_str}\n"
                                            f"{index_quote_explain_str}\n"
                                            "关于打分的确定度，可以被理解为不同人是否都会认可你给出的打分，当确定度高时，代表大部分人都会认可你的打分；当确定度低时，代表只有少部分人会认可你的打分。\n"
                                            "注意，打分过程中，不可增加、删除或修改token。你需要返回所有token的打分结果，打分时间可以较长，但绝对不可以只返回部分结果。"},
                {"role": "user", "content": "接下来我们模拟一下输入和返回的结果。本次模拟中打分的结果不具有参考性。模拟采用的文本为'今天天气很好'。\n"},
                {"role": "user", "content": "。{'token_list': [[0, '今天'], [1, '天气'], [2, '很好']]}"},
                {"role": "assistant", "content": "{'return_list': [[0, '今天', 4, 0.8], [1, '天气', 5, 0.8], [2, '很好', 5, 0.8]]}"},
                {"role": "user", "content": "你做得很好，接下来请按相同的格式给我返回，但之后的打分需要是正确的打分。\n"},
                {"role": "assistant", "content": "好的，我已经理解你的任务了。你可以给我你需要打分的文本了。\n"},
                {"role": "user", "content": f"{prompt_str}"}],
        )

        response_str = response["choices"][0]["message"]["content"].strip()
        print(response_str)

        try:
            response_value = json.loads(response_str.replace("\'", "\""))["return_list"]
            bool_check = False
            for token_index in range(len(tokens)):
                if tokens[token_index] != response_value[token_index][1]:
                    print("token not in response_value", response_str)
                    bool_check = True
                    raise Exception
            if not bool_check:
                return response_value
        except Exception as e:
            print(e, response_str)


def collect_index_and_save(df_text_unit, para_id, response_value, df_target_token, token_type, index_str):
    index_list = []
    index_credibility_list = []
    text_unit_list = []
    para_id_list = []
    df_text_unit_of_para_id = df_text_unit[df_text_unit["para_id"] == para_id]
    for token_index in range(len(response_value)):
        text_unit_component = df_target_token.iloc[token_index]["text_unit_component"][0]
        row_index = df_target_token.iloc[token_index]["row"][0]
        relevance = response_value[token_index][2]
        relevance_credibility = response_value[token_index][3]
        for col_index in text_unit_component:
            text_unit = df_text_unit_of_para_id[df_text_unit_of_para_id["row"] == row_index][df_text_unit_of_para_id["col"] == col_index]["word"].iloc[0]
            text_unit_list.append(text_unit)
            index_list.append(relevance)
            index_credibility_list.append(relevance_credibility)
            para_id_list.append(para_id)

    new_df = pd.DataFrame({
        'text_unit': text_unit_list,
        f'weight': index_list,
        f'weight_credibility': index_credibility_list,
        "para_id": para_id_list
    })
    # save_path = f"data/text/{configs.round}/weight/temp/8_21_{token_type}_relevance_from_gpt_{target_para_index[target_index]}.csv"
    save_path = f"data/text/{configs.round}/weight/temp/8_21_{token_type}_{index_str}_from_gpt_{para_id}.csv"
    new_df.to_csv(save_path, encoding='utf-8_sig', index=False)


def send_article_analyse_request(full_text):
    start_using_IDE()
    set_openai()

    response = openai.ChatCompletion.create(
        model="gpt-4-0613",
        messages=[
            {"role": "system",
             "content": f"你是一个分析文本阅读行为的专家。你需要找出用户在快速阅读时，最可能关注的那些能够“吸引眼球”的词语。注意，由于用户是在做快速阅读，所以对于首尾内容关注会更多，对于中间的内容则会更有选择性。"},
            {"role": "user", "content": "我会给你一段文本，你需要告诉我这段文本的类型是什么（新闻、推荐、微博、评论还是其他类型），可能是在哪个平台发布的；这篇文章的5个关键词；"
                                        "以及对于这个平台的用户而言，这些类型的文本应该着重关注哪些内容（what_to_focus，用自然语言描述，尽量全面）；基于关键词与着重关注内容，用户在阅读这篇文章中主要关注的文字（分词）（focus_words）；"
                                        "这类文本非关注的内容是哪些（what_to_ignore，用自然语言描述，尽量全面）；基于与关键词差别及着非重关注内容，用户在阅读这篇文章时没有关注的文字（分词）是哪些（ignored_words）。"
                                        "请注意，值得关注的内容与不值得关注的内容间不应该存在冲突（即对于2个相似的概念，不应该一个出现在值得关注，另一个出现在不值得关注）。你可以尽可能多地找出需要关注和不需要关注的内容。\n"
                                        "对于关注和不关注的划分，一个相对统一的标准是文本中“吸引眼球”的、极具话题性的词语一般都会被关注。"
                                        "但此外，不同类型的文本会有不同的标准。"
                                        "比如对于新闻，用户可能更关注新闻的主题、背景、趋势，而不关注具体的历史背景、原因探讨、详细的阐述部分；"
                                        "对于观点阐述类的文本，用户会更关注观点的核心内容、造成该观点的原因、作者对观点的举例、作者观点与主流想法的冲突点，而不关注那些大家都心知肚明的内容；"
                                        "对于评论类文本，用户会关注对被评论对象的细节描写，而不关注那些主观的印象。"
                                        "假设具体文本为："
                                        "“饭圈”产业价值千亿却走向癫狂，谁该负责"
                                        "“饭圈”文化愈演愈烈，粉丝与网络水军混杂，因各种立场、观点、利益冲突，而引发各类网上互撕互黑等风波。6月15日，中央网信办宣布开展为期两个月的“饭圈”乱象整治专项行动，聚焦明星榜单、热门话题、粉丝社群、互动评论等环节。突如其来的强监管，显示近年狂飙突进的“粉丝经济”走到了十字路口。"
                                        "官方文件直接点名“饭圈”，前所未见。“饭圈”是一个近年走红的网络用语，主要指娱乐明星粉丝（Fans）组成的圈子。不同于过去所谓“追星族”，“饭圈”更多是基于社群网络的半职业化组织，一些娱乐明星的粉丝业已形成职业分工运作模式，包括“粉头”“数据女工”等新型角色，深度参与明星日常活动，为明星造热度，维持形象和商业价值。\n"
                                        "你需要用字典返回以下六个内容，如"
                                        "{'type': '这段文本类型是新闻，可能在新闻网站或新闻APP上发布。', "
                                        "'keywords': ['饭圈', '整治', '乱象', '娱乐明星', '产业价值千亿']"
                                        "'what_to_focus': '读者在此类文本中应关注的内容包括文中谈到的主要问题和情况，即“饭圈”的现状以及由这种现状产生的问题。同时，他们还需要关注由官方提出的各种应对策略，以及这些策略预计会对所关注的问题产生的影响。', "
                                        "'what_to_ignore': '在这段文本中，需要关注的分词包括“饭圈”、“癫狂”、“中央网信办”、“整治专项行动”、“粉丝经济”、“十字路口”、“粉丝”、“半职业化组织”、“粉头”、“数据女工”。这些词语各自代表了此文的主题，背景，趋势，而它们之间的关系则构成了文章的主要论述。'"
                                        "'focus_words': '在快速阅读时，较具体的历史背景、原因探讨、详细的阐述部分可以忽略，比如“饭圈”的起源、粉丝分工运作模式等内容。这些内容对于理解文章的主旨和大意没有直接帮助，特别是在时间紧张的情况下。'"
                                        "'ignored_words: '可以忽略的分词包括“追星族”、“网络用语”、“走红”、“商业价值”。这些词对于文章的核心观点（饭圈乱象的现状以及应对策略）的理解并没有决定性的影响。'}\n"                                        
                                        "文本如下：\n"
                                        f"{full_text}"
                                        }])

    response_str = response["choices"][0]["message"]["content"].strip()
    print(response_str)

    try:
        response_value = json.loads(response_str.replace("\'", "\""))
        bool_check = False
        key_list = ['type', 'what_to_focus', 'what_to_ignore', 'focus_words', 'ignored_words']
        for key in key_list:
            if key not in response_value.keys():
                print("token not in response_value", response_str)
                bool_check = True
                raise Exception
        if not bool_check:
            return response_value
    except Exception as e:
        print(e, response_str)


def _save_gpt_prediction(token_type):
    # df_density_for_training, df_density_for_testing, df_training_statistic, df_testing_statistic = prepare_data(token_type)
    # training_str_list = prepare_text_for_gpt(df_density_for_training, bool_training=True)
    # testing_str_list = prepare_text_for_gpt(df_density_for_testing, bool_training=False)

    token_list = read_files.read_tokens()
    target_para_index = [90, 91, 92, 93, 94]
    target_token_list = []
    for para_index in target_para_index:
        target_token_list.append(token_list[para_index])

    raw_full_text_list = read_files.read_sorted_text()
    full_text_list = []
    for para_index in target_para_index:
        full_text_list.append([raw_full_text_list[para_index]["text"]])

    with Pool(8) as p:
        reading_analyse_result_list = p.starmap(send_article_analyse_request, full_text_list)

    text_unit_file_path = f"data/text/{configs.round}/text_sorted_mapping.csv"
    df_text_unit = pd.read_csv(text_unit_file_path, encoding="utf-8_sig")

    for target_index in range(len(target_token_list)):
        df_target_token = target_token_list[target_index]
        full_text = ""

        for token_index in range(df_target_token.shape[0]):
            full_text += df_target_token.iloc[token_index]["tokens"]
            if token_index + 1 < df_target_token.shape[0] - 1 and df_target_token.iloc[token_index]["row"] != df_target_token.iloc[token_index + 1]["row"]:
                cur_row = df_target_token.iloc[token_index]["row"][0]
                next_row = df_target_token.iloc[token_index + 1]["row"][0]
                for row_index in range(next_row - cur_row):
                    full_text += "\n"
                if df_target_token.iloc[token_index]["split"] == 1:
                    next_token = df_target_token.iloc[token_index + 1]["tokens"]
                    # take away the (.*) in next token
                    next_token = re.sub(r"\(.+\)", "", next_token)
                    df_target_token["tokens"][token_index + 1] = next_token

        token_list = df_target_token["tokens"].tolist()

        max_num_once = 20
        # 将token_list按max_num_once为单位分割成多个子列表
        token_list_divided = [token_list[i:i + max_num_once] for i in range(0, len(token_list), max_num_once)]
        args_list = []
        for response_index in range(len(token_list_divided)):
            token_list_divided_str = "[" + ",".join([f"({i + response_index * max_num_once}, '{token_list_divided[response_index][i]}')" for i in range(len(token_list_divided[response_index]))]) + "]"

            # prompt_str = "{" + f"'full_text': '{full_text}', 'token_list': {token_list_divided_str}" + "}"
            prompt_str = "{" + f"'token_list': {token_list_divided_str}" + "}"
            # response = openai.ChatCompletion.create(
            #     model="gpt-4-0613",
            #     messages=[
            #         # {"role": "system", "content": "你是一个分析文本与分词的专家。我会为你提供一段文本与它对应分词，你需要告诉我三个指标，一是每个分词与文章的关联程度，二是每个分词本身的阅读难度，三是这个分词在前后文中的阅读难度。"},
            #         {"role": "system", "content": "你是一个分析文本与分词的专家。我会为你提供一段文本与它对应分词，你需要告诉我一个指标：每个分词与文章的关联程度。"},
            #         {"role": "user", "content": "我会解释一下我将提供的数据及其含义。\n"
            #                                     "我会向你提供一句话的完整文本与其对应的分词及每个分词序号。如{'full_text': '今天天气很好', 'token_list': [(0, '今天'), (1, '天气'), (2, '很好')]}\n"
            #                                     "full_text中的'\n'代表换行。\n"
            #                                     # "你需要给我的返回是一个列表，其中包含8个成分：分词序号，分词，关联程度打分（rele），关联程度打分确定度（rele_cred），分词本身阅读难度打分（diff），分词本身阅读难度打分确定度（diff_cred），分词上下文阅读难度打分（diff_context），分词上下文阅读难度打分确定度（diff_context_cred）。"
            #                                     "你需要给我的返回是一个列表，其中包含4个成分：分词序号，分词，关联程度打分（rele），关联程度打分确定度（rele_cred）。"
            #                                     # "关联程度打分、分词本身阅读难度打分、分词上下文阅读难度打分的分数都是从1到5，分数低代表关联度低或难度低，分数高代表关联度高或难度高。确信度打分为0到1，分数低代表不确定，分数高代表确定。\n"
            #                                     "关联程度打分从1到5，分数低代表关联度低，分数高代表关联度高。确信度打分为0到1，分数低代表不确定，分数高代表确定。\n"
            #                                     # "关联程度代表一个分词与这篇文章想要传达的核心意思之间的关联有多大；分词本身难度，代表仅看这个分词，它有多难理解，如一些专业名词的难度就很高（5分），而一些助词、标点，难度就很低（1分）；分词上下文难度，指放在上下文中，一个分词是否容易理解，如叠词单看时容易理解，但放在上下文中则会引发与上下文内容相关的额外的思考。"
            #                                     "关联程度代表一个分词与这篇文章想要传达的核心意思之间的关联有多大。"
            #                                     "3个关于打分的确定度，可以被理解为不同人是否都会认可你给出的打分，当确定度高时，代表大部分人都会认可你的打分；当确定度低时，代表只有少部分人会认可你的打分。\n"
            #                                     "注意，打分过程中，不可增加、删除或修改token。你需要返回所有token的打分结果，打分时间可以较长，但绝对不可以只返回部分结果。"},
            #         {"role": "user", "content": "接下来我们模拟一下输入和返回的结果。本次模拟中打分的结果不具有参考性。\n"},
            #         {"role": "user", "content": "。{'full_text': '今天天气很好', 'token_list': [[0, '今天'], [1, '天气'], [2, '很好']]}"},
            #         # {"role": "assistant", "content": "{'return_list': [{'index': 0, 'token': '今天', 'rele': 4, 'rele_cred': 0.8, 'diff': 2, 'diff_cred': 0.9, 'diff_context': '3', 'diff_context_cred': 0.7}], "
            #                                          # "[{'index': 1, 'token': '天气', 'rele': 5, 'rele_cred': 0.8, 'diff': 2, 'diff_cred': 0.9, 'diff_context': '3', 'diff_context_cred': 0.7}], "
            #                                          # "[{'index': 2, 'token': '很好', 'rele': 4, 'rele_cred': 0.8, 'diff': 1, 'diff_cred': 0.9, 'diff_context': '2', 'diff_context_cred': 0.7}]}"},
            #         {"role": "assistant", "content": "{'return_list': [[0, '今天', 4, 0.8], [1, '天气', 5, 0.8], [2, '很好', 5, 0.8]]}"},
            #         {"role": "user", "content": "你做得很好，接下来请按相同的格式给我返回，但之后的打分需要是正确的打分。\n"},
            #         {"role": "assistant", "content": "好的，我已经理解你的任务了。你可以给我你需要打分的文本了。\n"},
            #         {"role": "user", "content": f"{prompt_str}"}],
            # )
            #
            # response_str = response["choices"][0]["message"]["content"].strip()
            # print(response_str)


            # relevance_response_value = send_single_request(token_list_divided[response_index], prompt_str,
            #                                                "分词与文本的相关性", "对相关性评分的可信度",
            #                                                "分词与文本的相关性指的是该分词是否与全文的主要含义有明显、强烈的关联。有些分词（如标点、一些助词）并不影响文本的主要含义，完全可以删去，这些分词的相关性就很弱；有些分词则与文本的主要含义密切相关。",
            #                                                "分词与文本的相关性打分从1分到5分，1分代表该分词与文本的主要内容没有关联，5分代表该分词与文本主要内容关联十分明显。")
            # response_value = send_single_request(token_list_divided[response_index], prompt_str,
            #                                                "分词信息指数", "对信息指数的可信度",
            #                                                "分词指数指的是在完成前文的阅读后，该分词是否带来了新的、不同的信息。有些分词（如标点、助词等）并没有带来新的信息，有些分词（如时间、地点等）带来了一些补充信息，有些分词（如与文章主要含义相关性极大的分词）则带来了较多的信息。",
            #                                                "分词与文本的相关性打分从1分到5分，1分代表该分词没有带来额外的信息，5分代表该分词带来了明显不同于之前的信息。")

            # args_list.append([token_list_divided[response_index], prompt_str,
            #                               "分词信息指数", "对分词信息指数的可信度",
            #                               "分词指数指的是在完成前文的阅读后，该分词是否带来了新的、不同的信息。有些分词（如标点、助词等）并没有带来新的信息，有些分词（如时间、地点等）带来了一些补充信息，有些分词（如与文章主要含义相关性极大的分词）则带来了较多的信息。",
            #                               "分词与文本的相关性打分从1分到5分，1分代表该分词没有带来额外的信息，5分代表该分词带来了明显不同于之前的信息。"])

            # args_list.append([token_list_divided[response_index], prompt_str,
            #                               "阅读停留时间", "对阅读停留时间的可信度",
            #                               "分词指数指的是你认为用户可能会在词汇上停留的时间，具体地说，它包括了词汇的含义，词汇的熟悉/陌生程度，词汇在文章中起到的作用，词汇是否容易预测等。注意，这段文本阅读时用户可能并没有非常仔细地逐字阅读，只是简单的扫过了关键信息。"
            #                               "有些分词（如标点、助词等）并没有什么有用的信息，则停留时间会很短；有些分词（如专业词汇等）一般不常见，则会有略长的停留时间；有些分词与文章主要含义相关性极大，则会停留较久。",
            #                               "阅读停留时间打分从1分到5分，1分代表在该分词上的停留时间很短，5分代表在该分词上的停留时间很长。"])

            args_list.append([token_list_divided[response_index], reading_analyse_result_list[target_index], full_text, prompt_str,
                                          "阅读关注程度", "对阅读关注程度的可信度",
                                          "阅读代表用户在阅读时会多关注该分词，用户越关注一个分词，阅读时在这个分词上的停留时间越短。简单地说，它包括了词汇的含义，词汇的熟悉/陌生程度，词汇在文章中起到的作用，词汇是否容易预测等。"
                                          # "同时，不同类别的文本，用户阅读的停留倾向也是不同的，如对于商品介绍类的文本，用户会更加关注参数、功能、价格等；对于观点输出类的文本，用户会更关注观点内容、论据等；对于推荐类的文本，用户会更关注推荐对象、理由等。"
                                          # "注意，这段文本阅读时用户可能并没有非常仔细地逐字阅读，只是简单的扫过了关键信息。"
                                          # "有些分词（如标点、助词等）并没有什么有用的信息，则停留时间会很短；有些分词（如专业词汇等）一般不常见，则会有略长的停留时间；有些分词与文章主要含义相关性极大，则会停留较久。",
                                          "此外，之前提供的对文本的总结与分析同样重要，你需要结合之前分析给出的文本类型、关键词、哪些内容应该关注、哪些内容不应该关注等信息，来判断用户在阅读时会关注哪些内容。",
                                          "阅读关注程度打分从1分到5分，"
                                          "1分代表完全不关注这个分词，分析中提到不该关注的信息及标点等都属于这一类；"
                                          "2分代表该分词不会被特意关注，它几乎没有传递有用的信息，一些连接词、助词或易从上文推测的词语可能会属于这一类；"
                                          "3分代表该分词会被简单关注，它传递了一些有用的信息，但与文章核心内容的联系不那么紧密；"
                                          "4分代表该分词会被关注，它与分词中提到的应该关注的信息存在较强的相关性；"
                                          "5分代表极为注意该分词，分析中提到的应该关注的信息属于这一类。"])

        with Pool(6) as p:
            result_list = p.starmap(send_single_rate_request, args_list)

        response_value = []
        for result in result_list:
            response_value.extend(result)

        collect_index_and_save(df_text_unit, target_para_index[target_index], response_value, df_target_token, token_type, "attention_first")


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

