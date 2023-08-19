import json
import os
import re
import time

import pandas as pd
from hanlp_restful import HanLPClient

import configs
import read_files


def tokenize_with_hanlp(HanLP, text):
    def replace_backslash_with_space(tokens):
        for sentence_index in range(len(tokens)):
            for token_index in range(len(tokens[sentence_index])):
                token = tokens[sentence_index][token_index]
                token = token.replace("\\", " ")
                tokens[sentence_index][token_index] = token

    hanlp_result = HanLP(text)
    fine_tokens = hanlp_result["tok/fine"]
    coarse_tokens = hanlp_result["tok/coarse"]

    replace_backslash_with_space(fine_tokens)
    replace_backslash_with_space(coarse_tokens)

    return fine_tokens, coarse_tokens


def tokenize_exp_text_sorted():
    def combine_tokens_in_sentence(tokens):
        combined_tokens = []
        for token in tokens:
            combined_tokens += token
        return combined_tokens

    def concatenate_space(tokens):
        start_pattern = r"^[a-zA-Z0-9()\[\]/%.\-=]+"
        end_pattern = r".*[a-zA-Z0-9()\[\]/%.\-=]+$"
        for row_index in range(len(tokens)):
            token_index = 0
            while token_index < len(tokens[row_index]):
                token = tokens[row_index][token_index]
                # 以下是round_4时就有的规则。
                if token.strip() == "":
                    while token_index + 1 < len(tokens[row_index]) and tokens[row_index][token_index + 1].strip() == "":
                        tokens[row_index][token_index] += tokens[row_index][token_index + 1]
                        del tokens[row_index][token_index + 1]
                    if tokens[row_index][token_index] == "    ":
                        # 这里的空格数目是4，说明当前是行首，可以单独作为一个token。
                        token_index += 1
                        continue
                    if token_index - 1 >= 0 and bool(re.match('^[a-zA-Z0-9 .亿万元]*$', tokens[row_index][token_index - 1])):
                        token_minus_1 = tokens[row_index][token_index - 1]
                        tokens[row_index][token_index - 1] += token
                        del tokens[row_index][token_index]
                    elif token_index + 1 < len(tokens[row_index]) and bool(re.match('^[a-zA-Z0-9 .亿万元]*$', tokens[row_index][token_index + 1])):
                        token_plus_1 = tokens[row_index][token_index + 1]
                        tokens[row_index][token_index + 1] = token + tokens[row_index][token_index + 1]
                        del tokens[row_index][token_index]
                    else:
                        token_index += 1
                elif token.startswith("    ") and len(token.strip()) > 0:
                    # 这种情况也极为特殊，出现在round_1。我把它提出来了。是hanlp的错误分词导致的，它将\\和3分到了一个分词中。
                    tokens[row_index].insert(token_index + 1, tokens[row_index][token_index][4:])
                    tokens[row_index][token_index] = tokens[row_index][token_index][:4]
                elif bool(re.match(start_pattern, token)) and token_index - 1 >= 0 and bool(re.match(end_pattern, tokens[row_index][token_index - 1])):
                    this_match = re.search(start_pattern, token)
                    this_str = token[this_match.start():this_match.end()]
                    last_match = re.search(end_pattern, tokens[row_index][token_index - 1])
                    last_str = tokens[row_index][token_index - 1][last_match.start():last_match.end()]
                    if len(this_str) % 2 == 1 and len(last_str) % 2 == 1:
                        tokens[row_index][token_index - 1] += token
                        del tokens[row_index][token_index]
                    elif abs(len(this_str) % 2 - len(last_str) % 2) == 1:
                        tokens[row_index][token_index - 1] += token
                        del tokens[row_index][token_index]
                    else:
                        token_index += 1

                # elif token == ")" and token_index + 1 < len(tokens[row_index]) and tokens[row_index][token_index + 1] == "(":
                #     # 排除一些特殊的情况，如将)(当成一个token。 # TODO 这种情况极为特殊，以后绝对不要再出现这样的把前后括号合并到一个text_unit中的情况！！
                #     token_1 = ")"
                #     token_2 = "("
                #     tokens[row_index][token_index] = token_1 + token_2
                #     del tokens[row_index][token_index + 1]
                #     token_index += 1
                # elif token == "=":
                #     # TODO 这种情况同样极为特殊，下次在遇到相关等号的词时需要谨慎处理。
                #     while token_index + 1 < len(tokens[row_index]) and bool(re.match('^[0-9 .亿万元]*$', tokens[row_index][token_index + 1])):
                #         tokens[row_index][token_index] += tokens[row_index][token_index + 1]
                #         del tokens[row_index][token_index + 1]
                # 以下应该是为了处理round_1新加的规则。
                # elif token.isdigit() and token_index + 1 < len(tokens[row_index]) and tokens[row_index][token_index + 1] == ".":
                #     # 这种情况相对比较常见，即数字后面出现.的情况。需要将两者合并成一个token。
                #     tokens[row_index][token_index] += tokens[row_index][token_index + 1]
                #     del tokens[row_index][token_index + 1]
                # elif bool(re.match('    [0-9]*$', token)) and token_index + 1 < len(tokens[row_index]) and tokens[row_index][token_index + 1] == ".":
                #     # 这种情况也极为特殊，但是是hanlp的错误分词导致的，它将\\和3分到了一个分词中。
                #     tokens[row_index][token_index + 1] = f"{tokens[row_index][token_index][-1]}."
                #     tokens[row_index][token_index] = tokens[row_index][token_index][:-1]
                # elif token[-1] == ")" and token_index + 1 < len(tokens[row_index]) and tokens[row_index][token_index + 1] == "-":
                #     # 这种情况是半角括号与-相邻，为了保证text_unit好处理，将两者合并。
                #     tokens[row_index][token_index] += tokens[row_index][token_index + 1]
                #     del tokens[row_index][token_index + 1]
                # elif token == "-" and token_index + 1 < len(tokens[row_index]) and bool(re.match('^[a-zA-Z0-9]*$', tokens[row_index][token_index + 1][0])):
                #     # 这种情况较为常见，是-和英文或数字接在了一起。需要将两者合并成一个token。
                #     tokens[row_index][token_index] += tokens[row_index][token_index + 1]
                #     del tokens[row_index][token_index + 1]
                # elif bool(re.match('\([a-zA-Z]*', token)) and token_index + 1 < len(tokens[row_index]) and bool(re.match('^[a-zA-Z0-9)]*$', tokens[row_index][token_index + 1][0])):
                #     # 这种情况较为常见，是半角括号(和数字接在了一起。需要将两者合并成一个token。
                #     tokens[row_index][token_index] += tokens[row_index][token_index + 1]
                #     del tokens[row_index][token_index + 1]
                # elif token == "(" and token_index + 1 < len(tokens[row_index]) and bool(re.match('^[a-zA-Z0-9]*$', tokens[row_index][token_index + 1])):
                #     token[row_index][token_index] += tokens[row_index][token_index + 1]
                #     del tokens[row_index][token_index + 1]
                # elif token == "/" and token_index + 1 < len(tokens[row_index]) and tokens[row_index][token_index + 1] == "Moss":
                #     # 罕见的情况，分词中出现斜杠，550w/Moss。
                #     tokens[row_index][token_index] += tokens[row_index][token_index + 1]
                #     del tokens[row_index][token_index + 1]
                # elif token == "a" and tokens[row_index][token_index + 1] == "==" and tokens[row_index][token_index + 2] == "b":
                #     tokens[row_index][token_index] = "a==b"
                #     del tokens[row_index][token_index + 1]
                #     del tokens[row_index][token_index + 1]
                # elif token == "23.6" and tokens[row_index][token_index + 1] == "%" and tokens[row_index][token_index + 2] == "(" and tokens[row_index][token_index + 3] == "1231":
                #     tokens[row_index][token_index] += tokens[row_index][token_index + 1] + tokens[row_index][token_index + 2] + tokens[row_index][token_index + 3]
                #     del tokens[row_index][token_index + 1]
                #     del tokens[row_index][token_index + 1]
                #     del tokens[row_index][token_index + 1]
                # elif token == "23.6%" and tokens[row_index][token_index + 1] == "(" and tokens[row_index][token_index + 2] == "1231年":
                #     tokens[row_index][token_index] += tokens[row_index][token_index + 1] + "1231"
                #     del tokens[row_index][token_index + 1]
                #     tokens[row_index][token_index + 1] = "年"
                # elif token == "[" and tokens[row_index][token_index + 1].isdigit() and tokens[row_index][token_index + 2] == "]":
                #     tokens[row_index][token_index] += tokens[row_index][token_index + 1] + tokens[row_index][token_index + 2]
                #     del tokens[row_index][token_index + 1]
                #     del tokens[row_index][token_index + 1]
                # elif token == "维生素C" and tokens[row_index][token_index + 1] == "-" and tokens[row_index][token_index + 2] == "WIKI":
                #     tokens[row_index][token_index] += tokens[row_index][token_index + 1] + tokens[row_index][token_index + 2]
                #     del tokens[row_index][token_index + 1]
                #     del tokens[row_index][token_index + 1]
                else:
                    token_index += 1

    sorted_text_list = read_files.read_sorted_text()

    file_key_hanlp = "data/key/HanLP.txt"
    f = open(file_key_hanlp, "r", encoding="utf-8")
    key = f.read().split("\n")[0]
    HanLP = HanLPClient('https://hanlp.hankcs.com/api', auth=key, language='zh')

    time_list = []

    for text_dict in sorted_text_list:
        # 用于控制每分钟最多调用HanLP 50次。
        current_time = time.time()
        time_list.append(current_time)
        if len(time_list) > 48:
            if current_time - time_list[0] < 65:
                print(f"sleep for {65 - (current_time - time_list[0])} seconds")
                time.sleep(65 - (current_time - time_list[0]))
                print("awake")
                time_list = []

        text_index = text_dict["text_index"]
        print(text_index)
        # if text_index != f'{configs.temp_token_debug_num}':
        #     continue
        text = text_dict["text"].replace(" ", "\\")

        fine_tokens, coarse_tokens = tokenize_with_hanlp(HanLP, text)
        concatenate_space(fine_tokens)
        concatenate_space(coarse_tokens)
        combined_fine_tokens = combine_tokens_in_sentence(fine_tokens)
        combined_coarse_tokens = combine_tokens_in_sentence(coarse_tokens)

        fine_df = pd.DataFrame({"tokens": combined_fine_tokens})
        coarse_df = pd.DataFrame({"tokens": combined_coarse_tokens})

        # 保存分词结果
        file_prefix = f"data/text/{configs.round}/tokens/"
        fine_file_path_prefix = file_prefix + "fine_tokens/"
        coarse_file_path_prefix = file_prefix + "coarse_tokens/"

        if not os.path.exists(os.path.dirname(fine_file_path_prefix)):
            os.makedirs(os.path.dirname(fine_file_path_prefix))
        if not os.path.exists(os.path.dirname(coarse_file_path_prefix)):
            os.makedirs(os.path.dirname(coarse_file_path_prefix))

        fine_df.to_csv(f"{fine_file_path_prefix}{text_index}.csv", encoding="utf-8_sig", index=False)
        coarse_df.to_csv(f"{coarse_file_path_prefix}{text_index}.csv", encoding="utf-8_sig", index=False)


def add_text_unit_index_to_tokens():
    def find_text_unit_of_tokens(text_unit_list, token_df):
        # 目前不同的token在token_df中，组成token的text_unit在text_unit_list中。text_unit_list是个二维数组，第一维代表行数，第二维代表每行的text_unit。
        # 这些token会依序被排在行中，
        # 需要将token_df中的token与text_unit_list中的text_unit进行匹配。
        bool_split = False
        target_token_index = 0
        split_probe_index = 0
        text_unit_component_list = [[] for _ in range(token_df.shape[0])]
        row_list = [[] for _ in range(token_df.shape[0])]

        for row_index in range(len(text_unit_list)):
            if not bool_split:
                text_unit_start_index = 0
                text_unit_end_index = 0
            else:
                text_unit_start_index = split_probe_index
                text_unit_end_index = split_probe_index
                bool_split = False

            matching_text = ""

            while text_unit_end_index < len(text_unit_list[row_index]):
                current_text_unit = text_unit_list[row_index][text_unit_end_index]
                matching_text += current_text_unit
                target_token = token_df["tokens"][target_token_index]
                if len(matching_text.strip()) == 0 and len(target_token.strip()) == 0 and matching_text != target_token:
                    # 都是只有空格，但是长度不一致。
                    text_unit_end_index += 1
                    continue
                elif len(matching_text.strip()) == 0 and len(target_token.strip()) == 0 and matching_text == target_token:
                    # 都是只有空格，但是长度一致了。
                    text_unit_component_list[target_token_index].append([i for i in range(text_unit_start_index, text_unit_end_index + 1)])
                    row_list[target_token_index].append(row_index)
                    target_token_index += 1
                    text_unit_start_index = text_unit_end_index + 1
                    matching_text = ""
                elif len(matching_text.strip()) != 0 and len(target_token.strip()) != 0 and matching_text.replace(" ", "") == target_token.replace(" ", ""):
                    # 其他含有空格的情况。
                    text_unit_component_list[target_token_index].append([i for i in range(text_unit_start_index, text_unit_end_index + 1)])
                    row_list[target_token_index].append(row_index)
                    target_token_index += 1
                    text_unit_start_index = text_unit_end_index + 1
                    matching_text = ""
                elif text_unit_end_index == len(text_unit_list[row_index]) - 1 and matching_text != target_token:
                    probe_text = matching_text
                    split_probe_index = 0
                    while row_index < len(text_unit_list) - 1 and split_probe_index < len(text_unit_list[row_index + 1]) and probe_text != target_token:
                        probe_text += text_unit_list[row_index + 1][split_probe_index]
                        split_probe_index += 1
                    if probe_text != target_token:
                        print(probe_text, target_token)
                        raise "matching error"

                    text_unit_component_list[target_token_index].append([i for i in range(text_unit_start_index, text_unit_end_index + 1)])
                    text_unit_component_list[target_token_index].append([i for i in range(0, split_probe_index)])
                    row_list[target_token_index].append(row_index)
                    row_list[target_token_index].append(row_index + 1)
                    target_token_index += 1
                    bool_split = True

                text_unit_end_index += 1
        return text_unit_component_list, row_list

    def get_row_position_and_split(df):
        split_list = [0 for _ in range(df.shape[0])]
        row_position_list = ["middle" for _ in range(df.shape[0])]
        row_position_list[0] = "passage_start"
        row_position_list[-1] = "passage_end"

        for text_unit_index in range(1, df.shape[0]):
            if len(df["row"].iloc[text_unit_index]) > 3:
                split_list[text_unit_index] = 1
                row_position_list[text_unit_index] = "row_split"
            if df["row"].iloc[text_unit_index] != df["row"].iloc[text_unit_index - 1] and len(df["row"].iloc[text_unit_index]) == 3 and len(df["row"].iloc[text_unit_index - 1]) == 3:
                row_position_list[text_unit_index] = "row_start"
                row_position_list[text_unit_index - 1] = "row_end"
            # 无split的对象的row为“[n]”，长度为3；split对象的row为“[n, n+1]”长度为5。

        return split_list, row_position_list

    sorted_text_mapping = read_files.read_text_mapping_of_sorted_data()

    fine_tokens_path_prefix = f"data/text/{configs.round}/tokens/fine_tokens/"
    coarse_tokens_path_prefix = f"data/text/{configs.round}/tokens/coarse_tokens/"
    fine_token_file_index_list, coarse_token_file_index_list, tokens_num = read_files.read_token_file_names()
    for file_index in range(tokens_num):
        fine_tokens_file_name = f"{fine_token_file_index_list[file_index]}.csv"
        coarse_tokens_file_name = f"{coarse_token_file_index_list[file_index]}.csv"
        fine_tokens_df = pd.read_csv(f"{fine_tokens_path_prefix}{fine_tokens_file_name}", encoding="utf-8_sig", skip_blank_lines=False)
        coarse_tokens_df = pd.read_csv(f"{coarse_tokens_path_prefix}{coarse_tokens_file_name}", encoding="utf-8_sig", skip_blank_lines=False)
        text_unit_list = sorted_text_mapping[file_index]
        # if file_index != configs.temp_token_debug_num:
        #     continue
        print(file_index)
        fine_text_unit_component_list, fine_row_list = find_text_unit_of_tokens(text_unit_list, fine_tokens_df)
        coarse_text_unit_component_list, coarse_row_list = find_text_unit_of_tokens(text_unit_list, coarse_tokens_df)
        fine_tokens_df["text_unit_component"] = fine_text_unit_component_list
        fine_tokens_df["row"] = fine_row_list
        coarse_tokens_df["text_unit_component"] = coarse_text_unit_component_list
        coarse_tokens_df["row"] = coarse_row_list
        fine_tokens_df["text_unit_component"] = fine_tokens_df["text_unit_component"].apply(json.dumps)
        fine_tokens_df["row"] = fine_tokens_df["row"].apply(json.dumps)
        coarse_tokens_df["text_unit_component"] = coarse_tokens_df["text_unit_component"].apply(json.dumps)
        coarse_tokens_df["row"] = coarse_tokens_df["row"].apply(json.dumps)

        fine_split_list, fine_row_change_list = get_row_position_and_split(fine_tokens_df)
        coarse_split_list, coarse_row_change_list = get_row_position_and_split(coarse_tokens_df)
        fine_tokens_df["split"] = fine_split_list
        fine_tokens_df["row_position"] = fine_row_change_list
        coarse_tokens_df["split"] = coarse_split_list
        coarse_tokens_df["row_position"] = coarse_row_change_list

        fine_tokens_df.to_csv(f"{fine_tokens_path_prefix}{fine_tokens_file_name}", encoding="utf-8_sig", index=False)
        coarse_tokens_df.to_csv(f"{coarse_tokens_path_prefix}{coarse_tokens_file_name}", encoding="utf-8_sig", index=False)


def add_text_context_to_tokens():
    def get_forward_and_backward(df):
        forward_1 = []
        backward_1 = []
        for token_index in range(df.shape[0]):
            forward_2 = []
            backward_2 = []

            probe_index = 0
            while token_index + probe_index >= 0 and abs(probe_index) < 4:
                # 一行开头添加<SOR>。
                if df["row"].iloc[token_index + probe_index] != df["row"].iloc[token_index]:
                    backward_2.insert(0, "<SOR>")
                    break
                if probe_index != 0:
                    backward_2.insert(0, df["tokens"].iloc[token_index + probe_index])
                # 文本开头处添加<SOR>。
                if abs(probe_index) < 3 and token_index + probe_index == 0:
                    backward_2.insert(0, "<SOR>")
                    break
                probe_index -= 1

            probe_index = 0
            while token_index + probe_index < df.shape[0] and probe_index < 4:
                # 换行处添加<EOR>。
                if df["row"].iloc[token_index + probe_index] != df["row"].iloc[token_index]:
                    forward_2.append("<EOR>")
                    break
                if abs(probe_index) < 3 and token_index + probe_index == df.shape[0] - 1:
                    forward_2.append("<EOR>")
                    break
                if probe_index != 0:
                    forward_2.append(df["tokens"].iloc[token_index + probe_index])
                probe_index += 1

            backward_1.append(backward_2)
            forward_1.append(forward_2)
        df["backward"] = backward_1
        df["forward"] = forward_1
        df["backward"] = df["backward"].apply(json.dumps)
        df["forward"] = df["forward"].apply(json.dumps)

    def split_cross_line_token(df):
        '''
        将那些跨行的token拆分成多个token。
        :param df:
        :return:
        '''
        token_index = 0
        while token_index < df.shape[0]:
            if len(df["row"].iloc[token_index]) > 1:
                last_row_token_length = len(df["text_unit_component"].iloc[token_index][0])
                last_row_token = df["tokens"].iloc[token_index][:last_row_token_length]
                next_row_token = df["tokens"].iloc[token_index][last_row_token_length:]
                last_row = df["row"].iloc[token_index][0]
                next_row = df["row"].iloc[token_index][1]
                last_text_unit_component = df["text_unit_component"].iloc[token_index][0]
                next_text_unit_component = df["text_unit_component"].iloc[token_index][1]
                df["tokens"].iloc[token_index] = last_row_token
                df["row"].iloc[token_index] = [last_row]
                df["text_unit_component"].iloc[token_index] = [last_text_unit_component]
                df["row_position"].iloc[token_index] = "row_end"

                new_df = pd.DataFrame(
                    {"tokens": f"({last_row_token}){next_row_token}", "row": [[next_row]], "text_unit_component": [[next_text_unit_component]], "row_position": "row_start", "split": 1},
                    index=[token_index + 0.5])
                df_1 = df.loc[:token_index]
                df_2 = df.loc[token_index + 1:]
                df = pd.concat([df_1, new_df, df_2]).reset_index(drop=True)

            token_index += 1

        return df

    def add_distance_to_row_start_and_end(df):
        row_list = df["row"].tolist()
        row_index_list = [item[0] for item in row_list]
        df["row"] = row_index_list
        start_dist_list = []
        end_dist_list = []
        df_grouped_by_row = df.groupby("row")
        for row, df_row in df_grouped_by_row:
            for token_index in range(df_row.shape[0]):
                start_dist_list.append(token_index)
                end_dist_list.append(df_row.shape[0] - token_index - 1)

        df["start_dist"] = start_dist_list
        df["end_dist"] = end_dist_list
        df["row"] = row_list
        print()

    def get_anterior_passage(df):
        punctuation_list = ["，", "。", "！", "？", "……", "    ", "；"]
        anterior_passage_1 = []
        for token_index in range(df.shape[0]):
            if token_index == 0:
                anterior_passage_1.append("<SOP>")
                continue

            anterior_passage_2 = []
            probe_index = -1
            bool_punctuation_one_time = False
            while token_index + probe_index >= 0:
                probe_token = df["tokens"].iloc[token_index + probe_index]
                if probe_token in punctuation_list:
                    if bool_punctuation_one_time:
                        break
                    else:
                        bool_punctuation_one_time = True
                anterior_passage_2.insert(0, probe_token)
                if token_index + probe_index == 0:
                    anterior_passage_2.insert(0, "<SOP>")
                    break
                probe_index -= 1
            anterior_passage_1.append("".join(anterior_passage_2))
        df["anterior_passage"] = anterior_passage_1
        df["anterior_passage"] = df["anterior_passage"].apply(json.dumps)

    fine_tokens_path_prefix = f"data/text/{configs.round}/tokens/fine_tokens/"
    coarse_tokens_path_prefix = f"data/text/{configs.round}/tokens/coarse_tokens/"
    fine_token_file_index_list, coarse_token_file_index_list, tokens_num = read_files.read_token_file_names()
    for file_index in range(tokens_num):
        fine_tokens_file_name = f"{fine_token_file_index_list[file_index]}.csv"
        coarse_tokens_file_name = f"{coarse_token_file_index_list[file_index]}.csv"
        fine_tokens_df = pd.read_csv(f"{fine_tokens_path_prefix}{fine_tokens_file_name}", encoding="utf-8_sig")
        coarse_tokens_df = pd.read_csv(f"{coarse_tokens_path_prefix}{coarse_tokens_file_name}", encoding="utf-8_sig")
        # fine_tokens_df.drop("Unnamed: 0", axis=1, inplace=True)
        # coarse_tokens_df.drop("Unnamed: 0", axis=1, inplace=True)

        read_files.json_load_for_df_columns(fine_tokens_df, ["text_unit_component", "row"])
        read_files.json_load_for_df_columns(coarse_tokens_df, ["text_unit_component", "row"])

        fine_tokens_df = split_cross_line_token(fine_tokens_df)
        coarse_tokens_df = split_cross_line_token(coarse_tokens_df)

        add_distance_to_row_start_and_end(fine_tokens_df)
        add_distance_to_row_start_and_end(coarse_tokens_df)

        get_forward_and_backward(fine_tokens_df)
        get_forward_and_backward(coarse_tokens_df)

        get_anterior_passage(fine_tokens_df)
        get_anterior_passage(coarse_tokens_df)

        save_path_prefix = f"data/text/{configs.round}/tokens/"
        fine_tokens_df.to_csv(f"{save_path_prefix}fine_tokens/{fine_tokens_file_name}", encoding="utf-8_sig", index=False)
        coarse_tokens_df.to_csv(f"{save_path_prefix}coarse_tokens/{coarse_tokens_file_name}", encoding="utf-8_sig", index=False)
