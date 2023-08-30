# 实验参数。
round = "round_1"
device = "tobii"

# 多进程相关参数。
num_of_processes = 14

# text density计算相关参数。
text_unit_density_distance_threshold = 128

# reading分析相关。
reading_density_distance_threshold = 50

# 实验GUI相关参数。
top = 120
left = 360
text_width = 40
text_height = 64
row_num = 12
col_num = 30

# fine tune训练相关参数。
fine_tune_training_para_num = 30
fine_tune_training_file_num = 8
# fine_tune_model_name = "curie:ft-pcg-2023-08-16-16-36-16"
# fine_tune_model_name = "curie:ft-pcg-2023-08-17-05-02-34"
fine_tune_model_name = "curie:ft-pcg-2023-08-18-19-07-57"
fine_tune_ver = 0

# 文本分词相关参数。
temp_token_debug_num = 23
punctuation_list = {'\'', '\"', '!', '?', '.', '/', '\\', '-', '，', ':', '：', '。', '……', '！', '？', '——', '（', '）', '【', '】', '“', '”', '’', '‘', '：', '；', '《', '》', '、', '—', '～', '·', '「', '」', '『', '』'}

# 空间对齐相关。
training_number_list = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9] # FIXME 选择需要的训练数据。
bool_log = False
bool_save_pic = False
test_str = "test_014"
empty_text_unit_penalty = -40
punctuation_text_unit_penalty = -20
far_from_text_unit_penalty = 180
dist_threshold_from_std = 32
H_rotation_penalty = 100
H_scale_penalty = 500
H_shear_penalty = 500
H_projection_penalty = 500
H_space_ratio_penalty = 1000
text_unit_density_threshold_for_empty = 6
first_row_text_penalty = 5
coeff_gpt_for_non_structural = 0
coeff_gpt = 0.0001
coeff_structural = 0.9999
generic_population_size = 500
generic_population_generation = 3
em_iteration = 15

