# 实验参数。
round = "round_1"
device = "tobii"

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

# 多进程相关参数。
num_of_processes = 16 # FIXME 选择需要的进程数。

# 空间对齐相关。
training_number_list = [i for i in range(0, 10)] # FIXME 选择需要的训练数据。
bool_log = False # FIXME 记录log
bool_save_pic = False # FIXME 记录图片
test_str = "test_024" # FIXME 文件名

target_quality_list = ["good", "poor", "invalid"]
num_of_sample_of_quality = [5, 0, 0] # FIXME 设置每个quality需要的样本数。

bool_pick_by_quality = False # FIXME 选择使用quality还是指定训练数据。True则为使用quality，False则为使用指定训练数据。

empty_text_unit_penalty = -40
punctuation_text_unit_penalty = -20
text_unit_density_threshold_for_empty = 5 # 空字符不惩罚的threshold。
continuous_empty_text_unit_threshold = int(col_num * 1 / 3) # 连续空字符不惩罚的threshold。
continuous_empty_text_unit_penalty = 2 # 该值的目的是避免大量text_unit没有被看到，但应该避免对语义距离产生过大影响。对于所有的std point，如果在所有的para下都不为空，且没有任何gaze point对应，则每个这样的text_unit施加一个penalty。

far_from_text_unit_penalty = 150 # 距离过远的惩罚
dist_threshold_from_std = 32 # 距离过远的threshold

H_rotation_penalty = 100
H_scale_penalty = 500
H_shear_penalty = 500
H_projection_penalty = 500
H_space_ratio_penalty = 1000

first_row_text_penalty = 5 # 首行文字未读的惩罚。

coeff_semantic_before_output = 0.5 # 在输出structural_distance_matrix和gpt_distance_matrix之前添加一个系数，使其与absolute distance能够更好地“协作”。
coeff_gpt_for_non_structural = 0
coeff_gpt = 0.9999
coeff_structural = 0.0001

generic_population_size = 500
generic_population_generation = 3
em_iteration = 15

bool_blur_weight = False
# gaussian_filter = [[0.02919022, 0.04812654, 0.02919022],
#                    [0.21568818, 0.35560969, 0.21568818],
#                    [0.02919022, 0.04812654, 0.02919022]]
gaussian_filter = [[0.01772047, 0.07106603, 0.01772047],
                   [0.13093757, 0.52511091, 0.13093757],
                   [0.01772047, 0.07106603, 0.01772047]]
