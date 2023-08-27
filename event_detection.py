from multiprocessing import Pool

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter, find_peaks

import read_files

FIRST_CHAR_X = 380
FIRST_CHAR_Y = 154
LAST_CHAR_X = 980
LAST_CHAR_Y = 922


'''
labels:
-2 - Unknown, 
-1 - Noise, 
0 - Fixation, 
1 - Saccade, 
2 - Return Sweep, 
3 - Blink
'''


def moving_average(x, w):
    return np.convolve(x, np.ones(w), 'same') / w


def event_detection(data, file_index, para_index):
    print(file_index, para_index)
    slice_tolerance = 2
    fixation_distance_threshold_max = 25
    fixation_distance_threshold_min = 5
    fixation_minimum_points = 4
    fixation_seek_points = 20
    return_sweep_distance_threshold = -480
    blink_distance_threshold = 25
    dispersion_threshold = 150
    dispersion_window_size = 11

    gaze_x, gaze_y = np.array(data["average_X"].tolist()), np.array(data["average_Y"].tolist())
    speed = np.array(data["speed"].tolist())
    time = np.diff(data["time"].tolist())
    time = np.insert(time, 0, 0)
    total_len = len(gaze_x)

    filtered_speed = savgol_filter(speed, window_length=min(11, len(speed)), polyorder=2)
    filtered_time = savgol_filter(time, window_length=min(11, len(time)), polyorder=2)
    ma_speed = moving_average(filtered_speed, min(101, len(filtered_speed)))
    # Gather fixation
    fix_start, fix_end = [], []
    for i in range(total_len):
        j = i
        while j >= 0 and np.linalg.norm([gaze_x[j] - gaze_x[i], gaze_y[j] - gaze_y[i]]) < np.clip(ma_speed[j], fixation_distance_threshold_min, fixation_distance_threshold_max):
            j -= 1
        k = i + 1
        while k < total_len and np.linalg.norm([gaze_x[k] - gaze_x[i], gaze_y[k] - gaze_y[i]]) < np.clip(ma_speed[k], fixation_distance_threshold_min, fixation_distance_threshold_max):
            k += 1
        if k - j > fixation_minimum_points:
            fix_start.append(j)
            fix_end.append(k)
            i = k + fixation_seek_points
        else:
            i = k

    speed_peaks, _ = find_peaks(filtered_speed, height=25)
    # -2 - Unknown, -1 - Noise, 0 - Fixation, 1 - Saccade, 2 - Return Sweep, 3 - Blink
    labels = [-1] * total_len
    last_end = 0
    for start, end in list(zip(fix_start, fix_end)):
        # Deal with the gap segment between fixation
        if start > last_end:
            # Check if there's a speed peak in the segment
            for p in speed_peaks:
                if p > last_end and p < start:
                    # Down and Up -> Blink
                    part_y_max_id = last_end + np.argmax(gaze_y[last_end:start])
                    if gaze_y[part_y_max_id] - gaze_y[max(last_end - slice_tolerance, 0)] > blink_distance_threshold and gaze_y[part_y_max_id] - gaze_y[min(start + slice_tolerance, total_len - 1)] > blink_distance_threshold:
                        for i in range(last_end, start):
                            labels[i] = 3
                    # Leftward -> Return Sweep
                    elif gaze_x[start] - gaze_x[last_end] < return_sweep_distance_threshold:
                        for i in range(last_end, start):
                            labels[i] = 2
                    # Otherwise saccade
                    else:
                        for i in range(last_end, start):
                            labels[i] = 1
                    break
            # Otherwise saccade
            else:
                for i in range(last_end, start):
                    labels[i] = 1
            # plt.xlim(0, 1920)
            # plt.ylim(-1200, 0)
            # plt.axis("scaled")
            # plt.scatter(gaze_x[last_end:start], -gaze_y[last_end:start])
            # for i in range(last_end, start):
            #     plt.text(gaze_x[i], -gaze_y[i], s=str(i))
            # print(labels[last_end])
            # plt.show()
        # Label fixation
        for i in range(start, end):
            labels[i] = 0
        last_end = end
    
    # DeNoise Large Dispersion  
    for i in range(total_len - dispersion_window_size):
        if np.max(gaze_y[i:i + dispersion_window_size]) - np.min(gaze_y[i:i + dispersion_window_size]) > dispersion_threshold:
            for i in range(i, i + dispersion_window_size):
                labels[i] = -1

    # fig, axes = plt.subplots(1, 2)
    # y_kernel_1 = np.array([-1, -1, -1, -1, 8, -1, -1, -1, -1])
    # y_kernel_2 = np.array([1, -1, 1, -1, 0, 1, -1, 1, -1])
    # conv_res = np.convolve(gaze_y, y_kernel_1, mode="same") - np.convolve(gaze_y, y_kernel_2, mode="same")
    # peaks, _ = find_peaks(conv_res, height=150)
    # peaks, _ = find_peaks(filtered_time, height=35000)
    # conv_res = (conv_res - np.min(conv_res)) / (np.max(conv_res) - np.min(conv_res))
    # axes[0].scatter(gaze_x, -gaze_y, c='black', s=0.5, alpha=conv_res)
    # axes[0].plot(conv_res)
    # axes[0].scatter(gaze_x, -gaze_y, c=[0.8, 0.8, 0.8], s=0.5)
    # axes[0].plot(gaze_x, -gaze_y, c=[0.8, 0.8, 0.8])
    # for i in range(total_len):
    #     if labels[i] == 0:
    #         axes[0].scatter(gaze_x[i], -gaze_y[i], c='#FFA500', s=0.5)
    #     if labels[i] == 1:
    #         axes[0].scatter(gaze_x[i], -gaze_y[i], c='#800080', s=0.5)
    #     if labels[i] == 2:
    #         axes[0].scatter(gaze_x[i], -gaze_y[i], c='#00b8ff', s=0.5)
    #     if labels[i] == 3:
    #         axes[0].scatter(gaze_x[i], -gaze_y[i], c='#FFC0CB', s=0.5)
    # axes[1].plot(filtered_speed)
    # axes[1].plot(conv_res)
    # for j in peaks:
    #     axes[1].scatter(j, conv_res[j])
    #     axes[0].scatter(gaze_x[j], -gaze_y[j])
    
    # plt.show()

    return labels


def pre_process_data(df_reading_data):
    df_reading_data_group_by_matrix_x = df_reading_data.groupby("matrix_x")
    data_list = []
    for matrix_x, df_matrix_x in df_reading_data_group_by_matrix_x:
        x_list = df_matrix_x["gaze_x"].tolist()
        y_list = df_matrix_x["gaze_y"].tolist()
        time_list = df_matrix_x["time"].tolist()
        # 反向遍历time_list，修改time_list和y_list的值。
        for i in range(len(time_list) - 1, -1, -1):
            time_list[i] = (time_list[i] - time_list[0]) * 1000

        x_list = np.array(x_list)
        y_list = np.array(y_list)

        data = pd.DataFrame({
            "average_X": x_list,
            "average_Y": y_list,
            "speed": df_matrix_x["speed"],
            "time": time_list
        })
        data_list.append(data)

    return data_list


def get_data_label(combined_reading_data_list=None):
    if combined_reading_data_list is None:
        reading_data_list = read_files.read_reading_data()

        combined_reading_data_list = []
        for file_index in range(len(reading_data_list)):
            reading_data_sorted_combined = read_files.sort_reading_data_by_para_id(reading_data_list[file_index])
            combined_reading_data_list.append(reading_data_sorted_combined)

    pre_processed_reading_data_list = []
    for file_index in range(len(combined_reading_data_list)):
        data = pre_process_data(combined_reading_data_list[file_index])
        pre_processed_reading_data_list.append(data)

    args_list = []
    for file_index in range(len(pre_processed_reading_data_list)):
        args = []
        for para_index in range(len(pre_processed_reading_data_list[file_index])):
            args.append((pre_processed_reading_data_list[file_index][para_index], file_index, para_index))
        args_list.append(args)

    labels_list = []
    # for file_index in range(len(pre_processed_reading_data_list)):
    #     labels = []
    #     for para_index in range(len(pre_processed_reading_data_list[file_index])):
    #         print(file_index, para_index)
    #         label = event_detection(pre_processed_reading_data_list[file_index][para_index])
    #         labels.append(label)
    #     labels_list.append(labels)
    with Pool(16) as p:
        for file_index in range(len(pre_processed_reading_data_list)):
            labels = p.starmap(event_detection, args_list[file_index])
            labels_list.append(labels)

    return labels_list













