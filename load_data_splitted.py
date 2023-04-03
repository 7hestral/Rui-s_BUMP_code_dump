from src.utils import data_load
import pandas as pd
from src.s3_utils import pandas_from_csv_s3
import numpy as np
import os
from datetime import datetime, timedelta
from utils import get_survey_question
data = data_load(data_keys={'bodyport', 'oura_activity', 'oura_sleep', "surveys"}, wave=7)
df_birth = data_load(data_keys={"birth"}, wave=5)['birth']



def get_usable_window(mask, window_size=21, tolerance=2):
    result_windows = []
    T = len(mask)
    i=0
    while i < (T - window_size + 1):
        curr_window = np.array(mask[i:i+window_size])
        # usable
        if np.sum(curr_window) < tolerance:
            result_windows.append((i, i+window_size))
            i += window_size
        else:
            # add tolerance window
            tolerance_window = np.array(mask[i:i+window_size+tolerance])
            miss_idx = np.where(tolerance_window == 1)[0]
            tolerance_counter = 0
            early_stop = False
            stop_idx = None
            for j in range(len(miss_idx) - 1):
                if miss_idx[j] + 1 == miss_idx[j + 1]:
                    tolerance_counter += 1
                else:
                    tolerance_counter = 0
                if tolerance_counter >= tolerance:
                    early_stop = True
                    stop_idx = j + 1
                    # break
            if early_stop:
                i += stop_idx + 1
            else:
                # usable
                result_windows.append((i, i+window_size))
                i += window_size
    return result_windows

def generate_csv_for_user(selected_user, preset_start_date=datetime(2009, 10, 12, 10, 10), preset_end_date=datetime(2030, 10, 12, 10, 10), file_name=''):

    # selected_user = 1441
    print(f"Curr user: {selected_user}")
    root_folder = file_name


    data_bodyport = data['bodyport']
    data_oura_activity = data['oura_activity']
    data_oura_sleep = data['oura_sleep']
    # data_survey = data['surveys']
    # survey_question_str = 'swollen'
    selected_data_bodyport = data_bodyport[data_bodyport['user_id'] == selected_user][['date', 
    # 'impedance_ratio', 
    # 'peripheral_fluid', 
    'impedance_mag_1_ohms', 'impedance_phase_1_degs', 
    'weight_kg']].groupby("date", as_index = False).first()
    edema_csv_path = f'/mnt/results/edema_coarse_label/user_{selected_user}_edema_coarse_label.csv'
    if not os.path.exists(edema_csv_path):
        print("Empty Edema")
        return
    selected_data_edema = pd.read_csv(edema_csv_path).groupby("date", as_index = False).first()

    selected_data_oura_sleep = data_oura_sleep[data_oura_sleep['user_id'] == selected_user][['breath_average', 'date']]
    selected_data_oura_activity = data_oura_activity[data_oura_activity['user_id'] == selected_user][[
    # 'cal_active',
    'cal_total',
    'daily_movement',
    'high',
    'inactive',
    'low',
    'medium',
    'non_wear',
    'rest',
    'steps',
    'date']]


    def get_min_date(df):
        return np.min(df['date'].astype('datetime64'))
    def get_max_date(df):
        return np.max(df['date'].astype('datetime64'))
    
    if not len(selected_data_bodyport):
        print("Empty bodyport")
        return
    if not len(selected_data_oura_activity):
        print("Empty Oura activity")
        return
    if not len(selected_data_edema):
        print("Empty Edema")
        return
    if not len(selected_data_oura_sleep):
        print("Empty Oura sleep")
        return

    overall_min_date = np.max(list(map(get_min_date, (selected_data_edema, selected_data_bodyport, selected_data_oura_activity, selected_data_oura_sleep))) + [preset_start_date])
    overall_max_date = np.min(list(map(get_max_date, (selected_data_edema, selected_data_bodyport, selected_data_oura_activity, selected_data_oura_sleep))) + [preset_end_date])
    
    date_range = pd.date_range(overall_min_date, overall_max_date, freq='d')
    print(overall_max_date-overall_min_date)
    if overall_max_date-overall_min_date < timedelta(days=10):
        return False
    date_df = pd.DataFrame()
    date_df['date'] = date_range
    date_df['date'] = date_df['date'].astype("datetime64")
    date_df.to_csv('/mnt/test2.csv')
    def change_date_type(df):
        df['date'] = df['date'].astype('datetime64')
        return pd.merge(date_df, df, how='left')

    selected_data_edema = change_date_type(selected_data_edema)
    selected_data_oura_activity = change_date_type(selected_data_oura_activity)
    selected_data_oura_sleep = change_date_type(selected_data_oura_sleep)
    selected_data_bodyport = change_date_type(selected_data_bodyport)

    unimputed_df = pd.DataFrame()
    unimputed_df['date'] = date_range
    unimputed_df = pd.merge(unimputed_df, selected_data_bodyport, how='left')
    unimputed_df = pd.merge(unimputed_df, selected_data_oura_activity, how='left')
    unimputed_df = pd.merge(unimputed_df, selected_data_oura_sleep, how='left')
    unimputed_df = pd.merge(unimputed_df, selected_data_edema, how='left')

    missingness_mask = unimputed_df.isna()
    missingness_mask = np.sum(missingness_mask, axis=1)
    missingness_mask[missingness_mask >= 1] = 1
    
    window_lst = get_usable_window(missingness_mask, tolerance=3)

    imputers = Imputers()
    imputers.list()
    method = 'hyperimpute'
    plugin = Imputers().get(method)



    if not os.path.exists(os.path.join('/', 'mnt', 'results', root_folder)):
        os.mkdir(os.path.join('/', 'mnt', 'results', root_folder))
    # save the windows
    for count, w in enumerate(window_lst):
        curr_window = unimputed_df[w[0]:w[1]]
        X = curr_window.drop('date', axis=1).drop('user_id', axis=1)
        # print(np.sum(X.isna()))
        # print(X.shape)
        curr_window = plugin.fit_transform(X.copy())
        curr_window.to_csv(f'/mnt/results/{root_folder}/user_{selected_user}_{file_name}_hyperimpute_slice{count}.csv', index=False, header=False)

        hyperimputed_df_with_date = curr_window.copy()
        hyperimputed_df_with_date['date'] = date_range[w[0]:w[1]]
        hyperimputed_df_with_date.to_csv(f'/mnt/results/{root_folder}/user_{selected_user}_{file_name}_hyperimpute_with_date_slice{count}.csv', index=False)

    return len(window_lst)


if __name__ == "__main__":


    # counter = 0
    # available_user = []
    # for user in df_birth['user_id'].unique():
    #     user = int(user)
    #     if len(df_birth[df_birth.user_id == user].birth_date):
    #         curr_birth_date = pd.to_datetime(df_birth[df_birth.user_id == user].birth_date.values[0])
    #         # only interested in the 3rd trimester
    #         third_trimester_start_date = curr_birth_date - timedelta(days=91)
    #         result = generate_csv_for_user(user, preset_start_date=third_trimester_start_date, preset_end_date=curr_birth_date, file_name='edema_pred_window')
    #         if result:
    #             available_user.append(user)
    #             counter += result


    # for each csv, normalize 