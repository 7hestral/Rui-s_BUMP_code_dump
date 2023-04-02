from src.utils import data_load
import pandas as pd
from src.s3_utils import pandas_from_csv_s3
import numpy as np
import os
from datetime import datetime, timedelta
from utils import get_survey_question
data = data_load(data_keys={'bodyport', 'oura_activity', 'oura_sleep', "surveys", "birth"})



data['bodyport']['subsource'].unique()
# subsource is only `weight`

data_bodyport = data['bodyport']
users = data_bodyport['user_id'].unique().tolist()
most_data_user = -1
most_data = -1
for u in users:
    num_data = len(data_bodyport[data_bodyport['user_id'] == u])
    if num_data > most_data:
        most_data = num_data
        most_data_user = u

# userid 74 has the most data

list_users_above_criteria = [
    1032,
    581,
    407,
    290,
    1436,
    1000,
    95,
    1386,
    1431,
    992,
    1717,
    1441,
    122,
    977,
    293,
    114,
    1700,
    1696,
    1744,
    622,
    291,

    1661,

    173,
    192,
    1373,
    84,
    1425,
    1393,
    1432,
    1378,
    225,
    1753,
    2084,
    969,
    280,
    99,
    53,
    983,
    2068,
    193,
    2056,
    289,
    2016,
    2109,
    127,
    1437,
    1995,
    1706,
    1966,
    2015,
    186,
    137,
    1658,
    2083,
    1383,
    429,
    279]

def generate_csv_for_user(selected_user, preset_start_date=datetime(2009, 10, 12, 10, 10), preset_end_date=datetime(2030, 10, 12, 10, 10), file_name=''):
    import numpy as np
    import pandas as pd
    # selected_user = 1441
    print(f"Curr user: {selected_user}")
    root_folder = file_name
    selected_data_bodyport = data_bodyport[data_bodyport['user_id'] == selected_user][['date', 
    # 'impedance_ratio', 
    # 'peripheral_fluid', 
    'impedance_mag_1_ohms', 'impedance_phase_1_degs', 
    'weight_kg']].groupby("date", as_index = False).mean()
    selected_data_bodyport.to_csv('/mnt/test4.csv')

    data_oura_activity = data['oura_activity']
    data_oura_sleep = data['oura_sleep']
    data_survey = data['surveys']
    survey_question_str = 'swollen'
    data_edema = get_survey_question(data_survey, survey_question_str)

    selected_data_edema = data_edema[data_edema['user_id'] == selected_user][['date', 'answer_text']].groupby("date", as_index = False).last()
    selected_data_edema['answer_text'] = selected_data_edema['answer_text'].astype('int')
    selected_data_edema.to_csv('/mnt/test3.csv')
    selected_data_oura_sleep = data_oura_sleep[data_oura_sleep['user_id'] == selected_user][['breath_average', 'date']]
    selected_data_oura_activity = data_oura_activity[data_oura_activity['user_id'] == selected_user][[
    # 'cal_active',
    'cal_total',
    'daily_movement',
    'high',
    'inactive',
    #  'inactivity_alerts',
    'low',
    'medium',
    # 'met_min_high',
    # 'met_min_inactive',
    # 'met_min_low',
    # 'met_min_medium',
    'non_wear',
    'rest',
    #  'rest_mode_state',
    #  'score',
    #  'score_meet_daily_targets',
    #  'score_move_every_hour',
    #  'score_recovery_time',
    #  'score_stay_active',
    #  'score_training_frequency',
    #  'score_training_volume',
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
    def na_rate(df):
        if not len(df): return 1
        return np.max(np.sum(df.isna()))/len(df)



    # # union
    # overall_min_date = np.min([get_min_date(selected_data_bodyport), get_min_date(selected_data_oura_activity)])
    # overall_max_date = np.max([get_max_date(selected_data_bodyport), get_max_date(selected_data_oura_activity)])
    # print('get_min_date(selected_data_bodyport)', get_min_date(selected_data_bodyport))
    # print(selected_data_bodyport)
    # print(get_min_date(selected_data_oura_activity))
    # intercept
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
    # selected_data_bodyport['date'] = selected_data_bodyport['date'].astype('datetime64')
    # selected_data_oura_activity['date'] = selected_data_oura_activity['date'].astype('datetime64')
    # selected_data_edema['date'] = selected_data_edema['date'].astype('datetime64')
    # selected_data_oura_sleep
    selected_data_edema.to_csv('/mnt/test1.csv')
    selected_data_edema = change_date_type(selected_data_edema)
    # print("selected_data_edema shape: ", selected_data_edema.to_numpy().shape)
    selected_data_edema.to_csv('/mnt/test.csv')
    selected_data_oura_activity = change_date_type(selected_data_oura_activity)
    selected_data_oura_sleep = change_date_type(selected_data_oura_sleep)
    selected_data_bodyport = change_date_type(selected_data_bodyport)

    # selected_data_bodyport = pd.merge(date_df, selected_data_bodyport, how='left')
    # selected_data_oura_activity = pd.merge(date_df, selected_data_oura_activity, how='left')

    unimputed_df = pd.DataFrame()
    unimputed_df['date'] = date_range
    unimputed_df = pd.merge(unimputed_df, selected_data_bodyport, how='left')
    unimputed_df = pd.merge(unimputed_df, selected_data_oura_activity, how='left')
    unimputed_df = pd.merge(unimputed_df, selected_data_oura_sleep, how='left')
    unimputed_df = pd.merge(unimputed_df, selected_data_edema, how='left')

    np.sum(unimputed_df.isna())

    if na_rate(selected_data_oura_activity) > 0.4:
        print("More than 0.4 missingness Oura activity")
        return
    if na_rate(selected_data_bodyport) > 0.4:
        print("More than 0.4 missingness bodyport")
        return
    if na_rate(selected_data_oura_sleep) > 0.4:
        print("More than 0.4 missingness Oura activity")
        return
    if na_rate(selected_data_edema) > 0.4:
        print("More than 0.4 missingness bodyport")
        return
    # fill missing days with randomly selected days from previous 14 days window
    np.random.seed(90)
    # activity_mask = selected_data_oura_activity.drop('date', axis=1).isna().all(1)
    # bodyport_mask = selected_data_bodyport.drop('date', axis=1).isna().all(1)
    
    def random_fill_missing_day(df, mask, window=30):
        if 'date' in df.columns.to_list(): 
            df = df.drop('date', axis=1)
        for i in range(len(mask)):
            if mask[i]:
                if i < window: # do random filling
                    available_indice = np.where(mask == False)[0]
                    random_observation = df.iloc[np.random.choice(available_indice)]
                else:
                    random_index = np.random.randint(low=1, high=window)
                    random_observation = df.iloc[i-random_index]
                df.iloc[i] = random_observation
        return df
    
    def get_random_filled_df(df): 
        result = random_fill_missing_day(df.drop('date', axis=1), df.drop('date', axis=1).isna().all(1))
        result['date'] = date_range
        return result
    # random_filled_oura_activity = random_fill_missing_day(selected_data_oura_activity.drop('date', axis=1), activity_mask)
    # #np.sum(out.isna().all(1))
    # random_filled_bodyport = random_fill_missing_day(selected_data_bodyport.drop('date', axis=1), bodyport_mask)
    # random_filled_oura_activity['date'] = date_range
    # random_filled_bodyport['date'] = date_range

    # random_filled_oura_sleep = random_fill_missing_day(selected_data_oura_sleep.drop('date', axis=1), activity_mask)
    # #np.sum(out.isna().all(1))
    # random_filled_bodyport = random_fill_missing_day(selected_data_bodyport.drop('date', axis=1), bodyport_mask)
    # random_filled_oura_activity['date'] = date_range
    # random_filled_bodyport['date'] = date_range
    random_filled_bodyport = get_random_filled_df(selected_data_bodyport)
    random_filled_oura_activity = get_random_filled_df(selected_data_oura_activity)
    random_filled_oura_sleep = get_random_filled_df(selected_data_oura_sleep)
    random_filled_edema = get_random_filled_df(selected_data_edema)

    overall_df = pd.merge(random_filled_oura_activity, random_filled_bodyport, how='left')
    overall_df = pd.merge(overall_df, random_filled_oura_sleep, how='left')
    overall_df = pd.merge(overall_df, random_filled_edema, how='left')
    if max(random_filled_edema['answer_text']) == 7.0:
        print("!!!!!!!7.0")
    print("max edema", max(random_filled_edema['answer_text']))



    import seaborn as sns
    import matplotlib.pyplot as plt
    try:
        if unimputed_df['impedance_ratio']:
            # overall_df['mask'] = unimputed_df['impedance_ratio'].isna()
            overall_df.loc[overall_df['impedance_ratio'] > 10,'impedance_ratio'] = np.nan
    except:
        pass
    
    np.sum(overall_df.isna())

    overall_df.describe()

    # hyperimpute requires python 3.7
    from hyperimpute.plugins.imputers import Imputers
    import numpy as np
    import pandas as pd
    import hyperimpute as hp

    imputers = Imputers()

    imputers.list()
    method = 'hyperimpute'
    plugin = Imputers().get(method)

    X = overall_df.drop('date', axis=1)# .drop('mask', axis=1)
    # print(np.sum(X.isna()))
    # print(X.shape)
    out = plugin.fit_transform(X.copy())
    if not os.path.exists(os.path.join('/', 'mnt', 'results', root_folder)):
        os.mkdir(os.path.join('/', 'mnt', 'results', root_folder))
    hyperimputed_df_without_date = out
    hyperimputed_df_without_date.to_csv(f'/mnt/results/{root_folder}/user_{selected_user}_{file_name}_hyperimpute.csv', index=False, header=False)

    hyperimputed_df_with_date = hyperimputed_df_without_date.copy()
    hyperimputed_df_with_date['date'] = date_range
    hyperimputed_df_with_date.to_csv(f'/mnt/results/{root_folder}/user_{selected_user}_{file_name}_hyperimpute_with_date.csv', index=False)

    overall_df_without_date = overall_df.drop('date', axis=1)
    overall_df_without_date.to_csv(f'/mnt/results/{root_folder}/user_{selected_user}_{file_name}_rand_filled.csv', index=False, header=False)
    return True

def generate_puqe_survey_for_user(selected_user):
    # handle survey
    df_survey = data['surveys']
    df_puqe = data['surveys'].loc[data['surveys']['title']=='PUQE']
    selected_user_puqe = df_puqe[df_puqe['user_id'] == selected_user]


    answer_dict = {
        'No': 1,
        'Yes': 2,
        'Not at all': 1,
        '1 hour or less': 2,
        '2 to 3 hours':3, 
        '4 to 6 hours':4, 
        '6 or more hours':5,
        '1-2 times': 2,
        '3-4 times': 3,
        '5-6 times': 4,
        '7 or more times': 5
    }

    answer_catagory = {
        (0, 6) : 0,
        (7, 12): 1,
        (13, 21): 2
    }
    def map_levels(x, map_dict):
        for key in map_dict:
            if isinstance(x, str):
                if x == key:
                    return map_dict[key]
            else:
                if x >= key[0] and x <= key[1]:
                    return map_dict[key]
    # 53, 54, 132, 404
    # 404: 'During the past week, did you experience any nausea?'
    # 132: 'On average in a day, for how long do you feel nauseated or sick to your stomach?'
    # 53: 'On average in a day, how many times do you vomit or throw up?'
    # 54: 'On average in a day, how many times have you had retching or dry heaves without bringing anything up?'

    # for qid in selected_user_puqe['question_id'].unique():
    #     question_df = selected_user_puqe.loc[selected_user_puqe['question_id'] == qid]
    #     question_df['answer_text'] = question_df['answer_text'].astype(object)
    #     print(question_df['answer_text'])
    #     if qid == 404:
    #         question_df.replace({"answer_text": answer_dict1}, inplace=True)
    #     elif qid == 132:
    #         question_df.replace({"answer_text": answer_dict2}, inplace=True)
    #     else:
    #         question_df.replace({"answer_text": answer_dict3}, inplace=True)
    selected_user_puqe['answer_text'] = selected_user_puqe['answer_text'].astype(object)
    selected_user_puqe.replace({"answer_text": answer_dict}, inplace=True)
    aggregated_user_puqe = selected_user_puqe.groupby('date', as_index=False).sum()
    for k in answer_catagory:
        aggregated_user_puqe['answer_text'] = np.where(aggregated_user_puqe['answer_text'].between(k[0],k[1]), answer_catagory[k], aggregated_user_puqe['answer_text'])

    aggregated_user_puqe.to_csv(f'/mnt/results/user_{selected_user}_puqe.csv', index=False, header=True)




available_user = []
for user in data['birth']['user_id'].unique():
    user = int(user)
    df_birth = data['birth']
    if len(df_birth[df_birth.user_id == user].birth_date):
        curr_birth_date = pd.to_datetime(df_birth[df_birth.user_id == user].birth_date.values[0])
        # only interested in the 3rd trimester
        third_trimester_start_date = curr_birth_date - timedelta(days=91)
        if generate_csv_for_user(user, preset_start_date=third_trimester_start_date, preset_end_date=curr_birth_date, file_name='edema_pred'):
            available_user.append(user)

        # generate_puqe_survey_for_user(user)
print(available_user)
    # for each csv, normalize 