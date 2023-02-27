from src.utils import data_load
import pandas as pd
from src.s3_utils import pandas_from_csv_s3
import numpy as np

data = data_load(data_keys={'bodyport', 'oura_activity', 'oura_sleep', "surveys"})



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

def generate_csv_for_user(selected_user):
    import numpy as np
    import pandas as pd
    # selected_user = 1441
    selected_data_bodyport = data_bodyport[data_bodyport['user_id'] == selected_user][['date', 'impedance_ratio', 'peripheral_fluid', 'impedance_mag_1_ohms', 'impedance_phase_1_degs', 'weight_kg']].groupby("date", as_index = False).mean()


    data_oura_activity = data['oura_activity']
    data_oura_activity[data_oura_activity['user_id'] == selected_user]
    data_oura_sleep = data['oura_sleep']

    selected_data_oura_activity = data_oura_activity[data_oura_activity['user_id'] == selected_user][[
    'cal_active',
    'cal_total',
    'daily_movement',
    'high',
    'inactive',
    #  'inactivity_alerts',
    'low',
    'medium',
    'met_min_high',
    'met_min_inactive',
    'met_min_low',
    'met_min_medium',
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
        return np.min(df['date'])
    def get_max_date(df):
        return np.max(df['date'])
    
    if not len(selected_data_bodyport):
        print("Empty bodyport")
        return
    if not len(selected_data_oura_activity):
        print("Empty Oura activity")
        return
    def na_rate(df):
        return np.max(np.sum(df.isna()))/len(df)

    if na_rate(selected_data_oura_activity) > 0.4:
        print("More than 0.4 missingness Oura activity")
        return
    if na_rate(selected_data_bodyport) > 0.4:
        print("More than 0.4 missingness Oura activity")
        return

    # # union
    # overall_min_date = np.min([get_min_date(selected_data_bodyport), get_min_date(selected_data_oura_activity)])
    # overall_max_date = np.max([get_max_date(selected_data_bodyport), get_max_date(selected_data_oura_activity)])
    print('get_min_date(selected_data_bodyport)', get_min_date(selected_data_bodyport))
    print(selected_data_bodyport)
    print(get_min_date(selected_data_oura_activity))
    # intercept
    overall_min_date = np.max([get_min_date(selected_data_bodyport), get_min_date(selected_data_oura_activity)])
    overall_max_date = np.min([get_max_date(selected_data_bodyport), get_max_date(selected_data_oura_activity)])

    date_range = pd.date_range(overall_min_date, overall_max_date, freq='d')
    date_df = pd.DataFrame()
    date_df['date'] = date_range
    selected_data_bodyport['date'] = selected_data_bodyport['date'].astype('datetime64')
    selected_data_oura_activity['date'] = selected_data_oura_activity['date'].astype('datetime64')
    selected_data_bodyport = pd.merge(date_df, selected_data_bodyport, how='left')
    selected_data_oura_activity = pd.merge(date_df, selected_data_oura_activity, how='left')

    unimputed_df = pd.DataFrame()
    unimputed_df['date'] = date_range
    unimputed_df = pd.merge(unimputed_df, selected_data_bodyport, how='left')
    unimputed_df = pd.merge(unimputed_df, selected_data_oura_activity, how='left')
    np.sum(unimputed_df.isna())
    date_range

    # fill missing days with randomly selected days from previous 14 days window
    np.random.seed(90)
    activity_mask = selected_data_oura_activity.drop('date', axis=1).isna().all(1)
    bodyport_mask = selected_data_bodyport.drop('date', axis=1).isna().all(1)
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
                    # print(':',i-random_index)
                    # print(i)
                    # print(random_index)

                df.iloc[i] = random_observation

        return df
    random_filled_oura_activity = random_fill_missing_day(selected_data_oura_activity.drop('date', axis=1), activity_mask)
    #np.sum(out.isna().all(1))
    random_filled_bodyport = random_fill_missing_day(selected_data_bodyport.drop('date', axis=1), bodyport_mask)
    random_filled_oura_activity['date'] = date_range
    random_filled_bodyport['date'] = date_range

    overall_df = pd.merge(random_filled_oura_activity, random_filled_bodyport, how='left')



    import seaborn as sns
    import matplotlib.pyplot as plt

    # sns.set_theme(style='darkgrid')
    # sns.set(rc={'figure.figsize':(11.7,8.27)})
    # sns.scatterplot(x=overall_df["date"], y=overall_df["non_wear"], hue=unimputed_df['non_wear'].isna())
    overall_df['mask'] = unimputed_df['impedance_ratio'].isna()
    # graph = sns.scatterplot(data=overall_df, x="date", y="impedance_ratio", hue="mask")

    # df_birth = data_load(data_keys={'birth'})['birth']
    # df_birth[df_birth.user_id == selected_user].birth_date.values[0]
    # graph.axvline(pd.to_datetime(df_birth[df_birth.user_id == selected_user].birth_date.values[0]))

    overall_df.loc[overall_df['impedance_ratio'] > 10,'impedance_ratio'] = np.nan
    # graph = sns.scatterplot(data=overall_df, x="date", y="impedance_ratio", hue="mask")

    # df_birth = data_load(data_keys={'birth'})['birth']
    # df_birth[df_birth.user_id == selected_user].birth_date.values[0]
    # graph.axvline(pd.to_datetime(df_birth[df_birth.user_id == selected_user].birth_date.values[0]))

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

    # X = pd.DataFrame([[1, 1, 1, 1], [4, 5, np.nan, 4], [3, 3, 9, 9], [2, 2, 2, 2]])

    # print(np.sum(X.isna()))
    # out = plugin.fit_transform(X)

    X = overall_df.drop('date', axis=1).drop('mask', axis=1)
    print(np.sum(X.isna()))
    print(X.shape)
    out = plugin.fit_transform(X.copy())

    # print(method, out)

    hyperimputed_df_without_date = out
    hyperimputed_df_without_date.to_csv(f'/mnt/results/user_{selected_user}_activity_bodyport_hyperimpute.csv', index=False, header=False)

    hyperimputed_df_with_date = hyperimputed_df_without_date.copy()
    hyperimputed_df_with_date['date'] = date_range
    hyperimputed_df_with_date.to_csv(f'/mnt/results/user_{selected_user}_activity_bodyport_hyperimpute_with_date.csv', index=False)

    # g2 = sns.scatterplot(x=hyperimputed_df_with_date["date"], y=hyperimputed_df_without_date["impedance_ratio"], hue=overall_df['impedance_ratio'].isna())
    # g2.axvline(pd.to_datetime(df_birth[df_birth.user_id == selected_user].birth_date.values[0]))

    overall_df_without_date = overall_df.drop('date', axis=1)
    overall_df_without_date.to_csv(f'/mnt/results/user_{selected_user}_activity_bodyport_rand_filled.csv', index=False, header=False)

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





for user in list_users_above_criteria:
    generate_csv_for_user(user)


    # for each csv, normalize 