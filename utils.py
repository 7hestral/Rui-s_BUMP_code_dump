

import pandas as pd
import numpy as np

survey_question_to_question_id = {
    'nausea': 203,
    'fatigue': 204, 
    'mood': 205, 
    'breath': 206,
    'swollen': 207,
    'walk': 208,
    'remember': 209
}
def na_rate(df):
    if not len(df): return 1
    return np.max(np.sum(df.isna()))/len(df)
def merge_two_df_by_userid(user_id, df1, df2, start=None, end=None, how='inner'):
    df1 = df1[df1.user_id == user_id]#.dropna()
    df2 = df2[df2.user_id == user_id]#.dropna()
    
    new_df = pd.merge(df1, df2, on="date", how=how)

    # if "creation_date" in new_df.columns:
    #     for i in range(len(new_df)):
    #         new_df["creation_date"][i] = new_df.datetime.strptime(new_df["creation_date"][i], '%Y-%m-%d %H:%M:%S')
    # new_df.set_index(new_df["date"], inplace=True)
    # new_df.sort_index(inplace=True)
    
    if start and end:
        mask = (new_df['date'] > (start)) & (new_df['date'] <= (end))
        # mask = pd.to_datetime(new_df["date"]).between(start.astype(str)[0], end.astype(str)[0], inclusive=True)
        new_df = new_df[mask]
    return new_df

def get_survey_question(df_survey, survey_question_str):
    return df_survey[df_survey['question_id'] == survey_question_to_question_id[survey_question_str]]

Y_to_features = {
    "Y1": 'hr_average',
    "Y2": "rmssd",
    "Y3": "breath_average", 
    "Y4": "answer_text"
}
import networkx as nx
def generate_network_from_coeff_df(coeff_df, Y_to_features=Y_to_features):
    without_intercept = coeff_df
    print(without_intercept)
    G = nx.DiGraph()
    for name in Y_to_features:
        G.add_node(name, feature=Y_to_features[name])
    for name in without_intercept.columns:
        if name != 'Target':
            G.add_node(name, feature=Y_to_features[name[:2]])
    
    without_intercept.set_index('Target')
    for index, row in without_intercept.iterrows():
        for col in without_intercept.columns:
            if col != 'Target':
                if row[col] != 0:
                    print('edge added between', col, row['Target'])
                    print("with weight", row[col])
                    G.add_edge(col, row['Target'], weight=row[col])
    return G
