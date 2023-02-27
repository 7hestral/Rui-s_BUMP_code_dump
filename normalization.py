import numpy as np
import pandas as pd
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
        1700,
        1744,
        622,

        192,
        1373,
        84,
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
        2016,
        2109, 
        1995,
        1706,
        2015,
        186,
        137,
        1658,
        2083,
        1383,
        429,
        279]
def normalization():
    # find min and max across population
    _, feature_size = pd.read_csv(f'/mnt/results/user_{list_users_above_criteria[0]}_activity_bodyport_hyperimpute.csv', header=None).to_numpy().shape
    min_value_lst = [99999] * feature_size
    max_value_lst = [-99999] * feature_size
    for target_feature in range(feature_size):
        for user in list_users_above_criteria:
            curr_user_mat = pd.read_csv(f'/mnt/results/user_{user}_activity_bodyport_hyperimpute.csv', header=None).to_numpy()
            curr_max = np.max(curr_user_mat[:, target_feature])
            curr_min = np.min(curr_user_mat[:, target_feature])
            if min_value_lst[target_feature] > curr_min:
                min_value_lst[target_feature] = curr_min
            if max_value_lst[target_feature] < curr_max:
                max_value_lst[target_feature] = curr_max
    
    for user in list_users_above_criteria:
        curr_user_mat = pd.read_csv(f'/mnt/results/user_{user}_activity_bodyport_hyperimpute.csv', header=None).to_numpy()
        for target_feature in range(feature_size):
            curr_user_mat[:, target_feature] = (curr_user_mat[:, target_feature] - min_value_lst[target_feature]) / (max_value_lst[target_feature] - min_value_lst[target_feature])
        pd.DataFrame(curr_user_mat).to_csv(f'/mnt/results/user_{user}_activity_bodyport_hyperimpute_normalized.csv', header=None, index=None)

        

if __name__ == "__main__":
    normalization()
