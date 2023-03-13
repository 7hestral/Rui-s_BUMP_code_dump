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


def normalization(list_users, file_name):
    # find min and max across population
    _, feature_size = pd.read_csv(f'/mnt/results/{file_name}/user_{list_users[0]}_{file_name}_hyperimpute.csv', header=None).to_numpy().shape
    min_value_lst = [99999] * feature_size
    max_value_lst = [-99999] * feature_size
    for target_feature in range(feature_size):
        for user in list_users:
            curr_user_mat = pd.read_csv(f'/mnt/results/{file_name}/user_{user}_{file_name}_hyperimpute.csv', header=None).to_numpy()
            curr_max = np.max(curr_user_mat[:, target_feature])
            curr_min = np.min(curr_user_mat[:, target_feature])
            if min_value_lst[target_feature] > curr_min:
                min_value_lst[target_feature] = curr_min
            if max_value_lst[target_feature] < curr_max:
                max_value_lst[target_feature] = curr_max
    
    for user in list_users:
        curr_user_mat = pd.read_csv(f'/mnt/results/{file_name}/user_{user}_{file_name}_hyperimpute.csv', header=None).to_numpy()
        for target_feature in range(feature_size):
            curr_user_mat[:, target_feature] = (curr_user_mat[:, target_feature] - min_value_lst[target_feature]) / (max_value_lst[target_feature] - min_value_lst[target_feature])
        pd.DataFrame(curr_user_mat).to_csv(f'/mnt/results/{file_name}/user_{user}_{file_name}_hyperimpute_normalized.csv', header=None, index=None)

        

if __name__ == "__main__":
    
    list_users = [581, 407, 290, 1436, 1000, 95, 992, 1717, 293, 622, 291, 192, 1373, 225, 969, 280, 53, 983, 193, 186, 137, 1383, 429]
    list_users = [28, 30, 38, 40, 42, 53, 54, 55, 64, 66, 67, 68, 74, 94, 95, 118, 135, 137, 159, 1373, 1000, 174, 186, 190, 192, 193, 1021, 976, 972, 225, 1004, 1429, 234, 280, 290, 291, 293, 404, 407, 408, 410, 1047, 428, 429, 980, 581, 603, 604, 622, 734, 983, 966, 969, 985, 987, 989, 991, 992, 997, 1024, 1041, 1403, 1038, 1367, 1383, 1389, 1422, 1426, 1427, 1436, 1440, 1444, 1453, 1717]
    normalization(list_users, 'edema_pred')
