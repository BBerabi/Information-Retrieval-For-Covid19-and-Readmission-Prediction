import pandas as pd
import numpy as np
import yaml


def group_diagnoses(df):
    # Create mapping from
    l_old = []
    l_new = []

    idx = 0
    tmp_list1 = list(range(390, 460))
    tmp_list1 += [785]
    tmp_list2 = [idx] * len(tmp_list1)
    idx += 1

    l_old = [*l_old, *tmp_list1]
    l_new = [*l_new, *tmp_list2]

    tmp_list1 = list(range(460, 520))
    tmp_list1 += [786]
    tmp_list2 = [idx] * len(tmp_list1)
    idx += 1

    l_old = [*l_old, *tmp_list1]
    l_new = [*l_new, *tmp_list2]

    tmp_list1 = list(range(520, 579))
    tmp_list1 += [787]
    tmp_list2 = [idx] * len(tmp_list1)
    idx += 1

    l_old = [*l_old, *tmp_list1]
    l_new = [*l_new, *tmp_list2]

    tmp_list1 = [str(i) for i in list(np.arange(250, 251, 0.01))]
    tmp_list2 = [idx] * len(tmp_list1)
    idx += 1
    l_old = [*l_old, *tmp_list1]
    l_new = [*l_new, *tmp_list2]

    tmp_list1 = range(800, 1000)
    tmp_list2 = [idx] * len(tmp_list1)
    idx += 1
    l_old = [*l_old, *tmp_list1]
    l_new = [*l_new, *tmp_list2]

    tmp_list1 = range(710, 740)
    tmp_list2 = [idx] * len(tmp_list1)
    idx += 1
    l_old = [*l_old, *tmp_list1]
    l_new = [*l_new, *tmp_list2]

    tmp_list1 = list(range(580, 630))
    tmp_list1 += [788]
    tmp_list2 = [idx] * len(tmp_list1)
    idx += 1
    l_old = [*l_old, *tmp_list1]
    l_new = [*l_new, *tmp_list2]

    tmp_list1 = range(140, 240)
    tmp_list2 = [idx] * len(tmp_list1)
    idx += 1
    l_old = [*l_old, *tmp_list1]
    l_new = [*l_new, *tmp_list2]

    l_old = [str(i) for i in l_old]
    d = dict(zip(l_old, l_new))

    df_new = df.copy()

    df_new = df_new.map(d)
    df_new = df_new.replace(df_new[pd.isna(df_new)], 8)
    df_new = df_new.astype(int)
    return df_new





def preprocessing(df,path_yaml="./task1/data/category_names.yaml",path_csv="./task1/data/dataset.csv"):
    # Drop unnecessary columns
    # weights, payer_code, diag_1_desc, diag_2_desc, diag_3_desc
    # df.drop(labels=['weight', 'payer_code', 'diag_1_desc', 'diag_2_desc', 'diag_3_desc'], axis=1, inplace=True)
    # df.drop(labels=['patient_nbr', 'encounter_id', 'weight', 'payer_code'], axis=1, inplace=True)
    df.drop(labels=['patient_nbr','medical_specialty','encounter_id','weight', 'payer_code'], axis=1, inplace=True)


    # Fill missing values
    # Race -> filled with Caucasian
    tmp = df['race'].copy()
    tmp[tmp == '?'] = 'Caucasian'
    df['race'] = tmp

    # Diag_3 filled with 250
    tmp1 = df['diag_3'].copy()
    tmp1[tmp1 == '?'] = '250'
    df['diag_3'] = tmp1

    # Binarize
    df['readmitted'].loc[df['readmitted'] == "NO"] = '0'
    df["readmitted"].loc[df['readmitted'] == ">30"] = '1'
    df["readmitted"].loc[df['readmitted'] == "<30"] = '1'

    # Group Diagnoses
    df['diag_1'] = group_diagnoses(df['diag_1'])
    df['diag_2'] = group_diagnoses(df['diag_2'])
    df['diag_3'] = group_diagnoses(df['diag_3'])

    # Encode string data to numericals
    to_cat = list(df.select_dtypes(['object']).columns)
    df[to_cat] = df[to_cat].astype('category')
    # Category names
    cat_names = dict()
    for n in to_cat:
        tmp = list(df[n].cat.categories)
        cat_names[n] = tmp
    cat_columns = df.select_dtypes(['category']).columns
    df[cat_columns] = df[cat_columns].apply(lambda x: x.cat.codes)


    # Get Readmitted as labels
    '''label = df['readmitted']
    df.drop(labels=['readmitted'], axis=1, inplace=True)'''
    with open(path_yaml, 'w') as file:
        doc = yaml.dump(cat_names, file)
    df.to_csv(path_csv,index=False)

def getData(path_csv):
    df = pd.read_csv(path_csv)
    labels = df['readmitted']
    df.drop(labels=['readmitted'],axis=1,inplace=True)
    return df, labels




# data = pd.read_csv("./task1/data/diabetic_data.csv")
# preprocessing(data)
# # dataset, labels, category_names = preprocessing(data)
#
#
#
#
# check_df, check_labels = getData("./task1/data/dataset.csv")
# print(check_df)
# print(check_labels.value_counts())



