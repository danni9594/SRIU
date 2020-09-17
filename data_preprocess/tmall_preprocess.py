import pandas as pd
import random
random.seed(1234)

# convert csv into pandas dataframe
df = pd.read_csv('../raw_datasets/user_log_format1.csv')
df['date'] = df['time_stamp'].apply(lambda x: 20140000 + x)
df = df[df['action_type'] == 0]  # keep click record only
df = df.drop(['cat_id', 'seller_id', 'brand_id', 'time_stamp', 'action_type'], axis=1)
df.columns = ['userId', 'itemId', 'date']  # rename

# extract 31 days data
start_date = 20141001
end_date = 20141031
df = df[(df['date'] >= start_date) & (df['date'] <= end_date)]

# remap id
user_id = sorted(df['userId'].unique().tolist())  # sort column
user_map = dict(zip(user_id, range(len(user_id))))  # create map, key is original id, value is mapped id starting from 0
df['userId'] = df['userId'].map(lambda x: user_map[x])  # map key to value in df

item_id = sorted(df['itemId'].unique().tolist())  # sort column
item_map = dict(zip(item_id, range(len(item_id))))  # create map, key is original id, value is mapped id starting from 0
df['itemId'] = df['itemId'].map(lambda x: item_map[x])  # map key to value in df

print(df.head(20))
print('num_users: ', len(user_map))  # 326984
print('num_items: ', len(item_map))  # 505708
print('num_records: ', len(df))  # 7624845


def gen_neg(num_items, pos_ls, num_neg):
    neg_ls = []
    for n in range(num_neg):  # generate num_neg
        neg = pos_ls[0]
        while neg in pos_ls:
            neg = random.randint(0, num_items - 1)
        neg_ls.append(neg)
    return neg_ls


# collect user history
df_user_gb = df.groupby(['userId'])
user_hists = []
count = 0
for row in df.itertuples():
    user_df = df_user_gb.get_group(row.userId)
    user_history_df = user_df[user_df['date'] <= row.date]
    userHist = user_history_df['itemId'].unique().tolist()
    user_hists.append(userHist)
    count += 1
    if count % 1000000 == 0:
        print('user done row {}'.format(count))

df['userHist'] = user_hists
print(df.head(20))

# negative sampling
num_neg = 5
df['neg_itemId_ls'] = df['userHist'].apply(lambda x: gen_neg(len(item_map), x, num_neg))

users, items, labels, dates = [], [], [], []
for row in df.itertuples():
    users.append(row.userId)
    items.append(row.itemId)
    labels.append(1)  # positive samples have label 1
    dates.append(row.date)
    for j in range(num_neg):
        users.append(row.userId)
        items.append(row.neg_itemId_ls[j])
        labels.append(0)  # negative samples have label 0
        dates.append(row.date)

df = pd.DataFrame({'userId': users,
                   'itemId': items,
                   'label': labels,
                   'date': dates})

print(df.head(20))
print(len(df))  # 2014.10.01-2014.10.31 5neg: 45749070

# save csv
# ['userId', 'itemId', 'label', 'date']
df.to_csv('../datasets/tmall_2014mth10_5neg.csv', index=False)
