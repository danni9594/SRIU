import pandas as pd
import random
random.seed(1234)

# convert csv into pandas dataframe
df = pd.read_csv('../raw_datasets/Sobazaar-hashID.csv.gz')

# preprocess
df['date'] = df['Timestamp'].apply(lambda x: int(''.join(c for c in x.split('T')[0] if c.isdigit())))  # extract date and convert to int
df = df.drop(['Action'], axis=1)  # drop useless
df.columns = ['timestamp', 'itemId', 'userId', 'date']  # rename
df = df[['userId', 'itemId', 'date', 'timestamp']]  # switch columns

# remap id
user_id = sorted(df['userId'].unique().tolist())  # sort column
user_map = dict(zip(user_id, range(len(user_id))))  # create map, key is original id, value is mapped id starting from 0
df['userId'] = df['userId'].map(lambda x: user_map[x])  # map key to value in df

item_id = sorted(df['itemId'].unique().tolist())  # sort column
item_map = dict(zip(item_id, range(len(item_id))))  # create map, key is original id, value is mapped id starting from 0
df['itemId'] = df['itemId'].map(lambda x: item_map[x])  # map key to value in df

print(df.head(20))
print('num_users: ', len(user_map))  # 17126
print('num_items: ', len(item_map))  # 24785
print('num_records: ', len(df))  # 842660

# sort records into 31 periods
df = df.sort_values(['timestamp']).reset_index(drop=True)
records_per_period = int(len(df) / 31)
df['index'] = df.index
df['period'] = df['index'].apply(lambda x: int(x / records_per_period) + 1)
df = df[df.period != 32]

count_per_period = df.groupby('period')['date'].count().reset_index().rename(columns={'date': 'count'})
min_per_period = df.groupby('period')['date'].min().reset_index().rename(columns={'date': 'min_date'})
max_per_period = df.groupby('period')['date'].max().reset_index().rename(columns={'date': 'max_date'})
period_df = count_per_period.merge(min_per_period, on='period', how='left').merge(max_per_period, on='period', how='left')

print(period_df)

df = df.drop(['index', 'date', 'timestamp'], axis=1)


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
    user_history_df = user_df[user_df['period'] <= row.period]
    userHist = user_history_df['itemId'].unique().tolist()
    user_hists.append(userHist)
    count += 1
    if count % 100000 == 0:
        print('user done row {}'.format(count))

df['userHist'] = user_hists
print(df.head(20))

# negative sampling
num_neg = 5
df['neg_itemId_ls'] = df['userHist'].apply(lambda x: gen_neg(len(item_map), x, num_neg))

users, items, labels, periods = [], [], [], []
for row in df.itertuples():
    users.append(row.userId)
    items.append(row.itemId)
    labels.append(1)  # positive samples have label 1
    periods.append(row.period)
    for j in range(num_neg):
        users.append(row.userId)
        items.append(row.neg_itemId_ls[j])
        labels.append(0)  # negative samples have label 0
        periods.append(row.period)

df = pd.DataFrame({'userId': users,
                   'itemId': items,
                   'label': labels,
                   'period': periods})

print(df.head(20))
print(len(df))  # 2014.09.01-2014.12.31: 5neg: 5055852

# save csv
# ['userId', 'itemId', 'label', 'period']
df.to_csv('../datasets/soba_2014mth09-12_5neg.csv', index=False)
