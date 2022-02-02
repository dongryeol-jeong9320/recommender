import random
import numpy as np
import pandas as pd
import torch

class MovieLensDataset(object):

    def __init__(self, file_dir, df_name='ratings.csv', 
                 min_counts=0, pos_rating=0, num_negatives=5,
                 zipfile=True, n_rows=None):
        self.min_counts = min_counts
        self.pos_rating = pos_rating
        self.num_negatives = num_negatives

        # Data Load
        self.data = self.data_load(file_dir, df_name, zipfile=zipfile)
        if isinstance(n_rows, int):
            self.data = self.data.iloc[:n_rows]

        # Preprocessing
        self.data = self.preprocess(self.data)
        self.train, self.test = self.train_test_split(self.data)

        # Negative Sampling
        self.user_pool = set(self.data['userId'])
        self.item_pool = set(self.data['movieId'])
        if self.num_negatives > 0:
            self.user_negatives, self.item_negatives, self.label_negatives = (
                self.get_negative_sampling(self.data)
            )
       
    
    def data_load(self, file_dir, df_name, zipfile=True):
        if zipfile == True:
            zf = ZipFile(file_dir)
            data = pd.read_csv(zf.open(df_name))
        else:
            data = pd.read_csv(file_dir)
        return data

    def preprocess(self, df):
        if self.min_counts > 0:
            user_counts = pd.DataFrame(df['userId'].value_counts())
            self.user_pool = set(user_counts.loc[user_counts['userId'] >= self.min_counts].index) # 최소 평점 개수 이상
            df = df.loc[df['userId'].isin(self.user_pool)]
            self.item_pool = set(df['movieId'].unique())
        
        user_to_idx = {u : i for i, u in enumerate(df['userId'].unique())}
        item_to_idx = {v : i for i, v in enumerate(df['movieId'].unique())}
        
        df['userId'] = df['userId'].apply(lambda x: user_to_idx[x])
        df['movieId'] = df['movieId'].apply(lambda x: item_to_idx[x])
        df['rating'] = np.where(df['rating'] >= self.pos_rating, 1, 0) # 연관성 판별 평점 기준
        return df[['userId', 'movieId', 'rating', 'timestamp']]

    def train_test_split(self, df):
        df['latest'] = df.groupby('userId')['timestamp'].rank(method='first', ascending=False)
        train = df.loc[df['latest'] > 1]
        test = df.loc[df['latest'] == 1]
        assert train['userId'].nunique() == test['userId'].nunique(), 'Not Match Train User with Test User'
        return train[['userId', 'movieId', 'rating']], test[['userId', 'movieId', 'rating']]
    
    def get_ml_datasets(self):
        user_train, item_train, label_train = list(self.train['userId']), list(self.train['movieId']), list(self.train['rating'])
        user_test, item_test, label_test = list(self.test['userId']), list(self.test['movieId']), list(self.test['rating'])
        n_neg_test = int(round(len(user_test) * (len(self.user_negatives) / len(user_train))))
        
        user_train += self.user_negatives
        item_train += self.item_negatives
        label_train += self.label_negatives

        indices = random.sample(k=n_neg_test, population=[i for i in range(len(self.user_negatives))])
        user_test += [self.user_negatives[idx] for idx in indices]
        item_test += [self.item_negatives[idx] for idx in indices]
        label_test += [self.label_negatives[idx] for idx in indices]

        random.shuffle(user_train)
        random.shuffle(item_train)
        random.shuffle(label_train)
    
        random.shuffle(user_test)
        random.shuffle(item_test)
        random.shuffle(label_test)

        trainset = NCFDataset(user_data=user_train, item_data=item_train, label_data=label_train)
        testset = NCFDataset(user_data=user_test, item_data=item_test, label_data=label_test)
        return trainset, testset

    def get_negative_sampling(self, df):
        
        neg_users, neg_items, neg_labels = [], [], []
        item_pool = set(df['movieId'])
        neg_counts = df.groupby('movieId').count()['userId'].to_dict()

        for u in df['userId'].unique():
            pos = set(df.loc[df['userId'] == u]['movieId'])
            neg_item_list = list(set(df['movieId']) - pos)
            neg_weights = np.array([neg_counts[i] ** 0.75 for i in neg_item_list])
            neg_weights /= sum(neg_weights)

            neg_users += [u] * self.num_negatives
            neg_items += list(np.random.choice(a=neg_item_list, size=self.num_negatives, p=neg_weights, replace=False))
            neg_labels += [0] * self.num_negatives
        return neg_users, neg_items, neg_labels
    
    def data_loader(self, batch_size=1, shuffle=True, num_workers=2):
        trainset, testset = self.get_ml_datasets()
        trainset = torch.utils.data.DataLoader(trainset, batch_size=batch_size, 
                                               shuffle=shuffle, num_workers=num_workers)
        testset = torch.utils.data.DataLoader(testset, batch_size=batch_size, 
                                               shuffle=shuffle, num_workers=num_workers)
        # return torch.utils.data.DataLoader(dataset, batch_size=self.num_ng_test+1, shuffle=False, num_workers=4)
        return trainset, testset


class NCFDataset(torch.utils.data.Dataset):

    def __init__(self, user_data, item_data, label_data):
        super(NCFDataset, self).__init__()
        self.user_data = user_data
        self.item_data = item_data
        self.label_data = label_data

    def __len__(self):
        return len(self.user_data)

    def __getitem__(self, idx):
        user = torch.tensor(self.user_data[idx], dtype=torch.long)
        item = torch.tensor(self.user_data[idx], dtype=torch.long)
        label = torch.tensor(self.label_data[idx], dtype=torch.float)
        return user, item, label



