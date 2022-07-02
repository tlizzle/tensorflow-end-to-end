import pandas as pd
import json
from clickhouse_driver import Client
from src.config import host

from tqdm import tqdm


class Encoder(object):
    def __init__(self, df):
        self.df = df
        
    def fit(self, encode_columns: None, min_freq= 30):
        id_encodding = {}
        i = 0
        if encode_columns:
            self.encode_columns = encode_columns
        else:
            self.encode_columns = self.df.columns

        for feature in self.encode_columns:
            feature_dict = {}
            tmp = self.df[feature].value_counts()
            encode_feature = tmp.index[tmp.ge(min_freq)].tolist()
            for key in encode_feature:
                feature_dict[key] = i
                i += 1
            feature_dict['<{}_unk>'.format(feature)] = i
            # i += 1
            id_encodding[feature] = feature_dict
            i = 0 
        self.id_encodding = id_encodding
        return self


    def transform(self):
        encoded = []
        for row in tqdm(zip(*[self.df[feature].values for feature in self.encode_columns]), total= self.df.shape[0]):
            encoded.append([self.id_mapping(feature, row[i]) for i, feature in enumerate(self.encode_columns)])
        return pd.DataFrame(encoded, columns=list(map(lambda x: 'encoded_' + x,  self.encode_columns)))
        

    def id_mapping(self, feature, category):
        if str(category) in self.id_encodding[feature]:
            return self.id_encodding[feature][str(category)]
        return self.id_encodding[feature]['<{}_unk>'.format(feature)]




        
if __name__ == '__main__':
    data = pd.read_csv('../resource/all_feat.csv')
    # client = Client(host=host, 
    #                 port= 9000, 
    #                 user= 'user1', 
    #                 password= '123456',
    #                 settings={'use_numpy': True})
    # data = client.query_dataframe('SELECT * FROM testing.winprice_estimation')
    # data.drop('date', inplace= True, axis= 1)

    # encoder = Encoder(data).fit(['city', 'statezip', 'country', 'weekday'])

    # output = encoder.transform()
    # output.encoded_city.value_counts()
    # output.encoded_statezip.value_counts()







