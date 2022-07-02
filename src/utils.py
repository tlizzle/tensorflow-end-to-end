from locale import normalize
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder
from clickhouse_driver import Client
from src.encoder import Encoder
from sklearn.preprocessing import StandardScaler, Normalizer
import pandas as pd
# tf.config.run_functions_eagerly(True)

# def prepar_data_set(data_df):
#     categoy_features = data_df.select_dtypes('object').columns
#     numerique_features = data_df.select_dtypes('number').columns
#     for col in categoy_features:
#         encoder = LabelEncoder()
#         data_df[col] = encoder.fit_transform(data_df[col])
#     return data_df,categoy_features,numerique_features
# train,categorical_features,num_features = prepar_data_set(data)


client = Client(host='localhost', 
                port= 9000, 
                user= 'user1', 
                password= '123456',
                settings={'use_numpy': True})
data = client.query_dataframe('SELECT * FROM testing.winprice_estimation')
data.drop(['date', 'street', 'country'], inplace= True, axis= 1)




categorical_features = data.select_dtypes('object').columns.tolist()
num_features = [c for c in data.select_dtypes('number').columns.tolist() if c != 'price']


encoder = Encoder(data).fit(categorical_features)
df2 = encoder.transform()

categorical_features = df2.columns.tolist()

s_scaler = StandardScaler()
df1 = pd.DataFrame(s_scaler.fit_transform(data[num_features]), columns= num_features)

# s_scaler = Normalizer()
# df1 = pd.DataFrame(s_scaler.fit_transform(data[num_features]), columns= num_features)



df = pd.concat([df1, df2], axis= 1)

df[num_features]
df[categorical_features]


y = data['price']


inputs = {}
concatenated_feature = []

for feature_name in num_features:
    if feature_name != 'price':
        inputs[feature_name] = tf.keras.layers.Input(shape= (1),\
                                        name= feature_name)
        concatenated_feature.append(inputs[feature_name])

for cat in categorical_features:
    vocab_size = df[cat].nunique()
    inputs[cat] = tf.keras.layers.Input(shape=(1,),name= cat)

    embed = tf.keras.layers.Embedding(input_dim= vocab_size,\
                                    output_dim= 30,\
                                    trainable=True, \
                                    embeddings_initializer=tf.initializers.random_normal, \
                                    name= cat + '_emb')

    embedding_lookup_weight = tf.keras.layers.Reshape(
        (30,))(embed(inputs[cat]))

    concatenated_feature.append(embedding_lookup_weight)


merge_models= tf.keras.layers.concatenate(concatenated_feature)

pre_preds = tf.keras.layers.BatchNormalization()(merge_models)
pre_preds = tf.keras.layers.Dense(256, \
                                activation='relu',
                               )(pre_preds)

pre_preds = tf.keras.layers.BatchNormalization()(pre_preds)
pre_preds = tf.keras.layers.Dense(128, \
                                activation='relu',
                               )(pre_preds)

pre_preds = tf.keras.layers.BatchNormalization()(pre_preds)
pre_preds = tf.keras.layers.Dense(64, \
                                activation='relu',
                               )(pre_preds)

pred = tf.keras.layers.Dense(1, \
                            activation="linear")(pre_preds)

model = tf.keras.models.Model(inputs= inputs,\
                                    outputs =pred)

model.compile(loss=tf.keras.losses.MeanSquaredError(),\
                    metrics=['mean_squared_error'],
                    optimizer='adam')

model.summary()


# input_dict= {
#     'bedrooms':df[['bedrooms']],
#     "bathrooms": df[['bathrooms']],
#     "sqft_living": df[['sqft_living']],
#     "sqft_lot": df[['sqft_lot']],
#     "floors": df[['floors']],
#     "waterfront": df[['waterfront']],
#     "view": df[['view']],
#     "condition": df[['condition']],
#     "sqft_above": df[['sqft_above']],
#     "sqft_basement": df[['sqft_basement']],
#     "yr_built": df[['yr_built']],
#     "yr_renovated": df[['yr_renovated']]
# }

input_dict= {
    'encoded_city':df[categorical_features[0]],
    'encoded_statezip':df[categorical_features[1]],
    'encoded_weekday':df[categorical_features[2]],
    'bedrooms':df[['bedrooms']],
    "bathrooms": df[['bathrooms']],
    "sqft_living": df[['sqft_living']],
    "sqft_lot": df[['sqft_lot']],
    "floors": df[['floors']],
    "waterfront": df[['waterfront']],
    "view": df[['view']],
    "condition": df[['condition']],
    "sqft_above": df[['sqft_above']],
    "sqft_basement": df[['sqft_basement']],
    "yr_built": df[['yr_built']],
    "yr_renovated": df[['yr_renovated']]
}

model.fit(
    input_dict,
    y,
    verbose= 1,
    epochs= 30,
    batch_size= 12,
    validation_steps= None,
)


model.predict(input_dict)
y

# 64597234564.62561
# 213475295232.0000
# 246587345436.97635
