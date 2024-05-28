import traceback

import pandas as pd
from nltk.corpus import stopwords
import pickle
from nltk.tokenize import RegexpTokenizer
import numpy as np
from sqlalchemy import create_engine
import nltk
from sqlalchemy.dialects.postgresql import insert
import requests
nltk.download('stopwords')

class Predictor:
    def __init__(self):
        url = 'https://raw.githubusercontent.com/olegdubetcky/Ukrainian-Stopwords/main/ukrainian'
        r = requests.get(url)
        with open('/root/nltk_data/corpora/stopwords/ukrainian', 'wb') as f:
            f.write(r.content)
        with open('/app/catboost_model.pickle', 'rb') as f:
            self.model = pickle.load(f)
        with open('/app/districts_vectors.pickle', 'rb') as f:
            self.districts_vectors = pickle.load(f)
        with open('/app/word2vec_model.pickle', 'rb') as f:
            self.word2vec_model = pickle.load(f)

    @staticmethod
    def preprocess_description(doc_set):
        # initialize regex tokenizer
        tokenizer = RegexpTokenizer(r'\w+')
        stop = set(stopwords.words('ukrainian')).union(set(stopwords.words('russian'))).union({'та'})
        # list for tokenized documents in loop
        texts = []
        # loop through document list
        for i in doc_set:
            # clean and tokenize document string
            raw = i.lower()
            tokens = tokenizer.tokenize(raw)
            tokens = [i for i in tokens if not i in stop]
            texts.append(tokens)
        return texts

    def __transform(self, df):
        df['district'] = df['district'].fillna('None')
        df['description'] = df['description'].str.replace('.', '').str.replace(',', '').str.replace('\n', ' ').fillna(
            'None')
        df['animals'] = df['animals'].fillna(False)

        df['description'] = self.preprocess_description(df['description'].values)
        df_description_vector = df['description'].apply(lambda text:
                                                                     np.mean(
                                                                         [self.word2vec_model.wv[word] for word in text
                                                                          if word in self.word2vec_model.wv],
                                                                         axis=0))
        df_description_vector.loc[df_description_vector.isna()] = df_description_vector.loc[df_description_vector.isna()].map(
            lambda x: self.word2vec_model.wv['квартира'])
        df_description_vector = pd.DataFrame(df_description_vector.to_list(), columns=[f'description_{i}' for i in range(13)])
        df = pd.concat([df.reset_index(drop=True), df_description_vector], axis=1)

        df_district_vector = pd.DataFrame(df['district'].map(self.districts_vectors).to_list())
        df_district_vector.columns = [f'district_{i}' for i in range(4)]
        df = pd.concat([df.reset_index(drop=True), df_district_vector], axis=1)

        return df

    def predict(self, df):
        df = self.__transform(df)
        predictors = ['rooms',
                         'floor',
                         'max_floor',
                         'total_area',
                         'kitchen_area',
                         'animals'
                     ] + [i for i in df if 'description_' in i or 'district_' in i]

        return self.model.predict(df[predictors]), df[['link'] + [i for i in df.columns if 'district_' in i or 'description_' in i]]


if __name__ == '__main__':
    def postgres_upsert(table, conn, keys, data_iter):
        data = [dict(zip(keys, row)) for row in data_iter]

        insert_statement = insert(table.table).values(data)
        upsert_statement = insert_statement.on_conflict_do_update(
            constraint=f"{table.table.name}_pkey",
            set_={c.key: c for c in insert_statement.excluded},
        )
        conn.execute(upsert_statement)
    pred = Predictor()
    DATABASE_URL = f"postgresql://postgres:admin@localhost:5433/postgres"
    engine = create_engine(DATABASE_URL)
    df = pd.read_sql("""SELECT * FROM ads""", engine)
    print(len(df))
    df['prediction'], vectors = pred.predict(df.copy())
    # df.set_index('link').to_sql('ads', engine, if_exists='append', method=postgres_upsert)
    try:
        vectors.set_index('link').to_sql('vectors', engine, if_exists='append', method=postgres_upsert)
    except Exception as e:
        with open('test.txt', 'w') as f:
            f.write(traceback.format_exc())
        raise Exception from e


