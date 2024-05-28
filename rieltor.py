import datetime
import sys
import traceback

import pandas
import asyncio
import aiohttp
import pandas as pd
import requests
import sqlalchemy
from bs4 import BeautifulSoup as BS
from fake_useragent import UserAgent
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy import create_engine
from logs import Logger
from predictor import Predictor

predictor = Predictor()
logger = Logger('logs.txt')
try:
    DATABASE_URL = f"postgresql://postgres:admin@172.18.0.2:5432/postgres"
    # DATABASE_URL = f"postgresql://postgres:admin@localhost:5433/postgres"
    engine = create_engine(DATABASE_URL)
except:
    logger.add_log(
        message='Error on connecting to db',
        level='ERROR',
        data={'exception': traceback.format_exc()}
    )


async def domria():
    data = []
    staturl = "https://rieltor.ua/kiev/flats-rent/?page=1"
    agent = {'User-Agent': UserAgent().random}
    re = requests.get(staturl)
    m = BS(re.text, 'lxml')
    maxp = m.find('li', {'class': 'last'}).find('a', {'class': 'pager-btn'}).text
    fp = 1
    maxp = int(maxp)
    lp = maxp
    print(lp)

    for p in range(fp, lp + 1):
        try:
            print(p)
            async with aiohttp.ClientSession() as session:
                url = f"https://rieltor.ua/kiev/flats-rent/?page={p}"
                async with session.get(url,
                                       headers=agent) as rp:
                    r = await aiohttp.StreamReader.read(rp.content)
                    html = BS(r, 'lxml')
                    aps = html.findAll('div', {'class': 'catalog-card'})

                    for ap in aps:
                        row = {}
                        row['link'] = ap.find('a', {'class': 'catalog-card-media'}).get('href')
                        row['price'] = ap.find('strong', {'class': 'catalog-card-price-title'}).text
                        row['address'] = ap.find('div', {'class': 'catalog-card-address'}).text
                        row['rooms'] = ap.find('div', {'class': 'catalog-card-details'}).findAll('span', {'class': ''})[
                            0].text
                        row['sizes'] = ap.find('div', {'class': 'catalog-card-details'}).findAll('span', {'class': ''})[
                            1].text
                        row['floor'] = ap.find('div', {'class': 'catalog-card-details'}).findAll('span', {'class': ''})[
                            2].text
                        location = ap.findAll('a', {'data-analytics-event': 'card-click-region'})
                        row['district'] = location[1].text.strip() if len(location) > 1 else None
                        row['tags'] = [i.text.strip() for i in
                                       ap.find('div', {'class': 'catalog-card-chips'}).findAll('a')]
                        description = ap.find('div', {'class': 'catalog-card-description'})
                        row['description'] = description.find('span').text.strip() if description is not None else None
                        data.append(row)
        except Exception as e:
            logger.add_log(
                message='Exception in extracting data',
                level='ERROR',
                data={'exception': traceback.format_exc()}
            )

    try:
        df = pandas.DataFrame(data)
        df['currency'] = df['price'].str.split(' ').str.get(-1).str.split('/').str.get(0).str.strip()
        df['price'] = df['price'].str.replace(r'\D', '', regex=True).astype('int')
        df['max_floor'] = df['floor'].str.split(' ').str.get(3).astype('int')
        df['floor'] = df['floor'].str.split(' ').str.get(1).astype('int')
        df['total_area'] = df['sizes'].str.split(' / ').str.get(0).astype('float')
        df['kitchen_area'] = df['sizes'].str.split(' / ').str.get(2).str.replace(r'[^\d\.]', '',
                                                                                 regex=True).astype('float')
        df['rooms'] = pd.to_numeric(df['rooms'].str.split(' ').str.get(0), errors='coerce').fillna(1)
        df['district'] = df['district'].str.split(' ').str.get(0)
        df['animals'] = df['tags'].map(lambda x: 'Можна з тваринами' in x)
        df['source'] = 'rieltor'
        df = df.drop(columns=['sizes', 'tags'])
        print(len(df))
        load_to_db(df)
    except Exception as e:
        logger.add_log(
            message='Exception in transforming and loading data',
            level='ERROR',
            data={'exception': traceback.format_exc()}
        )


def load_to_db(df):
    df['prediction'], vectors = predictor.predict(df.copy())
    df.drop_duplicates(subset=['link']).set_index('link').to_sql('ads', engine, if_exists='append',
                                                                 method=postgres_upsert)
    vectors.drop_duplicates(subset=['link']).set_index('link').to_sql('vectors', engine, if_exists='append',
                                                                      method=postgres_upsert)


def postgres_upsert(table, conn, keys, data_iter):
    data = [dict(zip(keys, row)) for row in data_iter]

    insert_statement = insert(table.table).values(data)
    upsert_statement = insert_statement.on_conflict_do_update(
        constraint=f"{table.table.name}_pkey",
        set_={c.key: c for c in insert_statement.excluded},
    )
    conn.execute(upsert_statement)


if __name__ == '__main__':
    start = datetime.datetime.now()
    loop = asyncio.get_event_loop()
    loop.run_until_complete(domria())
    print(datetime.datetime.now() - start)
