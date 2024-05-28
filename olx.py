import datetime
import traceback
import requests
import pandas as pd
from bs4 import BeautifulSoup
import re
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy import create_engine
from logs import Logger
from predictor import Predictor

predictor = Predictor()
logger = Logger('logs.txt')
try:
    DATABASE_URL = f"postgresql://postgres:admin@172.18.0.2:5432/postgres"
    engine = create_engine(DATABASE_URL)
except:
    logger.add_log(
        message='Error on connecting to db',
        level='ERROR',
        data={'exception': traceback.format_exc()}
    )

URL = "https://olx.ua/uk/nedvizhimost/kvartiry/dolgosrochnaya-arenda-kvartir/kiev/"


def get_max_page():
    r = requests.get(URL)
    soup = BeautifulSoup(r.content, "html.parser")
    return int(soup.findAll('a', class_='css-1mi714g')[-1].text)


def get_links(page):
    r = requests.get(URL + f'?page={page}')
    print(r.url)
    soup = BeautifulSoup(r.content, "html.parser")
    links = []
    news_elements = soup.find_all("a", class_="css-z3gu2d")
    for element in news_elements:
        link = element["href"]
        links.append(link)
    return set(links)


def get_page_content(link):
    try:
        if link[3:5] != 'uk':
            link = link[:3] + 'uk/' + link[3:]
        full_url = "https://www.olx.ua" + link
        page = requests.get(full_url)
        soup = BeautifulSoup(page.content, "html.parser")

        name = soup.find("h4", class_="css-1juynto").text if soup.find("h4", class_="css-1juynto") is not None else ''
        cost = soup.find("h3", class_="css-12vqlj3").text

        description = soup.find("div", class_="css-1t507yq er34gjf0").text if soup.find("div",
                                                                                        class_="css-1t507yq er34gjf0") is not None else None
        district = soup.find("ol", class_='css-xv75xi').text
        tags = [i.text for i in soup.find('ul', class_='css-sfcl1s').findAll('li')]
        return {'link': full_url, 'price': cost, 'description': name + '\n' + description, 'district': district,
                'tags': tags}
    except:
        logger.add_log(
            message='Error on loading from olx',
            level='ERROR',
            data={'exception': traceback.format_exc()}
        )


def parse_tags(x):
    fields = {'floor': None,
              'max_floor': None,
              'total_area': None,
              'kitchen_area': None,
              'animals': None,
              'rooms': None
              }

    for value in x:
        if 'Поверх:' in value or 'Этаж:' in value:
            fields['floor'] = int(re.sub('\D', '', value))
        if 'Поверховість:' in value or 'Этажность:' in value:
            fields['max_floor'] = int(re.sub('\D', '', value))
        if 'Загальна площа:' in value or 'Общая площадь:' in value:
            fields['total_area'] = float(re.sub('[^\d\.]', '', value))
        if 'Площа кухні:' in value or 'Площадь кухни:' in value:
            fields['kitchen_area'] = float(re.sub('[^\d\.]', '', value))
        if 'Домашні улюбленці:' in value or 'Домашние питомцы:' in value:
            fields['animals'] = 'Так' in value or 'Да' in value
        if 'Кількість кімнат:' in value or 'Количество комнат:' in value:
            fields['rooms'] = int(re.sub('\D', '', value))
    return fields


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


def main():
    max_page = get_max_page()
    total = pd.DataFrame()
    print(max_page)
    for page in range(1, max_page + 1):
        print(page)
        links = get_links(page)
        result = []
        for link in list(links):
            content = get_page_content(link)
            if content is not None:
                result.append(content)
        try:
            result = pd.DataFrame(result)
            result['tags'] = result['tags'].map(parse_tags)
            result = pd.concat([result, pd.json_normalize(result['tags'])], axis=1)
            result = result.drop(columns=['tags'])
            result['currency'] = result['price'].str.split(' ').str.get(-1).str.strip().str.replace('.', '')
            result['price'] = result['price'].str.replace(r'\D', '', regex=True).astype('int')
            result['district'] = result['district'].str.split('-').str.get(-1).str.strip()
            result['address'] = None
            result['source'] = 'olx'
            total = pd.concat([total, result])
        except:
            logger.add_log(
                message='Error on preprocessing',
                level='ERROR',
                data={'exception': traceback.format_exc()}
            )
    try:
        load_to_db(total)
    except:
        logger.add_log(
            message='Error on loading to db',
            level='ERROR',
            data={'exception': traceback.format_exc()}
        )


if __name__ == "__main__":
    start = datetime.datetime.now()
    main()
    print(datetime.datetime.now() - start)
