from functools import wraps
import json
from os import environ as env
from typing import Dict
from flask_swagger_ui import get_swaggerui_blueprint
import numpy as np
import requests
from six.moves.urllib.request import urlopen

from dotenv import load_dotenv, find_dotenv
from flask import Flask, request, jsonify, _request_ctx_stack, Response, make_response
from flask_cors import cross_origin
from jose import jwt
from sqlalchemy import create_engine
import pandas as pd
import pickle
from predictor import Predictor

AUTH0_DOMAIN = 'dev-lzz2ukzxdjuncnzl.us.auth0.com'
API_IDENTIFIER = 'https://dev-lzz2ukzxdjuncnzl.us.auth0.com/api/v2/'
CLIENT_ID = 'qpY3dU4mWc2DlBvBM8GiJkiStRTaIVDo',
CLIENT_SECRET = 'LzKTx5lfG75OVIA2JLbry8Ig7YzfitxLwuMrqfQNBxGiYXNnOYCWbSwFZiPZw62G'
ALGORITHMS = ["RS256"]
DATABASE_URL = f"postgresql://postgres:admin@localhost:5433/postgres"
PREDICTOR = Predictor()
with open('districts_vectors.pickle', 'rb') as f:
    DISTRICT_VECTORS = pickle.load(f)
with open('word2vec_model.pickle', 'rb') as f:
    WORD2VEC_MODEL = pickle.load(f)
APP = Flask(__name__)


# Format error response and append status code.
class AuthError(Exception):
    """
    An AuthError is raised whenever the authentication failed.
    """

    def __init__(self, error: Dict[str, str], status_code: int):
        super().__init__()
        self.error = error
        self.status_code = status_code


@APP.errorhandler(AuthError)
def handle_auth_error(ex: AuthError) -> Response:
    """
    serializes the given AuthError as json and sets the response status code accordingly.
    :param ex: an auth error
    :return: json serialized ex response
    """
    response = jsonify(ex.error)
    response.status_code = ex.status_code
    return response


def get_app_access_token():
    # get access token
    data = {
        'audience': API_IDENTIFIER,
        'grant_type': 'client_credentials',
        'client_id': CLIENT_ID,
        'client_secret': CLIENT_SECRET,
    }
    headers = {'content-type': 'application/x-www-form-urlencoded'}
    response = requests.post(f'https://{AUTH0_DOMAIN}/oauth/token', data=data, headers=headers)
    token_type = response.json()['token_type']
    access_token = response.json()['access_token']
    return token_type + ' ' + access_token


def get_search_params(inp_request):
    params = {}
    for param in ['query', 'max_price', 'min_price', 'min_rooms', 'max_rooms', 'min_floor', 'max_floor', 'district',
                  'min_total_area', 'max_total_area', 'min_kitchen_area', 'max_kitchen_area', 'animals', 'source',
                  'min_price_diff', 'max_price_diff']:
        if param in inp_request:
            params[param] = inp_request.get(param)
    query = []
    for param in params:
        if 'min_' in param:
            query.append(f"{param.split('_', 1)[1]} >= {params[param]}")
            continue
        if 'max_' in param:
            query.append(f"{param.split('_', 1)[1]} <= {params[param]}")
            continue
        if param == 'query':
            query.append(f"LOWER(description) LIKE '%%{params[param].lower()}%%'")
            continue
        if param in ['district', 'source']:
            query.append(f"""{param} = '{params[param].replace("'", "''")}'""")
            continue
        if param == 'animals':
            query.append(f"{param} = {params[param]}")
            continue
        query.append(f"{param.split('_', 1)[1]} = {params[param]}")
    return query


def get_token_auth_header() -> str:
    """Obtains the access token from the Authorization Header
    """
    auth = request.headers.get("Authorization", None)
    if not auth:
        raise AuthError({"code": "authorization_header_missing",
                         "description":
                             "Authorization header is expected"}, 401)

    parts = auth.split()

    if parts[0].lower() != "bearer":
        raise AuthError({"code": "invalid_header",
                         "description":
                             "Authorization header must start with"
                             " Bearer"}, 401)
    if len(parts) == 1:
        raise AuthError({"code": "invalid_header",
                         "description": "Token not found"}, 401)
    if len(parts) > 2:
        raise AuthError({"code": "invalid_header",
                         "description":
                             "Authorization header must be"
                             " Bearer token"}, 401)

    token = parts[1]
    return token


def requires_scope(required_scope: str) -> bool:
    """Determines if the required scope is present in the access token
    Args:
        required_scope (str): The scope required to access the resource
    """
    token = get_token_auth_header()
    unverified_claims = jwt.get_unverified_claims(token)
    if unverified_claims.get("scope"):
        token_scopes = unverified_claims["scope"].split()
        for token_scope in token_scopes:
            if token_scope == required_scope:
                return True
    return False


def requires_auth(func):
    """Determines if the access token is valid
    """

    @wraps(func)
    def decorated(*args, **kwargs):
        token = get_token_auth_header()
        jsonurl = urlopen("https://" + AUTH0_DOMAIN + "/.well-known/jwks.json")
        jwks = json.loads(jsonurl.read())
        try:
            unverified_header = jwt.get_unverified_header(token)
        except jwt.JWTError as jwt_error:
            raise AuthError({"code": "invalid_header",
                             "description":
                                 "Invalid header. "
                                 "Use an RS256 signed JWT Access Token"}, 401) from jwt_error
        if unverified_header["alg"] == "HS256":
            raise AuthError({"code": "invalid_header",
                             "description":
                                 "Invalid header. "
                                 "Use an RS256 signed JWT Access Token"}, 401)
        rsa_key = {}
        for key in jwks["keys"]:
            if key["kid"] == unverified_header["kid"]:
                rsa_key = {
                    "kty": key["kty"],
                    "kid": key["kid"],
                    "use": key["use"],
                    "n": key["n"],
                    "e": key["e"]
                }
        if rsa_key:
            try:
                payload = jwt.decode(
                    token,
                    rsa_key,
                    algorithms=ALGORITHMS,
                    audience=API_IDENTIFIER,
                    issuer="https://" + AUTH0_DOMAIN + "/"
                )
            except jwt.ExpiredSignatureError as expired_sign_error:
                raise AuthError({"code": "token_expired",
                                 "description": "token is expired"}, 401) from expired_sign_error
            except jwt.JWTClaimsError as jwt_claims_error:
                raise AuthError({"code": "invalid_claims",
                                 "description":
                                     "incorrect claims,"
                                     " please check the audience and issuer"}, 401) from jwt_claims_error
            except Exception as exc:
                raise AuthError({"code": "invalid_header",
                                 "description":
                                     "Unable to parse authentication"
                                     " token."}, 401) from exc

            _request_ctx_stack.top.current_user = payload
            return func(*args, **kwargs)
        raise AuthError({"code": "invalid_header",
                         "description": "Unable to find appropriate key"}, 401)

    return decorated


@APP.route('/api/register', methods=['POST'])
@cross_origin(headers=["Content-Type", "Authorization"])
def register():
    try:
        # get input values
        request_data = json.loads(request.data.decode(), strict=False)
        email = request_data.get('email')
        password = request_data.get('password')
        if email is None or password is None:
            return make_response("{'error': 'Invalid email or password'}", 400)

        # register new user
        data = json.dumps({
            "email": email,
            "connection": "Username-Password-Authentication",
            "password": password
        })
        headers = {
            'Content-Type': 'application/json',
            'Accept': 'application/json',
            'Authorization': get_app_access_token()
        }
        response = requests.post(f'https://{AUTH0_DOMAIN}/api/v2/users', headers=headers, data=data)

        return make_response(json.dumps(response.json()), response.status_code)
    except Exception as e:
        return make_response(json.dumps({'error': str(e)}), 500)


@APP.route('/api/token', methods=['POST'])
@cross_origin(headers=["Content-Type", "Authorization"])
def login():
    try:
        # get input values
        request_data = json.loads(request.data.decode(), strict=False)
        email = request_data.get('email')
        password = request_data.get('password')
        if email is None or password is None:
            return make_response("{'error': 'Invalid email or password'}", 400)

        # get access token
        headers = {
            'content-type': 'application/x-www-form-urlencoded'}

        data = {
            'grant_type': 'password',
            'username': email,
            'password': password,
            'audience': API_IDENTIFIER,
            'client_id': CLIENT_ID,
            'client_secret': CLIENT_SECRET,
            'scope': 'offline_access'
        }
        response = requests.post(f'https://{AUTH0_DOMAIN}/oauth/token', headers=headers, data=data)

        return make_response(json.dumps(response.json()), response.status_code)
    except Exception as e:
        return make_response(json.dumps({'error': str(e)}), 500)


@APP.route('/api/search', methods=['GET'])
@cross_origin(headers=["Content-Type", "Authorization"])
@cross_origin(headers=["Access-Control-Allow-Origin", "http://localhost:3000"])
@requires_auth
def search():
    try:
        query = get_search_params(request.args)
        select = """SELECT * FROM actual_ads"""
        if query:
            select += ' WHERE ' + ' AND '.join(query)
        try:
            df = pd.read_sql(select, create_engine(DATABASE_URL))
        except Exception as e:
            return make_response(str({'error': 'Invalid params ' + str(e)}), 400)

        return make_response(df.set_index('link').to_json(orient="index", force_ascii=False), 200)

    except Exception as e:
        return make_response(json.dumps({'error': str(e)}), 500)


@APP.route('/api/stats', methods=['GET'])
@cross_origin(headers=["Content-Type", "Authorization"])
@cross_origin(headers=["Access-Control-Allow-Origin", "http://localhost:3000"])
@requires_auth
def get_stats():
    try:
        query = get_search_params(request.args)
        group_by = request.args.get('group_by')
        group_by_dict = {'year': 'EXTRACT(year from updated_at)',
                         'month': 'EXTRACT(year from updated_at), EXTRACT(month from updated_at)',
                         'week': 'EXTRACT(year from updated_at), EXTRACT(week from updated_at)'}
        if group_by not in group_by_dict:
            return make_response("{'error': 'Invalid groupby param'}", 400)

        metric = request.args.get('metric')
        if metric not in ['avg', 'median', 'min', 'max']:
            return make_response("{'error': 'Invalid metric param'}", 400)

        select = f"""SELECT {group_by_dict[group_by]}, {metric}(CASE WHEN currency = 'грн' THEN price ELSE price * 40 END) FROM ads"""
        group_by_limit = {'year': 'EXTRACT(year from NOW()-created_at) <= 2',
                          'month': 'EXTRACT(month from NOW()-created_at) <= 5',
                          'week': 'EXTRACT(day from NOW()-created_at) <= 60'}
        select += ' WHERE ' + group_by_limit[group_by]
        if query:
            select += ' AND ' + ' AND '.join(query)
        select += f' GROUP BY {group_by_dict[group_by]}'
        try:
            df = pd.read_sql(select, create_engine(DATABASE_URL))
        except Exception as e:
            return make_response(str({'error': 'Invalid params ' + str(e)}), 400)

        group_by_columns = {'year': ['year'],
                            'month': ['year', 'month'],
                            'week': ['year', 'week']}

        df.columns = group_by_columns[group_by] + [metric]
        return make_response(df.set_index(group_by_columns[group_by]).to_json(orient="index", force_ascii=False), 200)

    except Exception as e:
        return make_response(json.dumps({'error': str(e)}), 500)


@APP.route('/api/custom/evaluate', methods=['GET'])
@cross_origin(headers=["Content-Type", "Authorization"])
@cross_origin(headers=["Access-Control-Allow-Origin", "http://localhost:3000"])
@requires_auth
def evaluate_custom():
    try:
        predictors = ['rooms', 'floor', 'max_floor', 'total_area', 'kitchen_area', 'animals', 'district', 'description']
        df = {}
        try:
            for predictor in predictors:
                df[predictor] = [request.args.get(predictor)]
                if predictor == 'animals':
                    df[predictor] = [bool(df[predictor][0])]
                elif predictor not in ['district', 'description']:
                    df[predictor] = [float(df[predictor][0])]
        except Exception as e:
            return make_response(str({'error': 'Invalid param ' + str(e)}), 400)
        df = pd.DataFrame(df)
        try:
            prediction, _ = PREDICTOR.predict(df)
        except Exception as e:
            return make_response(str({'error': 'Prediction error ' + str(e)}), 400)
        return make_response(str({'prediction': round(prediction[0], 2)}), 200)
    except Exception as e:
        return make_response(json.dumps({'error': str(e)}), 500)


@APP.route('/api/subscribe', methods=['POST'])
@cross_origin(headers=["Content-Type", "Authorization"])
@cross_origin(headers=["Access-Control-Allow-Origin", "http://localhost:3000"])
@requires_auth
def subscribe():
    try:
        request_data = json.loads(request.data.decode(), strict=False)
        email = request_data.get('email')
        if email is None:
            return make_response(f"{'error': 'Invalid email param'}", 400)
        query = get_search_params(request_data)
        select = """SELECT * FROM actual_ads WHERE EXTRACT(day from NOW()-created_at) = 0 """
        if query:
            select += ' AND ' + ' AND '.join(query)
        try:
            pd.read_sql(select, create_engine(DATABASE_URL))
        except Exception as e:
            return make_response(str({'error': 'Invalid params ' + str(e)}), 400)

        df = pd.DataFrame({'email': [email], 'search': [select]})
        df.set_index('email').to_sql('subscriptions', create_engine(DATABASE_URL), if_exists='append')

        return make_response(str({'status': 'success'}), 200)

    except Exception as e:
        return make_response(json.dumps({'error': str(e)}), 500)


@APP.route('/api/similar', methods=['GET'])
@cross_origin(headers=["Content-Type", "Authorization"])
@cross_origin(headers=["Access-Control-Allow-Origin", "http://localhost:3000"])
@requires_auth
def similar():
    try:
        link = request.args.get('link')
        select = f"""SELECT COUNT(*) FROM actual_ads WHERE link = '{link}'"""
        try:
            df = pd.read_sql(select, create_engine(DATABASE_URL))
            if df.iat[0, 0] == 0:
                return make_response(f"{'error': 'Link is not in actual data'}", 400)
        except Exception as e:
            return make_response(str({'error': 'Invalid params ' + str(e)}), 400)

        predictors = ['rooms',
                      'floor',
                      'building_floors',
                      'total_area',
                      'kitchen_area',
                      'price'
                      ]
        predictors = [f"(((al.{i} - (SELECT avg({i}) FROM actual_ads))/(SELECT stddev({i}) FROM actual_ads)) " \
                      f"- ((target.{i} - (SELECT avg({i}) FROM actual_ads))/(SELECT stddev({i}) FROM actual_ads))) * 2"
                      for i in predictors]
        pred_vectors = ['description_' + str(i) for i in range(13)] + ['district_' + str(i) for i in range(4)]
        predictors += [f"(((al.{i} - (SELECT avg({i}) FROM vectors))/(SELECT stddev({i}) FROM vectors)) " \
                       f"- ((target.{i} - (SELECT avg({i}) FROM vectors))/(SELECT stddev({i}) FROM vectors)))"
                       for i in pred_vectors]

        select = f"""SELECT al.link FROM ((SELECT * FROM actual_ads) a INNER JOIN (SELECT * FROM vectors) v USING(link)) as al
cross join((SELECT * FROM actual_ads WHERE link = '{link}') a_1 INNER JOIN (SELECT * FROM vectors WHERE link = '{link}') v_1 USING(link)) as target ORDER BY abs(""" \
                 + ' + '.join(predictors) + ')  LIMIT 11'
        select = f"SELECT * FROM actual_ads WHERE link IN ({select}) AND link <> '{link}'"
        try:
            df = pd.read_sql(select, create_engine(DATABASE_URL))
        except Exception as e:
            return make_response(str({'error': 'Invalid params ' + str(e)}), 400)

        return make_response(df.set_index('link').to_json(orient="index", force_ascii=False), 200)

    except Exception as e:
        return make_response(json.dumps({'error': str(e)}), 500)


@APP.route('/api/custom/similar', methods=['GET'])
@cross_origin(headers=["Content-Type", "Authorization"])
@cross_origin(headers=["Access-Control-Allow-Origin", "http://localhost:3000"])
@requires_auth
def custom_similar():
    try:
        predictors = ['rooms',
                      'floor',
                      'building_floors',
                      'total_area',
                      'kitchen_area',
                      'price'
                      ]
        predictors = [f"(((al.{i} - (SELECT avg({i}) FROM actual_ads))/(SELECT stddev({i}) FROM actual_ads)) " \
                      f"- (({request.args.get(i)} - (SELECT avg({i}) FROM actual_ads))/(SELECT stddev({i}) FROM actual_ads))) * 2"
                      for i in predictors if i in request.args]

        description = request.args.get('description')
        if description is not None:
            try:
                description = description.replace('.', '').replace(',', '').replace('\n', ' ')
                description = PREDICTOR.preprocess_description([description])[0]
                description_vector = np.mean(
                    [WORD2VEC_MODEL.wv[word] for word in description if word in WORD2VEC_MODEL.wv], axis=0)
                predictors += [
                    f"(((al.{'description_' + str(i)} - (SELECT avg({'description_' + str(i)}) FROM vectors))/(SELECT stddev({'description_' + str(i)}) FROM vectors)) " \
                    f"- (({description_vector[i]} - (SELECT avg({'description_' + str(i)}) FROM vectors))/(SELECT stddev({'description_' + str(i)}) FROM vectors)))"
                    for i in range(13)]
            except:
                return make_response(str({'error': 'Invalid description param'}), 400)

        district = request.args.get('district')
        if district is not None:
            try:
                district_vector = DISTRICT_VECTORS[district]
            except:
                return make_response(str({'error': 'Invalid district param'}), 400)
            predictors += [
                f"(((al.{'district_' + str(i)} - (SELECT avg({'district_' + str(i)}) FROM vectors))/(SELECT stddev({'district_' + str(i)}) FROM vectors)) " \
                f"- (({district_vector[i]} - (SELECT avg({'district_' + str(i)}) FROM vectors))/(SELECT stddev({'district_' + str(i)}) FROM vectors)))"
                for i in range(4)]

        if len(predictors) == 0:
            return make_response(str({'error': 'Invalid params'}), 400)

        select = f"""SELECT al.link FROM ((SELECT * FROM actual_ads) a INNER JOIN (SELECT * FROM vectors) v USING(link)) as al ORDER BY 
        abs(""" + ' + '.join(predictors) + ')  LIMIT 10'
        select = f"SELECT * FROM actual_ads WHERE link IN ({select})"
        try:
            df = pd.read_sql(select, create_engine(DATABASE_URL))
        except Exception as e:
            return make_response(str({'error': 'Invalid params ' + str(e)}), 400)

        return make_response(df.set_index('link').to_json(orient="index", force_ascii=False), 200)

    except Exception as e:
        return make_response(json.dumps({'error': str(e)}), 500)



if __name__ == "__main__":
    SWAGGER_URL = "/swagger"
    API_URL = "/static/swagger.json"

    swagger_ui_blueprint = get_swaggerui_blueprint(
        SWAGGER_URL,
        API_URL,
        config={
            'app_name': 'Rent API'
        }
    )
    APP.register_blueprint(swagger_ui_blueprint, url_prefix=SWAGGER_URL)
    APP.run(host="0.0.0.0", port=env.get("PORT", 3010))
