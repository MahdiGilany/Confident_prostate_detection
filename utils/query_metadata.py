import pymysql

CONFIG = {
    'host': '137.82.56.208',
    'user': 'samareh',
    'password': 'samareh',
    'database': "prostate"
}


def query(cur, target: str, location: str, patient_id: int, indicator: str = 'ID'):
    sql_select_query = f"SELECT {target} FROM {location} WHERE {indicator}={patient_id}"
    cur.execute(sql_select_query)
    return cur.fetchall()


def query_patient_info(patient_id, cursor=None, fields=None):
    close_cursor = False
    if cursor is None:
        cursor = open_connection()
        close_cursor = True

    core_metadata = {}
    if fields is None:
        fields = ['CalculatedInvolvement', 'Involvement', 'Revert', 'id', 'TrueLabel', 'GleasonScore',
                  'CoreName', 'CoreId', 'PrimarySecondary']
    elif not isinstance(fields, list):
        fields = [fields, ]

    for k in fields:
        core_metadata[k] = query(cursor, k, 'core', patient_id, 'PatientID')

    if close_cursor:
        close_connection(cursor)

    return core_metadata


def open_connection():
    cnx = pymysql.connect(**CONFIG)
    return cnx.cursor()


def close_connection(cursor):
    cursor.close()
