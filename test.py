import sqlite3
import sys
import json

def get_conf():
    con_file = ''
    print(sys.argv[1])
    with open('indeed_config_ai.json', 'r', encoding='utf-8') as f:
        con_file = json.loads(f.read())
    if sys.argv[1] == 'ai':
        con_file = con_file['CONFIGURE_AI']

        # print(con_file)
    return con_file

print(get_conf()['AGE'])
