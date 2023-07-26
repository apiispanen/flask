import mysql.connector
from mysql.connector import Error
import codecs

def run_sql_script(filename, cursor):
    # Open and read the file as a single buffer
    with codecs.open(filename, 'r', encoding='utf-8') as fd:

        sqlFile = fd.read()

    # all SQL commands (split on ';')
    sqlCommands = sqlFile.split(';')

    # Execute every command from the input file
    for command in sqlCommands:
        try:
            if command.strip() != '':
                cursor.execute(command)
        except IOError as msg:
            print("Command skipped: ", msg)
        except Exception as e:
            print("Command skipped: ", e)


# try:
#     cnx = mysql.connector.connect(
#         host='panel.hottievideo.com',
#         user='svcpanel',
#         password='~Klisch#!13',
#         database='spinnrdev'
#     )

#     if cnx.is_connected():
#         print('Successfully connected to the database')
#     else:
#         print('Failed to connect to the database')

# except Error as e:
#     print(f'Error: {e}')
# If the connection fails, connect to a local MySQL server

try:
    cnx_local = mysql.connector.connect(
    host='panel.hottievideo.com',
    user='svcpanel',
    password='~Klisch#!13',
    database='spinnrdev',
    charset='utf8'  # Specify the appropriate character encoding
    )
    cursor_local = cnx_local.cursor()

except:
    cnx_local = mysql.connector.connect(
        host='localhost',
        user='root',
        database='spinnr',
            charset='utf8'  # Specify the appropriate character encoding
    )
    cursor_local = cnx_local.cursor()


def get_interests(cursor_local=cursor_local):
    sql = "SELECT name FROM `interest` ORDER BY `interest`.`name` ASC"
    cursor_local.execute(sql)
    results = cursor_local.fetchall()
    return [row[0] for row in results]

# print(get_interests())
# for row in results:
#     print(row)

# run_sql_script('backup.sql', cursor_local)

# finally:
#     if cnx.is_connected():
#         cnx.close()
#     if 'cnx_local' in locals() and cnx_local.is_connected():
#         cnx_local.close()
# cursor_local.close()
