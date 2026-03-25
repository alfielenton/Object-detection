import sqlite3 as sql

def create_table(name, columns):

    query = f"CREATE TABLE {name} (" + ", ".join(columns) + ")"
    exists_query = f"SELECT * FROM sqlite_schema WHERE type = 'table' AND name = '{name}'"

    with sql.connect("animals.db") as db:

        cur = db.cursor()
        cur.execute(exists_query)
        exists = cur.fetchone()

        if exists:
            print(f"Table {name} already exists")
        else:
            try:
                cur.execute(query)
                db.commit()
                print("Table created")
            except:
                db.rollback()
                print(query)
                print("Could not create table")
                raise Exception

def get_schema_info():

    query = f"SELECT * FROM sqlite_schema"

    with sql.connect("animals.db") as db:

        cur = db.cursor()
        cur.execute(query)
        info = cur.fetchall()

    return info
 
def get_table_info(name):

    query = f"SELECT * FROM sqlite_schema WHERE name = '{name}'"

    with sql.connect("animals.db") as db:

        cur = db.cursor()
        cur.execute(query)
        info = cur.fetchone()

        if not info:
            print(f"Table {name} doesn't exist")
        else:
            return info