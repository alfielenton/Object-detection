import sqlite3 as sql

db_name = "object-detection.db"

def create_table(name, columns):

    query = f"CREATE TABLE {name} (" + ", ".join(columns) + ")"
    exists_query = f"SELECT * FROM sqlite_schema WHERE type = 'table' AND name = '{name}'"

    with sql.connect(db_name) as db:

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
            
    pass

def get_schema_info():

    query = f"SELECT * FROM sqlite_schema"

    with sql.connect(db_name) as db:

        cur = db.cursor()
        cur.execute(query)
        info = cur.fetchall()

    return info
 
def get_table_info(name):

    query = f"SELECT * FROM sqlite_schema WHERE name = '{name}'"

    with sql.connect(db_name) as db:

        cur = db.cursor()
        cur.execute(query)
        info = cur.fetchone()

    if not info:
        print(f"Table {name} doesn't exist")
    else:
        return info
        
def insert_animal(animal_name):

    query = f"INSERT INTO animals (animal_name) VALUES ('{animal_name}')"

    with sql.connect(db_name) as db:

        try:
            cur = db.cursor()
            cur.execute(query)
            db.commit()
            
        except:
            db.rollback()
            print("Insert error: ", query)
            raise Exception
        
    pass
        
def get_all_animals():

    query = f"SELECT * FROM animals"

    with sql.connect(db_name) as db:

        cur = db.cursor()
        cur.execute(query)
        animals = cur.fetchall()

    return animals

def insert_image(image_path):

    query = f"INSERT INTO images(image_path) VALUES ('{image_path}')"

    with sql.connect(db_name) as db:

        try:
            cur = db.cursor()
            cur.execute(query)
            db.commit()
        except:
            db.rollback()
            print("Insert error: ", query)
            raise Exception
        
    pass

def get_images(limit = None):

    query = "SELECT * FROM images"

    if limit is not None:
        query += f" LIMIT {limit}"

    with sql.connect("object-detection.db") as db:

        cur = db.cursor()
        cur.execute(query)
        images = cur.fetchall()

    if images:
        return images
    else:   
        print("No recorded images")

def find_image_id(image_path):

    query = f"SELECT image_id FROM images WHERE image_path = '{image_path}'"

    with sql.connect("object-detection.db") as db:

        cur = db.cursor()
        cur.execute(query)
        id = cur.fetchone()

    if id:
        return id
    else:
        print("No image found with this path")
        return None

def find_animal_id(animal_name):

    query = f"SELECT animal_id FROM animals WHERE animal_name = '{animal_name}'"     

    with sql.connect("object-detection.db") as db:

        cur = db.cursor()
        cur.execute(query)
        id = cur.fetchone()

    if id:
        return id
    else:
        print("No animal found with this path")
        return None

def insert_object(image_id, animal_id, x_center, y_center, width, height):

    query = f"INSERT INTO objects VALUES ({image_id}, {animal_id}, {x_center}, {y_center}, {width}, {height})"

    with sql.connect("object-detection.db") as db:

        try:
            cur = db.cursor()
            cur.execute(query)
            db.commit()
        except:
            db.rollback()
            print("Insert error: ", query)
            raise Exception
        
    pass

def get_objects(limit = None):

    query = "SELECT * FROM objects"

    if limit is not None:
        query += f" LIMIT {limit}"

    with sql.connect("object-detection.db") as db:

        cur = db.cursor()
        cur.execute(query)
        images = cur.fetchall()

    if images:
        return images
    else:   
        print("No recorded objects")   