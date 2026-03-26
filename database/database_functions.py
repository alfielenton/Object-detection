import sqlite3 as sql
from matplotlib import pyplot as plt
from matplotlib import image as mpimg
from matplotlib import colors as mcolors
import random
import numpy as np

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

def insert_image(image_path, image_height, image_width, image_depth, folder):

    query = f"INSERT INTO images(image_path, image_height, image_width, image_depth, folder) VALUES ('{image_path}', {image_height}, {image_width}, {image_depth}, '{folder}')"

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

def get_image_and_bbs(image_id):

    all_animals = get_all_animals()
    all_animals = [a[1] for a in all_animals]

    colors = mcolors.CSS4_COLORS
    colors = random.sample(sorted(colors), len(all_animals))

    query = f"SELECT image_path, image_height, image_width FROM images WHERE image_id = {image_id}"

    with sql.connect("object-detection.db") as db:

        cur = db.cursor()
        cur.execute(query)
        path, im_height, im_width = cur.fetchone()

    img = mpimg.imread(path)

    query = f"SELECT a.animal_name, o.x_center, o.y_center, o.width, o.height FROM objects o " \
            "JOIN animals a ON o.animal_id = a.animal_id " \
            f" WHERE image_id = {image_id}"

    with sql.connect("object-detection.db") as db:

        cur = db.cursor()
        cur.execute(query)
        bbs = cur.fetchall()

    animal_names = []
    x_centers = []
    y_centers = []
    widths = []
    heights = []

    for animal_name, x_center, y_center, width, height in bbs:
        animal_names.append(animal_name)
        x_centers.append(im_width * x_center)
        y_centers.append(im_height * y_center)
        widths.append(width * im_width)
        heights.append(height * im_height)

    x_centers = np.array(x_centers)
    y_centers = np.array(y_centers)

    y_mins = np.array(y_centers) - np.array(heights) / 2
    y_maxs = np.array(y_centers) + np.array(heights) / 2

    x_mins = np.array(x_centers) - np.array(widths) / 2
    x_maxs = np.array(x_centers) + np.array(widths) / 2

    arr_animal_names = np.array(animal_names)
    for a in np.unique(arr_animal_names):

        plt.scatter(x_centers[arr_animal_names==a], y_centers[arr_animal_names==a], color = colors[all_animals.index(str(a))], marker='+', s=5, label = a)
        plt.vlines(x_mins[arr_animal_names==a], y_mins[arr_animal_names==a], y_maxs[arr_animal_names==a], color='black')
        plt.vlines(x_maxs[arr_animal_names==a], y_mins[arr_animal_names==a], y_maxs[arr_animal_names==a], color='black')

        plt.hlines(y_mins[arr_animal_names==a], x_mins[arr_animal_names==a], x_maxs[arr_animal_names==a], color='black')
        plt.hlines(y_maxs[arr_animal_names==a], x_mins[arr_animal_names==a], x_maxs[arr_animal_names==a], color='black')
    
    animal_names = list(set(animal_names))
    title = "Pictre of " + ' and '.join(animal_names)
    plt.imshow(img)
    plt.title(title)
    plt.legend()

def get_csv(table_name, header, save_path):

    query = f"SELECT * FROM {table_name}"

    with sql.connect("object-detection.db") as db:

        cur = db.cursor()
        cur.execute(query)
        lines = cur.fetchall()

    with open(save_path, "w") as f:

        file = header + "\n"
        for line in lines:
            line = [str(l) for l in line]
            file += ', '.join(line)
            if line != lines[-1]:
                file += "\n"

        f.write(file)
    
    pass