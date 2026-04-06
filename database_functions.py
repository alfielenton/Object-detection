from mysql import connector
from matplotlib import pyplot as plt
from matplotlib import image as mpimg
from matplotlib import colors as mcolors
import random
import numpy as np
import cv2

connect_args = {"host":"localhost", 
                "username":"root", 
                "password":"Art!ch0ke23",
                "database":"objects"}

def get_wildlife_2_animals():

    path = "datasets//wildlife 2//data.yaml"
    animals_start = False
    animals = []

    with open(path, "r") as f:

        for line in f.readlines():

            if line.find("names:") != -1:
                animals_start = True
                continue

            if animals_start:
                line = line[:-1] if line.endswith("\n") else line
                animal = line.split(' ')[-1].lower()
                animals.append(animal)

    return animals

wildlife_2_animals = get_wildlife_2_animals()

def create_table(name, columns):

    query = f"CREATE TABLE {name} (" + ", ".join(columns) + ")"
    exists_query = f"SHOW TABLES LIKE {name}"


    with connector.connect(**connect_args) as db:

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

    query = f"SHOW TABLES"

    with connector.connect(**connect_args) as db:

        cur = db.cursor()
        cur.execute(query)
        info = cur.fetchall()

    return info
 
def get_table_info(name):

    query = f"DESCRIBE {name}"

    with connector.connect(**connect_args) as db:

        cur = db.cursor()
        cur.execute(query)
        info = cur.fetchall()

    if not info:
        print(f"Table {name} doesn't exist")
    else:
        return info
        
def insert_animal(animal_name):

    query = f"INSERT INTO animals (name) VALUES ('{animal_name}')"

    with connector.connect(**connect_args)  as db:

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

    with connector.connect(**connect_args)  as db:

        cur = db.cursor()
        cur.execute(query)
        animals = cur.fetchall()

    return animals

def insert_image(image_path, image_height, image_width, image_depth, folder):

    query = f"INSERT INTO images(path, height, width, depth, folder) VALUES ('{image_path}', {image_height}, {image_width}, {image_depth}, '{folder}')"

    with connector.connect(**connect_args)  as db:

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

    with connector.connect(**connect_args) as db:

        cur = db.cursor()
        cur.execute(query)
        images = cur.fetchall()

    if images:
        return images
    else:   
        print("No recorded images")

def find_image_id(image_path):

    query = f"SELECT id FROM images WHERE path = '{image_path}'"

    with connector.connect(**connect_args) as db:

        cur = db.cursor()
        cur.execute(query)
        id = cur.fetchone()

    if id:
        return id
    else:
        return None

def find_image_path(image_id):

    query = f"SELECT path FROM images WHERE id = {image_id}"

    with connector.connect(**connect_args) as db:

        cur = db.cursor()
        cur.execute(query)
        path = cur.fetchone()
    
    if path:
        return path
    else:
        return None

def show_image(image_id):

    query = f"SELECT path FROM images WHERE id = {image_id}"

    with connector.connect(**connect_args) as db:

        cur = db.cursor()
        cur.execute(query)
        path = cur.fetchone()

    img = mpimg.imread(path[0])
    plt.imshow(img)
    plt.show()

def count_animals_in_image(image_id):

    query = "SELECT COUNT(animal_id) " \
            "FROM instances " \
            f"WHERE image_id = {image_id} " \
            "GROUP BY image_id"
    
    with connector.connect(**connect_args) as db:

        cur = db.cursor()
        cur.execute(query)
        res = cur.fetchone()

    return res

def find_animal_id(animal_name):

    query = f"SELECT id FROM animals WHERE name = '{animal_name}'"     

    with connector.connect(**connect_args)  as db:

        cur = db.cursor()
        cur.execute(query)
        id = cur.fetchone()

    if id:
        return id
    else:
        return None
    
def find_animal_name(animal_id):

    query = f"SELECT name FROM animals WHERE id = {animal_id}"

    with connector.connect(**connect_args)  as db:

        cur = db.cursor()
        cur.execute(query)
        id = cur.fetchone()

    if id:
        return id
    else:
        return None

def insert_instance(image_id, animal_id, x_center, y_center, width, height):

    query = f"INSERT INTO instances VALUES ({image_id}, {animal_id}, {x_center}, {y_center}, {width}, {height})"

    with connector.connect(**connect_args) as db:

        try:
            cur = db.cursor()
            cur.execute(query)
            db.commit()
        except:
            db.rollback()
            print("Insert error: ", query)
            raise Exception
        
    pass

def get_instances(limit = None):

    query = "SELECT * FROM instances"

    if limit is not None:
        query += f" LIMIT {limit}"

    with connector.connect(**connect_args) as db:

        cur = db.cursor()
        cur.execute(query)
        images = cur.fetchall()

    if images:
        return images
    else:   
        print("No recorded objects")   

def show_bbs(image_id, coords = False):

    all_animals = get_all_animals()
    all_animals = [a[1] for a in all_animals]

    colors = mcolors.CSS4_COLORS
    colors = random.sample(sorted(colors), len(all_animals))

    query = f"SELECT path, height, width FROM images WHERE id = {image_id}"

    with connector.connect(**connect_args) as db:

        cur = db.cursor()
        cur.execute(query)
        path, im_height, im_width = cur.fetchone()

    img = mpimg.imread(path)

    query = f"SELECT a.name, i.x_center, i.y_center, i.width, i.height FROM instances i " \
            "JOIN animals a ON i.animal_id = a.id " \
            f" WHERE image_id = {image_id}"

    with connector.connect(**connect_args) as db:

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

        if coords:
            for x, y in zip(x_centers[arr_animal_names==a], y_centers[arr_animal_names==a]):
                plt.text(x, y, f'({x/im_width:.2f},{y/im_height:.2f})')
        plt.vlines(x_mins[arr_animal_names==a], y_mins[arr_animal_names==a], y_maxs[arr_animal_names==a], color='black')
        plt.vlines(x_maxs[arr_animal_names==a], y_mins[arr_animal_names==a], y_maxs[arr_animal_names==a], color='black')

        plt.hlines(y_mins[arr_animal_names==a], x_mins[arr_animal_names==a], x_maxs[arr_animal_names==a], color='black')
        plt.hlines(y_maxs[arr_animal_names==a], x_mins[arr_animal_names==a], x_maxs[arr_animal_names==a], color='black')
    
    animal_names = list(set(animal_names))
    title = "Picture of " + ' and '.join(animal_names)
    plt.imshow(img)
    plt.title(title)
    plt.legend()
    plt.show()

def get_csv(table_name, header, save_path):

    query = f"SELECT * FROM {table_name}"

    with connector.connect(**connect_args) as db:

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

def relabel_image(image_id):

    show_bbs(image_id)
    wish_to_relabel = input(f"Are you sure you want to relabel image {image_id} (Y/N)? -- ")
    if wish_to_relabel.lower() == "n":
        print("Will not relabel image")
        return

    relabel_entirely = input("Do you wish to relabel entirely or to add further labels " \
                                "(1 - relabel entirely, 2 - add further labels)? -- ")
    
    relabel_entirely = int(relabel_entirely) == 1

    if relabel_entirely:
        delete_query = f"DELETE FROM instances WHERE image_id = {image_id}"

        with connector.connect(**connect_args) as db:

            try:
                cur = db.cursor()
                cur.execute(delete_query)
                db.commit()
            except:
                print("Delete query error: ", delete_query)
                db.rollback()
                raise Exception
        
    dim_query = f"SELECT height, width FROM images WHERE id = {image_id}"

    with connector.connect(**connect_args) as db:

        cur = db.cursor()
        cur.execute(dim_query)
        im_height, im_width = cur.fetchone()
    
    im_path = find_image_path(image_id)[0]
    lab_path = im_path.replace("images", "labels")[:-3] + 'txt'

    labelling_complete = False
    label_writing_mode = "w" if relabel_entirely else "a"

    while not labelling_complete:

        animal_identified = False

        print("\n\nBeginning new label: \n")
        print("\tStart by identifying the type of animal you wish to label")
        while not animal_identified:

            show_bbs(image_id)
            animal = input("\tWhat animal are you labelling? -- ").lower()
            if find_animal_id(animal):
                animal_identified = True
                animal_id = find_animal_id(animal)[0]
            else:
                print("Animal not found in database")

        print(f"\n\tFind the highest point of the {animal}")
        show_bbs(image_id)
        top_point = input(f"\tWhat value is the highest point of the {animal}? -- ")
        top_point = int(top_point)

        print(f"\n\tFind the eastest points of the {animal}")
        show_bbs(image_id)
        east_point = input(f"\tWhat value is the eastest point of the {animal}? -- ")
        east_point = int(east_point)

        print(f"\n\tFind the lowest point of the {animal}")
        show_bbs(image_id)
        bottom_point = input(f"\tWhat value is the lowest point of the {animal}? -- ")
        bottom_point = int(bottom_point)

        print(f"\n\tFind the westest point of the {animal}")
        show_bbs(image_id)
        west_point = input(f"\tWhat value is the westest point of the {animal}? -- ")
        west_point = int(west_point)
        
        width = round((east_point - west_point) / im_width, 8)
        height = round((bottom_point - top_point) / im_height, 8)
        x_center = round(.5 * (east_point + west_point) / im_width, 8)
        y_center = round(.5 * (top_point + bottom_point) / im_height, 8)

        insert_instance(image_id, animal_id, x_center, y_center, width, height)
        show_bbs(image_id)
        y_n = input("\n\tAre you satisfied with this labelling (Y/N)? -- ")
        satisfied = y_n.lower() == "y"

        if satisfied:

            with open(lab_path, label_writing_mode) as f:
                f.write(f"{wildlife_2_animals.index(animal)} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")
            label_writing_mode = "a"
        else:
            print("\n\tDeleting this instance from database.")

            delete_query = "DELETE FROM instances WHERE " \
                            f"image_id = {image_id} AND " \
                            f"animal_id = {animal_id} AND " \
                            f"x_center = {x_center} AND " \
                            f"y_center = {y_center} AND " \
                            f"width = {width} AND " \
                            f"height = {height}"
            
            with connector.connect(**connect_args) as db:

                try:
                    cur = db.cursor()
                    cur.execute(delete_query)
                    db.commit()
                except:
                    print("Delete query error: ", delete_query)
                    db.rollback()
                    raise Exception
                
        if satisfied:
            y_n = input("\n\tHave you finished labelling all animals in the image (Y/N)? -- ")
            labelling_complete = y_n.lower() == "y"
        
    if labelling_complete:
        print("Labelling complete")
        show_bbs(image_id)

#########################################
##FUNCTIONS FOR COUNTING OBJECTS MODELS##
#########################################

def select_data(N):

    more_than_one_query = "SELECT im.id, COUNT(ins.animal_id) " \
                          "FROM instances ins " \
                          "JOIN images im " \
                          "ON im.id = ins.image_id " \
                          "WHERE im.depth = 3 " \
                          "GROUP BY ins.image_id " \
                          "HAVING COUNT(ins.animal_id) > 1"
    
    one_query = "SELECT im.id, COUNT(ins.animal_id) " \
                "FROM instances ins " \
                "JOIN images im " \
                "ON im.id = ins.image_id " \
                "WHERE im.depth = 3 " \
                "GROUP BY ins.image_id " \
                "HAVING COUNT(ins.animal_id) = 1"
    
    with connector.connect(**connect_args) as db:

        cur = db.cursor()

        cur.execute(more_than_one_query)
        more_than_one_images = cur.fetchall()

        cur.execute(one_query)
        one_images = cur.fetchall()

    all_images = more_than_one_images + random.sample(one_images, N - len(more_than_one_images))
    return all_images

def make_model_data(train_prop, valid_prop = None, N = 5000):

    if valid_prop is not None:
        assert(train_prop + valid_prop < 1.)
        train_size = int(N * train_prop)
        valid_size = int(N * valid_prop)
    else:
        assert(train_prop < 1.)
        train_size = int(N * train_prop)

    sampled_data = select_data(N)
    train_data = random.sample(sampled_data, train_size)
    reduced_sample = [d for d in sampled_data if d not in train_data]

    if valid_prop is not None:
        valid_data = random.sample(reduced_sample, valid_size)
        test_data = [d for d in reduced_sample if d not in valid_data]
        return train_data, valid_data, test_data
    else:
        test_data = reduced_sample
        return train_data, test_data
    
def build_image_batch(images):

    batch = []
    for im in images:

        path = find_image_path(im)[0]
        img = mpimg.imread(path)

        img = cv2.resize(img, (970, 750))
        img = img.transpose(2, 0, 1) / 255.
        batch.append(img)

    batch = np.stack(batch, axis=0)
    return batch
    