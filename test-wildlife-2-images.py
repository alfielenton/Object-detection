from mysql import connector
from database_functions import *
import json

connect_args = {"host":"localhost", 
                "username":"root", 
                "password":"Art!ch0ke23",
                "database":"objects"}

query = "SELECT ins.image_id, COUNT(ins.image_id) AS num_instances " \
        "FROM instances ins " \
        "JOIN images im " \
        "ON ins.image_id = im.id " \
        "AND im.folder = 'wildlife 2' " \
        "JOIN animals a " \
        "ON a.id = ins.animal_id " \
        "AND a.name <> 'butterfly' " \
        "AND a.name <> 'duck' " \
        "GROUP BY ins.image_id " \
        "HAVING num_instances > 1 " \
        "ORDER BY num_instances DESC"

with connector.connect(**connect_args) as db:

    cur = db.cursor()
    cur.execute(query)
    ims_and_ins = cur.fetchall()

end_point = (15615, 2)
ind = ims_and_ins.index(end_point)

n_images_to_do = len(ims_and_ins[:ind])
print("Images to go through: ", n_images_to_do)

for i, im_and_in in enumerate(ims_and_ins[:ind]):

    im_id, num_ins = im_and_in
    print(f"Image {im_id}. {i} out of {n_images_to_do} done")
    print(f"Number of instances: {num_ins}")
    show_bbs(im_id)

    msg = input("Problem with image ")
    print("\n")

    if msg:
        with open("problem-images.txt", "a") as f:
            line = f"{im_id} -- {msg}\n"
            f.write(line)
    

