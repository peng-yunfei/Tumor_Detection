import pandas as pd
import shutil
import os

image_directory = '/Users/fightfei/Desktop/Northeastern/2024Spring/INFO6205 Algo/Labs_6205/GroupProject/Data/Brain Tumor'
output_directory = '/Users/fightfei/Desktop/Northeastern/2024Spring/INFO6205 Algo/Labs_6205/GroupProject/brainTumorDetection/augmented data'
csv_file_path = '/Users/fightfei/Desktop/Northeastern/2024Spring/INFO6205 Algo/Labs_6205/GroupProject/Data/brain_tumor.csv'

class_0_dir = os.path.join(output_directory, 'No')
class_1_dir = os.path.join(output_directory, 'Yes')

if not os.path.exists(class_0_dir):
    os.makedirs(class_0_dir)

if not os.path.exists(class_1_dir):
    os.makedirs(class_1_dir)

df = pd.read_csv(csv_file_path)
# number of 1: 1683
# number of 0: 2079
# print('number of 1:', (df['Class'] == 1).sum())
# print('number of 0:', (df['Class'] == 0).sum())


for index, row in df.iterrows():
    image_name = row['Image'] + '.jpg'
    image_class = row['Class']
    source_path = os.path.join(image_directory, image_name)

    if os.path.exists(source_path):
        if image_class == 0:
            destination = class_0_dir
        elif image_class == 1:
            destination = class_1_dir
        else:
            print(f"Unknown class {image_class} for image {image_name}")
            continue

        shutil.move(source_path, os.path.join(destination, image_name))
    # else:
    #     print(f"Image not found: {source_path}")

print("Images have been sorted.")
