import os
import shutil
from sklearn.model_selection import train_test_split

base_dir = 'data/mrl'
categories = ['Close-Eyes', 'Open-Eyes']
splits = ['train', 'val', 'test']

for split in splits:
    for category in categories:
        os.makedirs(os.path.join(base_dir, split, category), exist_ok=True)

def split_data(category):
    category_path = os.path.join(base_dir, category)
    images = [f for f in os.listdir(category_path) if os.path.isfile(os.path.join(category_path, f))]
    
    train_val, test = train_test_split(images, test_size=0.2, random_state=42)
    train, val = train_test_split(train_val, test_size=0.25, random_state=42)
    
    for split, split_images in zip(splits, [train, val, test]):
        for image in split_images:
            src = os.path.join(category_path, image)
            dst = os.path.join(base_dir, split, category, image)
            shutil.copy(src, dst)

for category in categories:
    split_data(category)