import os,shutil
import json
from tqdm import tqdm
from config_c3d import cfg
import random

def copy_search_file(srcDir, drsDir):
    ls = os.listdir(srcDir)
    for line in tqdm(ls):
        filePath = os.path.join(srcDir, line)
        if os.path.isfile(filePath):
            shutil.copy2(filePath, drsDir)

def get_c3d_root(path):
    return '/'.join(path.split("/")[:-4])

def move_files(root,categories):
    for cate in categories:
        for label in ['gt','partial']:
            train_dir = os.path.join(root,'train',label,str(cate))
            val_dir = os.path.join(root,'val',label,str(cate))
            copy_search_file(val_dir,train_dir)

def get_categories(dc):
    return [item["taxonomy_id"] for item in dc[1:]]

def extent_c3d(dc,rate,drs):
    for i in range(1,len(dc)):
        print(len(dc[i]['val']))
        random.shuffle(dc[i]['val'])
        dc[i]['train'].extend(
            dc[i]['val'][:int(rate*len(dc[i]['val']))]
        )
    with open(drs,'w') as f:
        json.dump(dc,f)

c3d_root=get_c3d_root(cfg.DATASETS.COMPLETION3D.PARTIAL_POINTS_PATH)
with open(cfg.DATASETS.COMPLETION3D.CATEGORY_FILE_PATH) as f:
    dataset_categories = json.loads(f.read())
cates = get_categories(dataset_categories)
move_files(c3d_root,cates)

extent_c3d(dataset_categories,0.5,"./datasets/Completion3D_new.json")
