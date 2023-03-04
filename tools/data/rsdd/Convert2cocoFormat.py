import sys
import os
import json
import xml.etree.ElementTree as ET
import mmcv
import numpy as np
import os.path as osp
from mmrotate.core import eval_rbbox_map, obb2poly_np, poly2obb_np

START_BOUNDING_BOX_ID = 1
PRE_DEFINE_CATEGORIES = {}
# If necessary, pre-define category and its id
#  PRE_DEFINE_CATEGORIES = {"aeroplane": 1, "bicycle": 2, "bird": 3, "boat": 4,
                         #  "bottle":5, "bus": 6, "car": 7, "cat": 8, "chair": 9,
                         #  "cow": 10, "diningtable": 11, "dog": 12, "horse": 13,
                         #  "motorbike": 14, "person": 15, "pottedplant": 16,
                         #  "sheep": 17, "sofa": 18, "train": 19, "tvmonitor": 20}


def get(root, name):
    vars = root.findall(name)
    return vars


def get_and_check(root, name, length):
    vars = root.findall(name)
    if len(vars) == 0:
        raise NotImplementedError('Can not find %s in %s.'%(name, root.tag))
    if length > 0 and len(vars) != length:
        raise NotImplementedError('The size of %s is supposed to be %d, but is %d.'%(name, length, len(vars)))
    if length == 1:
        vars = vars[0]
    return vars


def get_filename_as_int(filename):
    try:
        filename = os.path.splitext(filename)[0]
        return int(filename)
    except:
        raise NotImplementedError('Filename %s is supposed to be an integer.'%(filename))


def convert(xml_dir, ann_file, json_file, version = "le90"):

    json_dict = { "info": {  "contributor": "TerrencePai",
                          "data_created": "2023",
                          "description": "This is correct RSDD dataset.",
                          "url": "https://github.com/TerrencePai/mmrotate",
                          "year": 2023 },
                 "images":[], "categories": [],
                 "annotations": [], "licenses":[], "type": "instances",}

    img_ids = mmcv.list_from_file(ann_file)
    bnd_id = START_BOUNDING_BOX_ID
    categories = PRE_DEFINE_CATEGORIES
    for image_id, filename in enumerate(img_ids):
        
        xml_path = osp.join(xml_dir,f'{filename}.xml')
        tree = ET.parse(xml_path)
        root = tree.getroot()
        
        image = {'id':image_id+1, 'file_name': f"{filename}.jpg", 
                 'height': int(root.find('size/height').text), 
                 'width': int(root.find('size/width').text),
                 'depth': int(root.find('size/depth').text),
                 "polarization": root.find('polarization').text, 
                 "resolution": float(root.find('resolution').text),
                 "segmented": int(root.find('segmented').text)}
        json_dict['images'].append(image)
        
        for obj in root.findall('object'):
            
            w = float(obj.find('robndbox/w').text)
            h = float(obj.find('robndbox/h').text)
            bbox = np.array([[
                float(obj.find('robndbox/cx').text),
                float(obj.find('robndbox/cy').text),
                max(w,h),
                min(w,h),
                float(obj.find('robndbox/angle').text),0]], dtype=np.float32)

            area = (bbox[0,2]*bbox[0,3]).item()
            polygon = obb2poly_np(bbox, 'le90')[0, :-1].astype(np.float32)
            if version != 'le90':
                bbox = np.array(
                    poly2obb_np(polygon, version), dtype=np.float32)
            else:
                bbox = bbox[0, :-1]  
            bbox = bbox.tolist()
            polygon = polygon.tolist()
            category = get_and_check(obj, 'name', 1).text
            if category not in categories:
                new_id = len(categories)+1
                categories[category] = new_id
            category_id = categories[category]                                    
            json_dict['annotations'].append({'area': area,
                                                 'segmentation': polygon,
                                                 'iscrowd': int(obj.find('difficult').text),
                                                 'image_id': image_id+1, 
                                                 'bbox': bbox,'ignore': 0,
                                                 'category_id': category_id, 
                                                 'id': bnd_id,
                                                 "ignore": 0
                                                 } )        
            bnd_id = bnd_id + 1        
                    
    for cate, cid in categories.items():
        cat = {'supercategory': 'none', 'id': cid, 'name': cate}
        json_dict['categories'].append(cat)
    json_fp = open(json_file, 'w')
    json_str = json.dumps(json_dict)
    json_fp.write(json_str)
    json_fp.close()

for data in ["train", "test", "test_inshore", "test_offshore"]:
    convert("/mmrotate/data/rsdd/Annotations", f"/mmrotate/data/rsdd/ImageSets/{data}.txt", f"/mmrotate/data/rsdd/ImageSets/{data}.json", version = "le90")    
