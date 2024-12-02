from IPython.display import Image  # for displaying images
import os 
import xml.etree.ElementTree as ET
from tqdm import tqdm
from PIL import Image, ImageDraw
import numpy as np
import matplotlib.pyplot as plt
from glob import glob
import argparse
import re

def get_class_names_logodet(path):
    classes = {}
    class_number = 0
    for folder in glob(path + '/*/', recursive = True):
        for subfolder in glob(folder + '/*/', recursive = True):
            class_name = subfolder.split('/')[-2]
            classes[class_name] = class_number
            class_number += 1

    return classes


def get_class_names_yaml_logodet(path):
    classes = []
    for folder in glob(path + '/*/', recursive = True):
        for subfolder in glob(folder + '/*/', recursive = True):
            classes.append(subfolder.split('/')[-2])

    return classes


def get_annotations_logodet(path):
    annotations = []
    for folder in glob(path + '/*/', recursive = True):
        for subfolder in glob(folder + '/*/', recursive = True):
            for xml in glob(subfolder + '*.xml'):
                annotations.append(xml)

    return annotations


def get_class_names(path):
    classes = {}
    class_number = -1
    current_class = ''
    with open(path) as f:
        lines = f.readlines()
        class_name = ''
        for line in lines:
            class_name = str(line.split(' ')[1])

            if current_class != class_name:
                class_number += 1
            classes[class_name] = class_number
            current_class = class_name

    return classes


def get_image_paths(path):
    annotations = []
    for image in glob(path + '/*.jpg'):
        annotations.append(image)

    return annotations


def extract_info_from_annotations(line):
    class_name = str(line.split(' ')[1])
    xmin = int(line.split(' ')[3])
    ymin = int(line.split(' ')[4])
    xmax = int(line.split(' ')[5])
    ymax = int(line.split(' ')[6])

    return {'class': class_name, 'xmin': xmin, 'ymin': ymin, 'xmax': xmax, 'ymax': ymax}


def get_annotations(path, current_image_name, width, height):
    with open(path) as f:
        lines = f.readlines()
        bboxes = []
        for line in lines:
            image_name = str(line.split(' ')[0].split('.')[0])

            if current_image_name == image_name:
                new_bbox = extract_info_from_annotations(line)
                if new_bbox not in bboxes:
                    bboxes.append(new_bbox)

    return {'bboxes': bboxes, 'filename': str(current_image_name + '.jpg'), 'image_size': (width, height, 3)}


def extract_info_from_xml(xml_file):
    root = ET.parse(xml_file).getroot()
    
    # Initialise the info dict 
    info_dict = {}
    info_dict['bboxes'] = []

    # Parse the XML Tree
    for elem in root:
        # Get the file name 
        if elem.tag == "filename":
            info_dict['filename'] = elem.text
            
        # Get the image size
        elif elem.tag == "size":
            image_size = []
            for subelem in elem:
                image_size.append(int(subelem.text))
            
            info_dict['image_size'] = tuple(image_size)
        
        # Get details of the bounding box 
        elif elem.tag == "object":
            bbox = {}
            for subelem in elem:
                if subelem.tag == "name":
                    bbox["class"] = re.split(r'-\d+', subelem.text)[0]
                    
                elif subelem.tag == "bndbox":
                    for subsubelem in subelem:
                        bbox[subsubelem.tag] = int(subsubelem.text)            
            info_dict['bboxes'].append(bbox)
    
    return info_dict


# Convert the info dict to the required yolo format and write it to disk
def convert_to_yolov5(info_dict, ann, class_name_to_id_mapping):
    print_buffer = []
    
    # For each bounding box
    for b in info_dict["bboxes"]:
        try:
            class_id = class_name_to_id_mapping[b["class"]]

            # Transform the bbox co-ordinates as per the format required by YOLO v5
            b_center_x = (b["xmin"] + b["xmax"]) / 2 
            b_center_y = (b["ymin"] + b["ymax"]) / 2
            b_width    = (b["xmax"] - b["xmin"])
            b_height   = (b["ymax"] - b["ymin"])
            
            # Normalise the co-ordinates by the dimensions of the image
            image_w, image_h, image_c = info_dict["image_size"]  
            b_center_x /= image_w 
            b_center_y /= image_h 
            b_width    /= image_w 
            b_height   /= image_h 
            
            #Write the bbox details to the file 
            print_buffer.append("{} {:.3f} {:.3f} {:.3f} {:.3f}".format(class_id, b_center_x, b_center_y, b_width, b_height))
        except KeyError:
            print("Invalid Class. Must be one from ", class_name_to_id_mapping.keys())
        
    # Name of the file which we have to save 
    out_ann = ann.split('.')[0]
    out_ann = out_ann.split('/')
    del out_ann[-1]
    out_ann = '/'.join(out_ann)
    save_file_name = os.path.join(out_ann, info_dict["filename"].replace("jpg", "txt"))
    
    # Save the annotation to disk
    print("\n".join(print_buffer), file= open(save_file_name, "w"))


def plot_bounding_box(image, annotation_list, class_id_to_name_mapping):
    annotations = np.array(annotation_list)
    w, h = image.size
    
    plotted_image = ImageDraw.Draw(image)

    transformed_annotations = np.copy(annotations)
    transformed_annotations[:,[1,3]] = annotations[:,[1,3]] * w
    transformed_annotations[:,[2,4]] = annotations[:,[2,4]] * h 
    
    transformed_annotations[:,1] = transformed_annotations[:,1] - (transformed_annotations[:,3] / 2)
    transformed_annotations[:,2] = transformed_annotations[:,2] - (transformed_annotations[:,4] / 2)
    transformed_annotations[:,3] = transformed_annotations[:,1] + transformed_annotations[:,3]
    transformed_annotations[:,4] = transformed_annotations[:,2] + transformed_annotations[:,4]
    
    for ann in transformed_annotations:
        obj_cls, x0, y0, x1, y1 = ann
        plotted_image.rectangle(((x0,y0), (x1,y1)))
        
        plotted_image.text((x0, y0 - 10), class_id_to_name_mapping[(int(obj_cls))])
    
    plt.imshow(np.array(image))
    plt.show()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, required=True, help='flickr27 or logodet3k')
    parser.add_argument('--plot', action='store_true', help='To plot converted bboxes')
    parser.add_argument('--image', default='data/flickr_logos_27_dataset/flickr_logos_27_dataset_images/4771736332.txt', type=str, help='path to plot image')
    opt = parser.parse_args()

    if opt.dataset == 'flickr27':
        annotations_path = 'data/flickr_logos_27_dataset/flickr_logos_27_dataset_training_set_annotation.txt'
        class_name_to_id_mapping = get_class_names(annotations_path)
        class_id_to_name_mapping = dict(zip(class_name_to_id_mapping.values(), class_name_to_id_mapping.keys()))
        image_paths = get_image_paths('data/flickr_logos_27_dataset/flickr_logos_27_dataset_images')


        for image_path in tqdm(image_paths):
            current_image = Image.open(image_path)
            width, height = current_image.size
            current_image_name = image_path.split('/')[-1].split('.')[0]
            out = get_annotations(annotations_path, current_image_name, width, height)
            convert_to_yolov5(out, image_path, class_name_to_id_mapping)

    else:
        # Dictionary that maps class names to IDs
        class_name_to_id_mapping = get_class_names_logodet('data/LogoDet-3K')
        annotations = get_annotations_logodet('data/LogoDet-3K')
        class_id_to_name_mapping = dict(zip(class_name_to_id_mapping.values(), class_name_to_id_mapping.keys()))

        # Convert and save the annotations
        for ann in tqdm(annotations):
            info_dict = extract_info_from_xml(ann)
            convert_to_yolov5(info_dict, ann, class_name_to_id_mapping)

    # Get any random annotation file
    # To test the annotations please uncomment the lines below
    if opt.plot:
        annotation_file = opt.image
        with open(annotation_file, "r") as file:
            annotation_list = file.read().split("\n")[:-1]
            annotation_list = [x.split(" ") for x in annotation_list]
            annotation_list = [[float(y) for y in x ] for x in annotation_list]

        # Get the corresponding image file
        image_file = annotation_file.replace("txt", "jpg")
        assert os.path.exists(image_file)

        #Load the image
        image = Image.open(image_file)

        #Plot the Bounding Box
        plot_bounding_box(image, annotation_list, class_id_to_name_mapping)


if __name__ == "__main__":
    main()
