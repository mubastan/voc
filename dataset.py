"""
Custom object detection dataset loader from JSON for PyTorch with OpenCV, and in-memory PASCAL VOC evaluation.

This dataset loader is for annotation files in JSON format, keyed by image file name (see below for details),
but easy to modify to load other formats too, e.g., csv.
The loader is compatible with the widely used DL object detectors.

It performs PASCAL VOC object detection evaluation in memory; no need to save the ground truth bbox annotations/labels
to individual XML files for each image, as in the original VOC evaluation implementation.

The code was kept simple, to bare minimums to make it easy to read/understand and modify/adapt.

See '__main__' below for how to use this loader to load and evaluate.

Author: Muhammet Bastan, mubastan@gmail.com, 06 November 2018
"""
import os
import sys
import numpy
import json
import cv2
from torch.utils.data import Dataset

from voc_eval import voc_evaluate

class DetectionDataset(Dataset):
    def __init__(self, annotation_file_path, image_dir, classes, preproc=None, mode='test'):
        """
        Custom object detection dataset loader for PyTorch with in-memory PASCAL VOC evaluation

        :param annotation_file_path: Full file path for the ground truth annotations in JSON format
        :param image_dir: Image directory
        :param classes: List of classes, e.g., ['__background__', 'bunny', 'duck', 'goat'] - first class is background
        :param preproc: Pre-processing to be applied to images and bounding boxes (see object detector implementations)
        :param mode: train/test mode of operation
        """

        assert mode in {'train', 'test'}, f"mode should be one of (train, test)"
        assert len(classes) >= 2, f"There must be at least two classes, first one being the __background__"

        self.classes = classes
        self.image_dir = image_dir
        self.preproc = preproc
        self.mode = mode
        self.cls2ind = dict(zip(self.classes, range(len(self.classes))))

        self.load_annotations(annotation_file_path)

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, index):
        """
        Load and return the image at `index`
        :param index:
        :return: image as OpenCV image
        """
        img, image_file = self.load_image(index)
        # return only the image at test time, no bbox/label
        if self.mode == 'test':
            if self.preproc is not None:
                img = self.preproc(img)
            return img

        # bounding boxes and labels: numpy array of [x1,y1,x2,y2,class_id]
        objects = self.annotations[image_file]

        if self.preproc is not None:
            img, objects = self.preproc(img, objects)

        return img, objects

    def load_image(self, index):
        """
        Load and return image at 'index' as OpenCV image

        :param index: index to the image list
        :return: image (as OpenCV image), and image filename
        """
        image_file = self.image_list[index]
        image_path = os.path.join(self.image_dir, image_file)
        img = cv2.imread(image_path, cv2.IMREAD_COLOR)
        
        return img, image_file

    def get_objects(self, ann_dict, image_file):

        anns = ann_dict.get(image_file, [])
        objects = []
        for bbox, label in anns:
            x1, y1, x2, y2 = [float(x) for x in bbox[:4]]
            cid = self.cls2ind[label]
            objects.append([x1, y1, x2, y2, cid])

        return objects

    def load_annotations(self, annotation_file_path):
        """
        Load ground truth bounding box annotations with class labels from JSON file
        See labels/ folder for samples.
        Loads image list (file names) to self.image_list,
        and bounding annotations to self.annotations dictionary, keyed on image file name,
        e.g., self.annotations["image1.jpg"] = numpy.array([[10,20, 200, 300, 1]])

        :param annotation_file_path: full path of the annotation file, which is in JSON file
                                     (easy to change to load any format)
        :return:
        """

        with open(annotation_file_path) as fp:
            ann_dict = json.load(fp)
        print('load_annotations:', annotation_file_path, len(ann_dict), 'images')

        self.image_list = []
        self.annotations = {}
        for image_file in ann_dict:
            objects = self.get_objects(ann_dict, image_file)
            if len(objects)==0 and not self.load_negatives: continue
            objects = numpy.array(objects)
            self.image_list.append(image_file)
            self.annotations[image_file] = objects
        
        print(f'Loaded annotations.')
        print(f'Number of images with annotations: {len(self.image_list)}/{len(self.annotations)}', flush=True)

    def evaluate_detections(self, detections_dict):
        aps = []
        print(f'-----------------------------------------')
        print(f'Class \tAP \tRec \tPrec ')
        print(f'-----------------------------------------')
        for classname in self.classes:
            # Skip the background class
            if classname == '__background__':
                continue
            cid = self.cls2ind[classname]
            rec, prec, ap = voc_evaluate(detections_dict, self.annotations, cid, ovthresh=0.5)
            aps += [ap]
            print(f'{classname} \t{ap:.2f} \t{rec:.2f} \t{prec:.2f}')

        # Mean Average Precision (mAP)
        mAP = numpy.mean(aps)
        print('-----------------------------------------')
        print('mAP: \t{:.2f}'.format(mAP))
        print('-----------------------------------------')

        return aps, mAP

if __name__ == '__main__':
    annotation_file_path = 'labels/test.json'
    image_dir = 'images/'
    classes = ['__background__', 'logo']
    dataset = DetectionDataset(annotation_file_path, image_dir, classes, preproc=None, mode='test')
    print(dataset.annotations)

    # Suppose we have the following detections for '__background__' vs 'logo' detection task:
    detections = {0: {},
                  1: {
                      "image10.jpg": numpy.array([[100,150,200,300, 0.9], [300,140,400,600, 0.6]]),
                      "image11.jpg": numpy.array([[100, 150, 200, 300, 0.8]]),
                      "image12.jpg": numpy.array([[100, 150, 200, 300, 0.4], [300, 140, 400, 600, 0.55]])
                  }
                  }
    dataset.evaluate_detections(detections)
    # This is the same as the ground truth, so we should get 100% for all the metrics (recall, precision, ap)
    # -----------------------------------------
    # Class   AP      Rec     Prec
    # -----------------------------------------
    # logo    1.00    1.00    1.00
    # -----------------------------------------
    # mAP:    1.00
    # -----------------------------------------


    # If we have the following detections (not the same as the ground truth):
    detections = {0: {},
                  1: {
                      "image10.jpg": numpy.array([[100, 150, 200, 300, 0.9]]),
                      "image11.jpg": numpy.array([[100, 150, 200, 300, 0.8]]),
                      "image12.jpg": numpy.array([[100, 150, 200, 300, 0.4], [400, 140, 800, 600, 0.55]])
                  }
                  }

    dataset.evaluate_detections(detections)
    # Since these detections are not the same as ground truth, we get lower accuracy:
    # -----------------------------------------
    # Class   AP      Rec     Prec
    # -----------------------------------------
    # logo    0.55    0.60    0.75
    # -----------------------------------------
    # mAP:    0.55
    # -----------------------------------------
