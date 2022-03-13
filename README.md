# voc
Simple PASCAL VOC object detection in-memory evaluation on a custom dataset in JSON format, without saving to disk. The evaluation supports the following metrics:
- Average precision (AP) for each class, and mean average precision (mAP) for all the classes
- Recall
- Precision

This code supports JSON format for the bounding box annotations, but it is easy to modify to support other formats, like csv. The JSON format:
```json
{
    "image_file": [[bounding_box, "class_label"], ...]
}
```
```json
{
    "image_file": [[[x1,y1,x2,y2], "class_label"], [[x1,y1,x2,y2], "class_label"]]
}
```

```json
{
    "image01.jpg": [[[100,150,200,300], "duck"], [[300,140,400,600], "bunny"]],
    "image02.jpg": [[[100,150,200,300], "goat"]],
    "image03.jpg": [[[100,150,200,300], "chicken"], [[300,140,400,600], "goose"]]
}
```

# Files
  - **dataset.py**: Custom dataset loader in PyTorch for JSON annotation files, with sample VOC evaluation code (see main at the end)
  - **voc_eval.py**: PASCAL VOC object detection evaluation in memory, adapted from the [original code](https://github.com/GOATmessi7/RFBNet/blob/master/data/voc_eval.py) (by Bharath Hariharan -- the original code requires the annotations to be saved in XML format for each image). This code performs the evaluation in memory, and does not require XML files.

# Usage
```python
    annotation_file_path = 'test.json'
    image_dir = './'
    classes = ['__background__', 'logo']
    dataset = DetectionDataset(annotation_file_path, image_dir, classes, preproc=None, mode='test')
    print(dataset.annotations)

    # Suppose we have the following detections for '__background__' vs 'logo' detection task:
    detections = {0: {},
                  1: {
                      "image10.jpg": numpy.array([[100, 150, 200, 300, 0.9]]),
                      "image11.jpg": numpy.array([[100, 150, 200, 300, 0.8]]),
                      "image12.jpg": numpy.array([[100, 150, 200, 300, 0.4], [400, 140, 800, 600, 0.55]])
                  }
                  }

    dataset.evaluate_detections(detections)
    # Output:
    # -----------------------------------------
    # Class   AP      Rec     Prec
    # -----------------------------------------
    # logo    0.55    0.60    0.75
    # -----------------------------------------
    # mAP:    0.55
    # -----------------------------------------
```
