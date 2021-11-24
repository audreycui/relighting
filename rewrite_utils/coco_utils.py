import os

import matplotlib.pyplot as plt
from PIL import Image
from pycocotools.coco import COCO

DATA_ROOT = os.environ.get('DATA_ROOT', '/data/datasets')
COCO_DATA_ROOT = os.path.join(DATA_ROOT, 'CocoStuff')


def get_coco(data_type='instances', split='val', year='2017'):
    return COCO(
        os.path.join(COCO_DATA_ROOT, f'annotations/{data_type}_{split}{year}.json')
    )


class COCOCaptionsDataset:
    def __init__(self, split='val', year='2017', transform=None, target_transform=None, no_image=False):
        self.split = split
        self.coco = get_coco(split=split, year=year)
        self.coco_caps = get_coco(data_type='captions', split=split, year=year)
        self.cats = self.coco.loadCats(self.coco.getCatIds())
        self.imgIds = self.coco.getImgIds(catIds=self.coco.getCatIds(self.cats))
        self.imgs = self.coco.loadImgs(self.imgIds)
        self.transform = transform
        self.target_transform = target_transform
        self.no_image = no_image

    def __getitem__(self, index):
        img = self.imgs[index]
        data = None
        if not self.no_image:
            data = Image.open(
                os.path.join(COCO_DATA_ROOT, f'{self.split}_img', img['file_name'])
            )
        ann_ids = self.coco_caps.getAnnIds(imgIds=img['id'])
        target = self.coco_caps.loadAnns(ann_ids)
        if self.transform is not None:
            data = self.transform(data)
        if self.target_transform is not None:
            data = self.target_transform(target)
        return data, target

    def __len__(self):
        return len(self.imgs)

    def show_example(self, example):
        img, anns = example
        plt.imshow(img)
        plt.axis('off')
        plt.show()
        self.coco_caps.showAnns(anns)
