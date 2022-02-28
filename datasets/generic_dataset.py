import glob
from PIL import Image
from pycocotools.coco import COCO
import numpy as np


class GenericSegmentationDataset:

    def __init__(self, root, mode, transform=None):
        self.transform = transform
        self.img_paths, self.seg_paths = self._get_paths(root, mode)

    def _get_paths(self, root, mode):
        img_paths = sorted(glob.glob(f'{root}/{mode}/imgs/*.jpg'))
        seg_paths = sorted(glob.glob(f'{root}/{mode}/s_segs/*.png'))
        return img_paths, seg_paths

    def __getitem__(self, index):
        img = Image.open(self.img_paths[index]).convert('RGB')
        seg = Image.open(self.seg_paths[index])

        if self.transform is not None:
            img, seg = self.transform(data=img, target=seg)

        return img, seg

    def __len__(self):
        return len(self.img_paths)


class COCODataset:

    def __init__(self, root, mode, transform):
        self.folder = f'{root}/{mode}'
        self.transform = transform
        self.coco = COCO(f'{self.folder}/annots.json')
        self.img_ids = self.coco.getImgIds()

    def __getitem__(self, index):
        img_id = self.img_ids[index]
        img_info = self.coco.loadImgs(img_id)[0]
        img = Image.open(f'{self.folder}/imgs/{img_info["file_name"]}').convert('RGB')

        seg = np.zeros((img_info['height'], img_info['width']))
        ann_ids = self.coco.getAnnIds(img_id)
        anns = self.coco.loadAnns(ann_ids)
        for ann in anns:
            # TODO offsetにするか？
            clazz = ann['category_id'] + 1
            seg = np.maximum(self.coco.annToMask(ann) * clazz, seg)
        seg = Image.fromarray(seg, 'L')

        if self.transform is not None:
            img, seg = self.transform(data=img, target=seg)

        return img, seg

    def __len__(self):
        return len(self.img_ids)
