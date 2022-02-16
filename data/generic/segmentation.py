import glob
from PIL import Image


class GenericSegmentation:

    def __init__(self, root, mode, sync_transform=None):
        self.sync_transform = sync_transform
        self.img_paths, self.seg_paths = self._get_paths(root, mode)

    def _get_paths(self, root, mode):
        img_paths = sorted(glob.glob(f'{root}/{mode}/imgs/*.jpg'))
        seg_paths = sorted(glob.glob(f'{root}/{mode}/s_segs/*.png'))
        return img_paths, seg_paths

    def __getitem__(self, index):
        img = Image.open(self.img_paths[index]).convert('RGB')
        seg = Image.open(self.seg_paths[index])

        if self.sync_transform is not None:
            img, seg = self.sync_transform(img, seg)

        return img, seg

    def __len__(self):
        return len(self.img_paths)
