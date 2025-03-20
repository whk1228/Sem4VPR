from pathlib import Path
import numpy as np
from PIL import Image
from torch.utils.data import Dataset

DATASET_ROOT = '/root/autodl-tmp/Tokyo247/'
GT_ROOT = '/root/autodl-tmp/SemVG/dataset/'  # BECAREFUL, this is the ground truth that comes with GSV-Cities

path_obj = Path(DATASET_ROOT)
if not path_obj.exists():
    raise Exception(f'Please make sure the path {DATASET_ROOT} to Tokyo247 dataset is correct')

if not path_obj.joinpath('ref') or not path_obj.joinpath('query'):
    raise Exception(f'Please make sure the directories query and ref are situated in the directory {DATASET_ROOT}')


class TokyoDataset(Dataset):
    def __init__(self, input_transform=None):
        self.input_transform = input_transform

        # reference images names
        self.dbImages = np.load(GT_ROOT + 'Tokyo247/tokyo247_dbImages.npy')

        # query images names
        self.qImages = np.load(GT_ROOT + 'Tokyo247/tokyo247_qImages.npy')

        # ground truth
        self.ground_truth = np.load(GT_ROOT + 'Tokyo247/tokyo247_gt.npy', allow_pickle=True)

        # reference images then query images
        self.images = np.concatenate((self.dbImages, self.qImages))

        self.num_references = len(self.dbImages)
        self.num_queries = len(self.qImages)

    def __getitem__(self, index):
        img = Image.open(DATASET_ROOT + self.images[index])

        if self.input_transform:
            img = self.input_transform(img)

        return img, index

    def __len__(self):
        return len(self.images)