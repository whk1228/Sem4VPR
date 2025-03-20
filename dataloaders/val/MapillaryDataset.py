from torch.utils.data import Dataset
import numpy as np
from PIL import Image
from pathlib import Path
DATASET_ROOT = '/root/autodl-tmp/msls_val_dataset/'
GT_ROOT = '/root/autodl-tmp/SemVG/dataset/'  # BECAREFUL, this is the ground truth that comes with GSV-Cities


class MSLS(Dataset):
    def __init__(self, input_transform=None):
        self.input_transform = input_transform

        self.dbImages = np.load(GT_ROOT + 'msls_val/msls_val_dbImages_new.npy')
        self.qIdx = np.load(GT_ROOT + 'msls_val/msls_val_qIdx.npy')
        self.qImages = np.load(GT_ROOT + 'msls_val/msls_val_qImages_new.npy')
        self.ground_truth = np.load(GT_ROOT + 'msls_val/msls_val_pIdx.npy', allow_pickle=True)

        # reference images then query images
        self.images = np.concatenate((self.dbImages, self.qImages[self.qIdx]))
        self.num_references = len(self.dbImages)
        self.num_queries = len(self.qImages[self.qIdx])

    def __getitem__(self, index):
        img = Image.open(DATASET_ROOT + self.images[index])

        if self.input_transform:
            img = self.input_transform(img)

        return img, index

    def __len__(self):
        return len(self.images)
    def save_predictions(self, preds, path):
        with open(path, 'w') as f:
            for i in range(len(preds)):
                q = Path(self.qImages[i]).stem
                db = ' '.join([Path(self.dbImages[j]).stem for j in preds[i]])
                # db = ' '.join([Path(self.dbImages[j]).stem for j in preds.get(i, [])])
                f.write(f"{q} {db}\n")
