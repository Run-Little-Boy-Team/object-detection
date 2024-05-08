from torch.utils.data import Dataset
from augmentation import Augmentation
from random import shuffle
from glob import glob
from cv2 import (
    imread,
    imshow,
    waitKey,
)
from utils import load_configuration, init_logger, draw_annotation


class Dataset(Dataset):
    def __init__(self, configuration, test=False, shuffle_order=True, augment=True):
        self.configuration = configuration
        if test:
            self.x = glob(configuration["test_path"] + "/*.jpg")
            self.y = glob(configuration["test_path"] + "/*.txt")
        else:
            self.x = glob(configuration["train_path"] + "/*.jpg")
            self.y = glob(configuration["train_path"] + "/*.txt")
        self.augment = augment
        if self.augment:
            self.augmentation = Augmentation(configuration)
        if shuffle_order:
            indexes = list(range(len(self.x)))
            shuffle(indexes)
            self.x = [self.x[i] for i in indexes]
            self.y = [self.y[i] for i in indexes]

    def __len__(self):
        return len(self.x)

    def __getitem__(self, index):
        x = self.x[index]
        y = self.y[index]

        x = imread(x)
        y = [
            tuple(float(i) for i in line.split(" "))
            for line in open(y, "r").readlines()
        ]

        if self.augment:
            x, y = self.augmentation([x], [y])
            x = x[0]
            y = y[0]

        return x, y


if __name__ == "__main__":
    configuration = load_configuration("config.yaml")
    dataset = Dataset(configuration, augment=True)
    x, y = dataset[0]
    x = draw_annotation(x, y, configuration["classes"])
    imshow("image", x)
    waitKey(0)
