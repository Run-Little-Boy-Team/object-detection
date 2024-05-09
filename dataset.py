from torch.utils.data import Dataset
from augmentation import Augmentation
import glob
import cv2
import utils
import numpy as np


class Dataset(Dataset):
    def __init__(self, configuration, test=False, augment=False):
        self.configuration = configuration
        if test:
            self.x = glob.glob(configuration["test_path"] + "/*.jpg")
            self.y = glob.glob(configuration["test_path"] + "/*.txt")
        else:
            self.x = glob.glob(configuration["train_path"] + "/*.jpg")
            self.y = glob.glob(configuration["train_path"] + "/*.txt")
        self.augment = augment
        if self.augment:
            self.augmentation = Augmentation(configuration)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, index):
        x = self.x[index]
        y = self.y[index]

        x = cv2.imread(x)
        y = [
            tuple(float(i) for i in line.split(" "))
            for line in open(y, "r").readlines()
        ]

        if self.augment:
            x, y = self.augmentation([x], [y])
            x = x[0]
            y = y[0]

        y = np.array(y)

        return x, y


if __name__ == "__main__":
    configuration = utils.load_configuration("config.yaml")
    dataset = Dataset(configuration, augment=True)
    x, y = dataset[0]
    x = utils.draw_annotation(x, y, configuration["classes"])
    cv2.imshow("image", x)
    cv2.waitKey(0)
