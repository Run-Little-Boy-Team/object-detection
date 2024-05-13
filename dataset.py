from torch.utils.data import Dataset
from augmentation import Augmentation
import glob
import cv2
import numpy as np
import torch

class Dataset(Dataset):
    def __init__(self, configuration, test=False, augment=False):
        self.configuration = configuration
        if test:
            self.x = glob.glob(configuration["test_path"] + "/*.jpg")
            self.y = glob.glob(configuration["test_path"] + "/*.txt")
        else:
            self.x = glob.glob(configuration["train_path"] + "/*.jpg")
            self.y = glob.glob(configuration["train_path"] + "/*.txt")
        self.x.sort()
        self.y.sort()
        self.augment = augment
        if self.augment:
            self.augmentation = Augmentation(configuration)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, index):
        x = self.x[index]
        y = self.y[index]

        if x.split(".")[0] != y.split(".")[0]:
            raise Exception("Image and annotation file mismatch: ", x, y)

        x = cv2.imread(x)
        y = [
            tuple(float(i) for i in line.split(" "))
            for line in open(y, "r").readlines()
        ]

        if self.augment:
            x, y = self.augmentation([x], [y])
            x = x[0]
            y = y[0]

        x = self.preprocess(x)
        y = [(0,) + i for i in y]
        y = np.array(y)

        x = torch.from_numpy(x)
        y = torch.from_numpy(y)

        return x, y

    def preprocess(self, image: cv2.typing.MatLike) -> cv2.typing.MatLike:
        input_size = self.configuration["input_size"]
        processed_image = image.copy()
        processed_image = cv2.resize(processed_image, (input_size, input_size))
        processed_image = cv2.cvtColor(processed_image, cv2.COLOR_BGR2RGB)
        processed_image = processed_image.astype("float32")
        processed_image /= 255.0
        processed_image = processed_image.transpose((2, 0, 1))
        return processed_image


if __name__ == "__main__":
    import utils

    configuration = utils.load_configuration("config.yaml")
    dataset = Dataset(configuration, augment=True)
    x, y = dataset[0]
    x = x.transpose((1, 2, 0))
    x = cv2.cvtColor(x, cv2.COLOR_RGB2BGR)
    x = utils.draw_annotation(x, y, configuration["classes"])
    cv2.imshow("image", x)
    cv2.waitKey(0)
