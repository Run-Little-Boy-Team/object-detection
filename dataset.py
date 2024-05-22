from torch.utils.data import Dataset
from augmentation import Augmentation
import glob
import cv2
import numpy as np
import torch
import os
import random
import utils
import matplotlib.pyplot as plt


class Dataset(Dataset):
    def __init__(self, configuration, test=False, augment=False):
        self.configuration = configuration
        self.test = test
        if self.test:
            self.x = glob.glob(configuration["test_path"] + "/*.jpg")
        else:
            self.x = glob.glob(configuration["train_path"] + "/*.jpg")
        self.y = [f"{os.path.splitext(file)[0]}.txt" for file in self.x]
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

        x = cv2.imread(x)
        if os.path.exists(y):
            y = [
                tuple(float(i) for i in line.split(" "))
                for line in open(y, "r").readlines()
            ]
        else:
            y = []

        if self.augment:
            x, y = self.augmentation([x], [y])
            x = x[0]
            y = y[0]

        x = self.preprocess(x)
        y = [(0, i[0] + self.configuration["class_id_offset"]) + i[1:] for i in y]
        y = np.array(y, dtype=np.float32)

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

    def show_sample(self):
        index = random.randint(0, len(self) - 1)
        x, y = self[index]
        x = x.numpy()
        y = y.numpy()
        x = x.transpose((1, 2, 0))
        x = cv2.cvtColor(x, cv2.COLOR_RGB2BGR)
        y = [value[1:] for value in y]
        x = utils.draw_annotation(x, y, self.configuration["classes"])
        cv2.imshow("image", x)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def distribution(self):
        distribution = {}
        for y in self.y:
            if os.path.exists(y):
                for line in open(y, "r").readlines():
                    line = line.split(" ")
                    key = int(line[0]) + self.configuration["class_id_offset"]
                    if key not in distribution:
                        distribution[key] = 0
                    distribution[key] += 1
        total = sum(distribution.values())
        for key in distribution:
            distribution[key] /= total
        sorted_keys = sorted(distribution.keys())
        distribution = {key: distribution[key] for key in sorted_keys}
        distribution = {
            self.configuration["classes"][int(key)]: value
            for key, value in distribution.items()
        }
        return distribution

    def show_distribution(self):
        distribution = self.distribution()
        plt.figure(figsize=(20, 5))
        plt.title(f"{'Test' if self.test else 'Train'} distribution")
        plt.bar(distribution.keys(), distribution.values())
        plt.xticks(rotation=270)
        plt.show()


def collate_fn(batch):
    img, label = zip(*batch)
    for i, l in enumerate(label):
        if l.shape[0] > 0:
            l[:, 0] = i
    return torch.stack(img), torch.cat(label, 0)


if __name__ == "__main__":
    configuration = utils.load_configuration("config.yaml")
    dataset = Dataset(configuration, augment=True)
    dataset.show_sample()
