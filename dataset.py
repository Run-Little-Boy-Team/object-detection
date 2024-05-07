from torch.utils.data import Dataset
from augmentation import Augmentation
from random import shuffle
from glob import glob
from cv2 import imread, cvtColor, COLOR_BGR2RGB, resize, rectangle, waitKey, imshow


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
            self.augmentation = Augmentation(configuration, log_level=None)
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
        x = cvtColor(x, COLOR_BGR2RGB)

        size = int(self.configuration["input_size"])
        x = resize(x, (size, size))
        y = [
            tuple(float(i) for i in line.split(" "))
            for line in open(y, "r").readlines()
        ]

        if self.augment:
            x, y = self.augmentation([x], [y], resize=False)
            x = x[0]
            y = y[0]

        return x, y


if __name__ == "__main__":
    from yaml import safe_load

    configuration = safe_load(open("config.yaml", "r"))
    dataset = Dataset(configuration, augment=False, shuffle_order=False)
    x, y = dataset[0]
    shape = x.shape
    for i in range(len(y)):
        bbox = y[i]

        bbox = (
            bbox[0],
            int(bbox[1] * shape[1]),
            int(bbox[2] * shape[0]),
            int(bbox[3] * shape[1]),
            int(bbox[4] * shape[0]),
        )

        x = rectangle(x, bbox[1:], (0, 255, 0), 2)

    imshow(str(y), x)
    waitKey(0)
