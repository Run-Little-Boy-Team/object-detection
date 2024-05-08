from numpy import all, any, array, clip, float32, max, min, uint8
from os import makedirs
from time import time
from random import seed, random, uniform, randint
from utils import init_logger, load_configuration, draw_annotation
from glob import glob
from cv2 import (
    resize,
    imread,
    imshow,
    waitKey,
    destroyAllWindows,
    imwrite,
    cvtColor,
    COLOR_BGR2GRAY,
    COLOR_GRAY2BGR,
    COLOR_BGR2HSV,
    COLOR_HSV2BGR,
    split,
    merge,
    addWeighted,
    GaussianBlur,
    flip,
    warpAffine,
    getRotationMatrix2D,
    transform,
)
from cv2.typing import MatLike
from typing import Any


class Augmentation:
    def __init__(
        self,
        configuration: str | Any = None,
        name: str = "Augmentation",
    ):
        self.name = name
        self.logger = init_logger(name, configuration["log_level"])
        seed()

        t_0 = time()
        self.logger.info("Loading configuration...")

        if configuration is None:
            self.logger.error("Configuration must be provided")
            exit(1)

        self.configuration = configuration
        if isinstance(configuration, str):
            self.configuration = load_configuration(configuration)

        t_1 = time()
        self.logger.info(f"Configuration loaded in {(t_1 - t_0) * 1000:.2f} ms")

    def __call__(
        self,
        images: list[MatLike] | list[str] | MatLike | str = None,
        bboxes: list[list[tuple[int]]] | list[str] | list[tuple[int]] | str = None,
        resize_image: bool = False,
        context: bool = False,
        show: bool = False,
        save: bool = False,
        save_path: str = "./outputs",
        save_name: str = "augmented",
    ) -> list[tuple[MatLike, tuple[int]]]:
        start = time()

        if images is None:
            self.logger.error("Images must be provided")
            exit(1)
        if bboxes is None:
            self.logger.error("Bboxes must be provided")
            exit(1)
        if len(images) != len(bboxes):
            self.logger.error("Number of images and bboxes must be the same")
            exit(1)
        augmentation_per_image = self.configuration["augmentation"][
            "augmentation_per_image"
        ]
        if augmentation_per_image < 1:
            self.logger.error("Augmentation per image must be greater than 0")
            exit(1)

        if False not in [isinstance(image, str) for image in images]:
            self.logger.info("Loading images...")
            t_0 = time()
            images = [imread(image) for image in images]
            t_1 = time()
            self.logger.info(f"Images loaded in {(t_1 - t_0) * 1000:.2f} ms")

        self.logger.info("Loading bboxes...")
        t_0 = time()

        if False not in [isinstance(bbox, str) for bbox in bboxes]:
            bboxes = [
                [
                    [float(i) for i in line.split(" ")]
                    for line in open(bbox, "r").readlines()
                ]
                for bbox in bboxes
            ]

        for i, bbox in enumerate(bboxes):
            bboxes[i] = [
                (
                    int(b[0]),
                    int(b[1] * images[i].shape[1]),
                    int(b[2] * images[i].shape[0]),
                    int(b[3] * images[i].shape[1]),
                    int(b[4] * images[i].shape[0]),
                )
                for b in bbox
            ]

        t_1 = time()
        self.logger.info(f"Bboxes loaded in {(t_1 - t_0) * 1000:.2f} ms")

        if resize_image:
            self.logger.info("Resizing images...")
            t_0 = time()
            size = int(self.configuration["input_size"])
            for i in range(len(images)):
                x_ratio = size / images[i].shape[1]
                y_ratio = size / images[i].shape[0]
                images[i] = resize(images[i], (size, size))
                for j, b in enumerate(bboxes[i]):
                    bboxes[i][j] = (
                        b[0],
                        int(b[1] * x_ratio),
                        int(b[2] * y_ratio),
                        int(b[3] * x_ratio),
                        int(b[4] * y_ratio),
                    )

            t_1 = time()
            self.logger.info(f"Images resized in {(t_1 - t_0) * 1000:.2f} ms")

        images_outputs = []
        bboxes_outputs = []
        if context:
            backgrounds_path = self.configuration["augmentation"]["backgrounds_path"]
            backgrounds_color = [
                int(i)
                for i in self.configuration["augmentation"]["backgrounds_color"].split(
                    ";"
                )
            ]
            formats = ["jpg", "jpeg", "png"]
            backgrounds = []
            for format in formats:
                backgrounds.extend(glob(f"{backgrounds_path}/*.{format}"))
            if len(backgrounds) == 0:
                self.logger.error("Backgrounds must be provided")
                exit(1)

            self.logger.info(
                f"Running in context mode for {len(images)} image{'s'if len(images) > 1 else ''} and {augmentation_per_image} augmentation{'s'if augmentation_per_image > 1 else ''} per image..."
            )

            for i in range(len(images)):
                image = images[i]
                bbox = bboxes[i]
                mask = all(image == backgrounds_color[::-1], axis=-1)
                for _ in range(augmentation_per_image):
                    index = randint(0, len(backgrounds) - 1)
                    background = backgrounds[index]
                    if isinstance(background, str):
                        background = imread(background)
                    background = resize(background, (image.shape[:2][::-1]))
                    image[mask] = background[mask]
                    images_outputs.append(image.copy())
                    bboxes_outputs.append(bbox)

        else:
            self.logger.info(
                f"Running for {len(images)} image{'s'if len(images) > 1 else ''} and {augmentation_per_image} augmentation{'s'if augmentation_per_image > 1 else ''} per image..."
            )

            for i in range(len(images)):
                for _ in range(augmentation_per_image):
                    self.image = images[i].copy()
                    self.shape = self.image.shape
                    self.bbox = bboxes[i].copy()

                    self.__translate__()
                    self.__rotate__()
                    self.__scale__()
                    self.__stretch__()
                    self.__shear__()

                    self.__vertical_flip__()
                    self.__horizontal_flip__()

                    self.__monochrome__()
                    self.__hsv__()
                    self.__contrast__()
                    self.__sharpness__()

                    images_outputs.append(self.image)
                    bboxes_outputs.append(self.bbox)

        end = time()
        self.logger.info(f"Finished in {(end - start) * 1000:.2f} ms")

        for i, bboxes in enumerate(bboxes_outputs):
            image = images_outputs[i]
            bboxes_outputs[i] = [
                (
                    b[0],
                    b[1] / image.shape[1],
                    b[2] / image.shape[0],
                    b[3] / image.shape[1],
                    b[4] / image.shape[0],
                )
                for b in bboxes
            ]

        if save:
            self.logger.info("Saving images...")
            t_0 = time()

            makedirs(save_path, exist_ok=True)
            for i in range(len(images_outputs)):
                image = images_outputs[i]
                bbox = bboxes_outputs[i]
                imwrite(f"{save_path}/{save_name}_{i}.jpg", image)
                with open(f"{save_path}/{save_name}_{i}.txt", "w") as file:
                    file.writelines([" ".join(map(str, b)) + "\n" for b in bbox])

            t_1 = time()
            self.logger.info(f"Images saved in {(t_1 - t_0) * 1000:.2f} ms")

        if show:
            self.logger.info(
                "Showing augmented images, press any key to continue or ESC to exit..."
            )
            for i in range(len(images_outputs)):
                self.logger.info(f"{i+1}/{len(images_outputs)}")
                self.image = images_outputs[i]
                self.bbox = bboxes_outputs[i]
                draw = draw_annotation(
                    self.image, self.bbox, self.configuration["classes"]
                )
                imshow(self.name, draw)
                key = waitKey(0)
                if key == 27:
                    break
            destroyAllWindows()

        return images_outputs, bboxes_outputs

    def __translate__(self):
        probability = float(
            self.configuration["augmentation"]["translation_probability"]
        )
        if probability < 0 or probability > 1:
            self.logger.error("Translation probability must be between 0 and 1")
            exit(1)
        amplitude = float(self.configuration["augmentation"]["translation_amplitude"])
        if amplitude < 0:
            self.logger.error("Translation amplitude must be greater than 0")
            exit(1)
        if random() <= probability:
            x = uniform(-amplitude, amplitude) * self.shape[1]
            y = uniform(-amplitude, amplitude) * self.shape[0]

            translation_matrix = float32([[1, 0, x], [0, 1, y]])

            if self.bbox is not None:
                for i, b in enumerate(self.bbox):
                    self.bbox[i] = (
                        b[0],
                        int(b[1] + x),
                        int(b[2] + y),
                        b[3],
                        b[4],
                    )

            self.image = warpAffine(
                self.image, translation_matrix, (self.shape[1], self.shape[0])
            )

    def __rotate__(self):
        probability = float(self.configuration["augmentation"]["rotation_probability"])
        if probability < 0 or probability > 1:
            self.logger.error("Rotation probability must be between 0 and 1")
            exit(1)
        amplitude = float(self.configuration["augmentation"]["rotation_amplitude"])
        if abs(amplitude) > 90 or amplitude < 0:
            self.logger.error("Rotation amplitude must be between 0 and 90")
            exit(1)
        if random() <= probability:
            angle = uniform(-amplitude, amplitude)

            center = (int(self.shape[1]) // 2, int(self.shape[0]) // 2)

            rotation_matrix = getRotationMatrix2D(center, angle, 1.0)

            if self.bbox is not None:
                for i, b in enumerate(self.bbox):
                    c, x, y, w, h = b
                    x_min = x - w // 2
                    y_min = y - h // 2
                    x_max = x + w // 2
                    y_max = y + h // 2

                    points = array(
                        [[x_min, y_min], [x_max, y_min], [x_min, y_max], [x_max, y_max]]
                    ).reshape(4, 1, 2)
                    rotated_points = (
                        transform(points, rotation_matrix).reshape(4, 2).astype(int)
                    )

                    x_min = int(min(rotated_points[:, 0]))
                    x_max = int(max(rotated_points[:, 0]))
                    y_min = int(min(rotated_points[:, 1]))
                    y_max = int(max(rotated_points[:, 1]))

                    w = x_max - x_min
                    h = y_max - y_min
                    x = x_min + w // 2
                    y = y_min + h // 2

                    self.bbox[i] = (c, x, y, w, h)

            self.image = warpAffine(
                self.image, rotation_matrix, (int(self.shape[1]), int(self.shape[0]))
            )

    def __scale__(self):
        probability = float(self.configuration["augmentation"]["scaling_probability"])
        if probability < 0 or probability > 1:
            self.logger.error("Scaling probability must be between 0 and 1")
            exit(1)
        amplitude = float(self.configuration["augmentation"]["scaling_amplitude"])
        if amplitude < 0:
            self.logger.error("Scaling amplitude must be greater than 0")
            exit(1)
        if random() <= probability:
            scale = 1 + uniform(-amplitude, amplitude)

            center = (int(self.shape[1]) // 2, int(self.shape[0]) // 2)

            scale_matrix = getRotationMatrix2D(center, 0, scale)

            if self.bbox is not None:
                for i, b in enumerate(self.bbox):
                    points = array(
                        [
                            [b[1], b[2]],
                            [b[1] + b[3], b[2] + b[4]],
                        ]
                    ).reshape(2, 1, 2)
                    scaled_points = (
                        transform(points, scale_matrix).reshape(4).astype(int)
                    )
                    self.bbox[i] = (
                        b[0],
                        scaled_points[0],
                        scaled_points[1],
                        scaled_points[2] - scaled_points[0],
                        scaled_points[3] - scaled_points[1],
                    )

            self.image = warpAffine(
                self.image, scale_matrix, (int(self.shape[1]), int(self.shape[0]))
            )

    def __stretch__(self):
        probability = float(
            self.configuration["augmentation"]["stretching_probability"]
        )
        if probability < 0 or probability > 1:
            self.logger.error("Stretching probability must be between 0 and 1")
            exit(1)
        amplitude = float(self.configuration["augmentation"]["stretching_amplitude"])
        if amplitude < 0:
            self.logger.error("Streching amplitude must be greater than 0")
            exit(1)
        if random() <= probability:
            x = 1 + uniform(-amplitude, amplitude)
            y = 1 + uniform(-amplitude, amplitude)

            stretch_matrix = float32([[x, 0, 0], [0, y, 0]])

            if self.bbox is not None:
                for i, b in enumerate(self.bbox):
                    stretched_points = (
                        int(b[1] * x),
                        int(b[2] * y),
                        int((b[1] + b[3]) * x),
                        int((b[2] + b[4]) * y),
                    )
                    self.bbox[i] = (
                        b[0],
                        stretched_points[0],
                        stretched_points[1],
                        stretched_points[2] - stretched_points[0],
                        stretched_points[3] - stretched_points[1],
                    )

            stretched = warpAffine(
                self.image, stretch_matrix, (self.shape[1], self.shape[0])
            )
            self.image = stretched

    def __shear__(self):
        probability = float(self.configuration["augmentation"]["shearing_probability"])
        if probability < 0 or probability > 1:
            self.logger.error("Shearing probability must be between 0 and 1")
            exit(1)
        amplitude = float(self.configuration["augmentation"]["shearing_amplitude"])
        if amplitude < 0:
            self.logger.error("Shear amplitude must be greater than 0")
            exit(1)
        if random() <= probability:
            x = uniform(-amplitude, amplitude)
            y = uniform(-amplitude, amplitude)

            shear_matrix = float32([[1, x, 0], [y, 1, 0]])

            if self.bbox is not None:
                for i, b in enumerate(self.bbox):
                    c, x_min, y_min, w, h = b
                    x_max = x_min + w
                    y_max = y_min + h

                    points = array(
                        [[x_min, y_min], [x_max, y_min], [x_min, y_max], [x_max, y_max]]
                    ).reshape(4, 1, 2)
                    sheared_points = transform(points, shear_matrix).reshape(4, 2)

                    x_min = int(min(sheared_points[:, 0]))
                    y_min = int(min(sheared_points[:, 1]))
                    x_max = int(max(sheared_points[:, 0]))
                    y_max = int(max(sheared_points[:, 1]))

                    self.bbox[i] = (c, x_min, y_min, x_max - x_min, y_max - y_min)

            self.image = warpAffine(
                self.image, shear_matrix, (int(self.shape[1]), int(self.shape[0]))
            )

    def __vertical_flip__(self):
        probability = float(
            self.configuration["augmentation"]["vertical_flip_probability"]
        )
        if probability < 0 or probability > 1:
            self.logger.error("Vertical flip probability must be between 0 and 1")
            exit(1)
        if random() <= probability:
            if self.bbox is not None:
                for i, b in enumerate(self.bbox):
                    self.bbox[i] = (
                        b[0],
                        self.shape[1] - b[1],
                        b[2],
                        b[3],
                        b[4],
                    )

            self.image = flip(self.image, 1)

    def __horizontal_flip__(self):
        probability = float(
            self.configuration["augmentation"]["horizontal_flip_probability"]
        )
        if probability < 0 or probability > 1:
            self.logger.error("Horizontal flip probability must be between 0 and 1")
            exit(1)
        if random() <= probability:
            if self.bbox is not None:
                for i, b in enumerate(self.bbox):
                    self.bbox[i] = (
                        b[0],
                        b[1],
                        self.shape[0] - b[2],
                        b[3],
                        b[4],
                    )

            self.image = flip(self.image, 0)

    def __monochrome__(self):
        probability = float(
            self.configuration["augmentation"]["monochrome_probability"]
        )
        if probability < 0 or probability > 1:
            self.logger.error("Monochrome probability must be between 0 and 1")
            exit(1)
        if random() <= probability:
            self.image = cvtColor(cvtColor(self.image, COLOR_BGR2GRAY), COLOR_GRAY2BGR)

    def __hsv__(self):
        h_probability = float(self.configuration["augmentation"]["hue_probability"])
        s_probability = float(
            self.configuration["augmentation"]["saturation_probability"]
        )
        v_probability = float(
            self.configuration["augmentation"]["brightness_probability"]
        )
        if any(
            [
                probability < 0 or probability > 1
                for probability in [h_probability, s_probability, v_probability]
            ]
        ):
            self.logger.error(
                "Hue, saturation, and brightness probabilities must be between 0 and 1"
            )
            exit(1)
        h_amplitude = float(self.configuration["augmentation"]["hue_amplitude"])
        s_amplitude = float(self.configuration["augmentation"]["saturation_amplitude"])
        v_amplitude = float(self.configuration["augmentation"]["brightness_amplitude"])
        if any([amplitude < 0 for amplitude in [s_amplitude, v_amplitude]]):
            self.logger.error(
                "Saturation and brightness amplitudes must greater than 0"
            )
            exit(1)
        if h_amplitude < 0:
            self.logger.error("Hue amplitude must be greater than 0")
            exit(1)
        hsv = cvtColor(self.image, COLOR_BGR2HSV)
        h, s, v = split(hsv)

        if random() <= h_probability:
            h = clip(h + uniform(-h_amplitude, h_amplitude) * 180, 0, 180).astype(uint8)
        if random() <= s_probability:
            s = clip(s + uniform(-s_amplitude, s_amplitude) * 255, 0, 255).astype(uint8)
        if random() <= v_probability:
            v = clip(v + uniform(-v_amplitude, v_amplitude) * 255, 0, 255).astype(uint8)

        hsv = merge((h, s, v))
        self.image = cvtColor(hsv, COLOR_HSV2BGR)

    def __contrast__(self):
        probability = float(self.configuration["augmentation"]["contrast_probability"])
        if probability < 0 or probability > 1:
            self.logger.error("Contrast probability must be between 0 and 1")
            exit(1)
        amplitude = float(self.configuration["augmentation"]["contrast_amplitude"])
        if amplitude < 0:
            self.logger.error("Contrast amplitude must greater than 0")
            exit(1)
        if random() <= probability:
            contrast = uniform(-amplitude, amplitude) * 255
            f = 131 * (contrast + 127) / (127 * (131 - contrast))
            alpha_c = f
            gamma_c = 127 * (1 - f)

            self.image = addWeighted(self.image, alpha_c, self.image, 0, gamma_c)

    def __sharpness__(self):
        probability = float(self.configuration["augmentation"]["sharpness_probability"])
        if probability < 0 or probability > 1:
            self.logger.error("Sharpness probability must be between 0 and 1")
            exit(1)
        amplitude = float(self.configuration["augmentation"]["sharpness_amplitude"])
        if amplitude < 0:
            self.logger.error("Sharpness amplitude must greater than 0")
            exit(1)
        if random() <= probability:
            sharpness = 1 + uniform(-amplitude, amplitude)
            if sharpness > 1:
                blur = GaussianBlur(self.image, (3, 3), sharpness)
                self.image = addWeighted(self.image, 1.5, blur, -0.5, 0, blur)
            elif sharpness < 1:
                self.image = GaussianBlur(self.image, (3, 3), sharpness)


if __name__ == "__main__":
    configuration = load_configuration("config.yaml")
    augmentation = Augmentation(configuration)

    images = glob("./moons*.jpg")
    bboxes = glob("./moons*.txt")

    augmentation(images=images, bboxes=bboxes, show=True)

    augmentation(
        context=True,
        images=images,
        bboxes=bboxes,
        show=True,
    )
