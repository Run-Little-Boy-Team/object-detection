import numpy as np
import cv2
import os
import time
import yaml
from random import seed, random, uniform, randint
from utils import init_logger
from typing import Tuple, List, Union
from glob import glob


class Augmentation:
    def __init__(
        self,
        configuration_path: str = None,
        name: str = "Augmentation",
        log_level: str = "INFO",
    ):
        self.name = name
        self.logger = init_logger(name, log_level)
        seed()

        t_0 = time.time()
        self.logger.info("Loading configuration...")
        if configuration_path is None:
            self.logger.error("Configuration path must be provided")
            exit(1)

        with open(configuration_path, "r") as file:
            self.configuration = yaml.safe_load(file)

        t_1 = time.time()
        self.logger.info(f"Configuration loaded in {(t_1 - t_0) * 1000:.2f} ms")

    def run(
        self,
        images: Union[List[cv2.typing.MatLike], List[str]] = None,
        bboxes: Union[List[Tuple[int]], List[str]] = None,
        augmentation_per_image: int = 1,
        resize: bool = True,
        context: bool = False,
        backgrounds: Union[List[cv2.typing.MatLike], List[str], None] = None,
        images_background_color: Tuple[int] = (0, 255, 0),
        show: bool = False,
        save: bool = False,
        save_path: str = "./outputs",
        save_name: str = "augmented",
    ) -> List[Tuple[cv2.typing.MatLike, Tuple[int]]]:
        start = time.time()

        if images is None:
            self.logger.error("Images must be provided")
            exit(1)
        if bboxes is None:
            self.logger.error("Bboxes must be provided")
            exit(1)
        if len(images) != len(bboxes):
            self.logger.error("Number of images and bboxes must be the same")
            exit(1)
        if augmentation_per_image < 1:
            self.logger.error("Augmentation per image must be greater than 0")
            exit(1)

        if all(isinstance(image, str) for image in images):
            self.logger.info("Loading images...")
            t_0 = time.time()
            images = [cv2.imread(image) for image in images]
            t_1 = time.time()
            self.logger.info(f"Images loaded in {(t_1 - t_0) * 1000:.2f} ms")

        if all(isinstance(bbox, str) for bbox in bboxes):
            self.logger.info("Loading bboxes...")
            t_0 = time.time()
            bboxes = [
                [float(i) for i in open(bbox, "r").readline().split(" ")]
                for bbox in bboxes
            ]
            for i in range(len(bboxes)):
                bboxes[i] = (
                    int(bboxes[i][0] * images[i].shape[1]),
                    int(bboxes[i][1] * images[i].shape[0]),
                    int(bboxes[i][2] * images[i].shape[1]),
                    int(bboxes[i][3] * images[i].shape[0]),
                )
            t_1 = time.time()
            self.logger.info(f"Bboxes loaded in {(t_1 - t_0) * 1000:.2f} ms")

        if resize:
            self.logger.info("Resizing images...")
            t_0 = time.time()
            size = int(self.configuration["size"])
            for i in range(len(images)):
                ratio = size / max(images[i].shape[:2])
                images[i] = cv2.resize(images[i], (size, size))
                bboxes[i] = [int(b * ratio) for b in bboxes[i]]
            t_1 = time.time()
            self.logger.info(f"Images resized in {(t_1 - t_0) * 1000:.2f} ms")

        outputs = []
        if context:
            if backgrounds is None or len(backgrounds) == 0:
                self.logger.error("Backgrounds must be provided")
                exit(1)

            self.logger.info(
                f"Running in context mode for {len(images)} image{'s'if len(images) > 1 else ''} and {augmentation_per_image} augmentation{'s'if augmentation_per_image > 1 else ''} per image..."
            )

            for i in range(len(images)):
                image = images[i]
                bbox = bboxes[i]
                mask = np.all(image == images_background_color[::-1], axis=-1)
                for _ in range(augmentation_per_image):
                    index = randint(0, len(backgrounds) - 1)
                    background = backgrounds[index]
                    if isinstance(background, str):
                        background = cv2.imread(background)
                    background = cv2.resize(background, (image.shape[:2][::-1]))
                    image[mask] = background[mask]
                    output = [image.copy(), bbox]
                    outputs.append(output)

        else:
            self.logger.info(
                f"Running for {len(images)} image{'s'if len(images) > 1 else ''} and {augmentation_per_image} augmentation{'s'if augmentation_per_image > 1 else ''} per image..."
            )

            for i in range(len(images)):
                for _ in range(augmentation_per_image):
                    self.image = images[i].copy()
                    self.shape = self.image.shape
                    self.bbox = bboxes[i]

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

                    output = [self.image, self.bbox]
                    outputs.append(output)

        end = time.time()
        self.logger.info(f"Finished in {(end - start) * 1000:.2f} ms")

        if save:
            self.logger.info("Saving images...")
            t_0 = time.time()

            os.makedirs(save_path, exist_ok=True)
            for i, output in enumerate(outputs):
                image = output[0]
                bbox = output[1]
                bbox = (
                    bbox[0] / image.shape[1],
                    bbox[1] / image.shape[0],
                    bbox[2] / image.shape[1],
                    bbox[3] / image.shape[0],
                )
                cv2.imwrite(f"{save_path}/{save_name}_{i}.jpg", image)
                with open(f"{save_path}/{save_name}_{i}.txt", "w") as file:
                    file.write(" ".join(map(str, bbox)))

            t_1 = time.time()
            self.logger.info(f"Images saved in {(t_1 - t_0) * 1000:.2f} ms")

        if show:
            self.logger.info(
                "Showing augmented images, press any key to continue or ESC to exit..."
            )
            for i, output in enumerate(outputs):
                self.logger.info(f"{i+1}/{len(outputs)}")
                self.image = output[0]
                self.bbox = output[1]
                self.__show__()
                key = cv2.waitKey(0)
                if key == 27:
                    break
            cv2.destroyAllWindows()

        return outputs

    def __show__(self):
        draw = self.image.copy()
        if self.bbox is not None:
            draw = cv2.rectangle(
                draw,
                (self.bbox),
                (0, 255, 0),
                2,
            )
        cv2.imshow(self.name, draw)

    def __translate__(self):
        probability = float(self.configuration["translation_probability"])
        if probability < 0 or probability > 1:
            self.logger.error("Translation probability must be between 0 and 1")
            exit(1)
        amplitude = float(self.configuration["translation_amplitude"])
        if amplitude < 0:
            self.logger.error("Translation amplitude must be greater than 0")
            exit(1)
        if random() <= probability:
            x = uniform(-amplitude, amplitude) * self.shape[1]
            y = uniform(-amplitude, amplitude) * self.shape[0]

            translation_matrix = np.float32([[1, 0, x], [0, 1, y]])

            self.bbox = (
                int(self.bbox[0] + x),
                int(self.bbox[1] + y),
                self.bbox[2],
                self.bbox[3],
            )

            self.image = cv2.warpAffine(
                self.image, translation_matrix, (self.shape[1], self.shape[0])
            )

    def __rotate__(self):
        probability = float(self.configuration["rotation_probability"])
        if probability < 0 or probability > 1:
            self.logger.error("Rotation probability must be between 0 and 1")
            exit(1)
        amplitude = float(self.configuration["rotation_amplitude"])
        if abs(amplitude) > 90 or amplitude < 0:
            self.logger.error("Rotation amplitude must be between 0 and 90")
            exit(1)
        if random() <= probability:
            angle = uniform(-amplitude, amplitude)

            center = (int(self.shape[1]) // 2, int(self.shape[0]) // 2)

            rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)

            x, y, width, height = self.bbox
            x_min = x
            y_min = y
            x_max = x + width
            y_max = y + height

            points = np.array(
                [[x_min, y_min], [x_max, y_min], [x_min, y_max], [x_max, y_max]]
            ).reshape(4, 1, 2)
            rotated_points = (
                cv2.transform(points, rotation_matrix).reshape(4, 2).astype(int)
            )

            x_min = int(np.min(rotated_points[:, 0]))
            x_max = int(np.max(rotated_points[:, 0]))
            y_min = int(np.min(rotated_points[:, 1]))
            y_max = int(np.max(rotated_points[:, 1]))

            self.bbox = (x_min, y_min, x_max - x_min, y_max - y_min)

            self.image = cv2.warpAffine(
                self.image, rotation_matrix, (int(self.shape[1]), int(self.shape[0]))
            )

    def __scale__(self):
        probability = float(self.configuration["scaling_probability"])
        if probability < 0 or probability > 1:
            self.logger.error("Scaling probability must be between 0 and 1")
            exit(1)
        amplitude = float(self.configuration["scaling_amplitude"])
        if amplitude < 0:
            self.logger.error("Scaling amplitude must be greater than 0")
            exit(1)
        if random() <= probability:
            scale = 1 + uniform(-amplitude, amplitude)

            center = (int(self.shape[1]) // 2, int(self.shape[0]) // 2)

            scale_matrix = cv2.getRotationMatrix2D(center, 0, scale)

            points = np.array(
                [
                    [self.bbox[0], self.bbox[1]],
                    [self.bbox[0] + self.bbox[2], self.bbox[1] + self.bbox[3]],
                ]
            ).reshape(2, 1, 2)
            scaled_points = cv2.transform(points, scale_matrix).reshape(4).astype(int)
            self.bbox = (
                scaled_points[0],
                scaled_points[1],
                scaled_points[2] - scaled_points[0],
                scaled_points[3] - scaled_points[1],
            )

            self.image = cv2.warpAffine(
                self.image, scale_matrix, (int(self.shape[1]), int(self.shape[0]))
            )

    def __stretch__(self):
        probability = float(self.configuration["stretching_probability"])
        if probability < 0 or probability > 1:
            self.logger.error("Stretching probability must be between 0 and 1")
            exit(1)
        amplitude = float(self.configuration["stretching_amplitude"])
        if amplitude < 0:
            self.logger.error("Streching amplitude must be greater than 0")
            exit(1)
        if random() <= probability:
            x = 1 + uniform(-amplitude, amplitude)
            y = 1 + uniform(-amplitude, amplitude)

            stretch_matrix = np.float32([[x, 0, 0], [0, y, 0]])

            stretched_points = (
                int(self.bbox[0] * x),
                int(self.bbox[1] * y),
                int((self.bbox[0] + self.bbox[2]) * x),
                int((self.bbox[1] + self.bbox[3]) * y),
            )
            self.bbox = (
                stretched_points[0],
                stretched_points[1],
                stretched_points[2] - stretched_points[0],
                stretched_points[3] - stretched_points[1],
            )

            stretched = cv2.warpAffine(
                self.image, stretch_matrix, (self.shape[1], self.shape[0])
            )
            self.image = stretched

    def __shear__(self):
        probability = float(self.configuration["shearing_probability"])
        if probability < 0 or probability > 1:
            self.logger.error("Shearing probability must be between 0 and 1")
            exit(1)
        amplitude = float(self.configuration["shearing_amplitude"])
        if amplitude < 0:
            self.logger.error("Shear amplitude must be greater than 0")
            exit(1)
        if random() <= probability:
            x = uniform(-amplitude, amplitude)
            y = uniform(-amplitude, amplitude)

            shear_matrix = np.float32([[1, x, 0], [y, 1, 0]])

            x_min, y_min, width, height = self.bbox
            x_max = x_min + width
            y_max = y_min + height

            points = np.array(
                [[x_min, y_min], [x_max, y_min], [x_min, y_max], [x_max, y_max]]
            ).reshape(4, 1, 2)
            sheared_points = cv2.transform(points, shear_matrix).reshape(4, 2)

            x_min = int(np.min(sheared_points[:, 0]))
            y_min = int(np.min(sheared_points[:, 1]))
            x_max = int(np.max(sheared_points[:, 0]))
            y_max = int(np.max(sheared_points[:, 1]))

            self.bbox = (x_min, y_min, x_max - x_min, y_max - y_min)

            self.image = cv2.warpAffine(
                self.image, shear_matrix, (int(self.shape[1]), int(self.shape[0]))
            )

    def __vertical_flip__(self):
        probability = float(self.configuration["vertical_flip_probability"])
        if probability < 0 or probability > 1:
            self.logger.error("Vertical flip probability must be between 0 and 1")
            exit(1)
        if random() <= probability:
            self.bbox = (
                self.shape[1] - self.bbox[0] - self.bbox[2],
                self.bbox[1],
                self.bbox[2],
                self.bbox[3],
            )

            self.image = cv2.flip(self.image, 1)

    def __horizontal_flip__(self):
        probability = float(self.configuration["horizontal_flip_probability"])
        if probability < 0 or probability > 1:
            self.logger.error("Horizontal flip probability must be between 0 and 1")
            exit(1)
        if random() <= probability:
            self.bbox = (
                self.bbox[0],
                self.shape[0] - self.bbox[1] - self.bbox[3],
                self.bbox[2],
                self.bbox[3],
            )

            self.image = cv2.flip(self.image, 0)

    def __monochrome__(self):
        probability = float(self.configuration["monochrome_probability"])
        if probability < 0 or probability > 1:
            self.logger.error("Monochrome probability must be between 0 and 1")
            exit(1)
        if random() <= probability:
            self.image = cv2.cvtColor(
                cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY), cv2.COLOR_GRAY2BGR
            )

    def __hsv__(self):
        h_probability = float(self.configuration["hue_probability"])
        s_probability = float(self.configuration["saturation_probability"])
        v_probability = float(self.configuration["brightness_probability"])
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
        h_amplitude = float(self.configuration["hue_amplitude"])
        s_amplitude = float(self.configuration["saturation_amplitude"])
        v_amplitude = float(self.configuration["brightness_amplitude"])
        if any([amplitude < 0 for amplitude in [s_amplitude, v_amplitude]]):
            self.logger.error(
                "Saturation and brightness amplitudes must greater than 0"
            )
            exit(1)
        if h_amplitude < 0:
            self.logger.error("Hue amplitude must be greater than 0")
            exit(1)
        hsv = cv2.cvtColor(self.image, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)

        if random() <= h_probability:
            h = np.clip(h + uniform(-h_amplitude, h_amplitude) * 180, 0, 180).astype(
                np.uint8
            )
        if random() <= s_probability:
            s = np.clip(s + uniform(-s_amplitude, s_amplitude) * 255, 0, 255).astype(
                np.uint8
            )
        if random() <= v_probability:
            v = np.clip(v + uniform(-v_amplitude, v_amplitude) * 255, 0, 255).astype(
                np.uint8
            )

        hsv = cv2.merge((h, s, v))
        self.image = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    def __contrast__(self):
        probability = float(self.configuration["contrast_probability"])
        if probability < 0 or probability > 1:
            self.logger.error("Contrast probability must be between 0 and 1")
            exit(1)
        amplitude = float(self.configuration["contrast_amplitude"])
        if amplitude < 0:
            self.logger.error("Contrast amplitude must greater than 0")
            exit(1)
        if random() <= probability:
            contrast = uniform(-amplitude, amplitude) * 255
            f = 131 * (contrast + 127) / (127 * (131 - contrast))
            alpha_c = f
            gamma_c = 127 * (1 - f)

            self.image = cv2.addWeighted(self.image, alpha_c, self.image, 0, gamma_c)

    def __sharpness__(self):
        probability = float(self.configuration["sharpness_probability"])
        if probability < 0 or probability > 1:
            self.logger.error("Sharpness probability must be between 0 and 1")
            exit(1)
        amplitude = float(self.configuration["sharpness_amplitude"])
        if amplitude < 0:
            self.logger.error("Sharpness amplitude must greater than 0")
            exit(1)
        if random() <= probability:
            sharpness = 1 + uniform(-amplitude, amplitude)
            if sharpness > 1:
                blur = cv2.GaussianBlur(self.image, (3, 3), sharpness)
                self.image = cv2.addWeighted(self.image, 1.5, blur, -0.5, 0, blur)
            elif sharpness < 1:
                self.image = cv2.GaussianBlur(self.image, (3, 3), sharpness)


if __name__ == "__main__":
    augmentation = Augmentation("./augmentation_config.yaml")

    images = ["./moon.jpg"]
    bboxes = ["./moon.txt"]
    backgrounds = glob("./backgrounds/*.jpg")

    augmentation.run(
        images=images, 
        bboxes=bboxes, 
        augmentation_per_image=100, 
        show=True
    )

    augmentation.run(
        context=True,
        images=images,
        bboxes=bboxes,
        augmentation_per_image=100,
        backgrounds=backgrounds,
        images_background_color=(0, 0, 0),
        show=True,
    )
