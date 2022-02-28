import os
import itertools
import numpy as np
import matplotlib.pyplot as plt
import mermaid.simple_interface as SI
from torchvision.transforms import ToPILImage
from skimage import color, img_as_float32
from skimage import io
from skimage.transform import rescale, resize
from numpy import asarray
from numpy import pad
from PIL import Image
import time


def add_square_padding(image):
    additional_padding = 16
    if image.shape[0] > image.shape[1]:
        difference = image.shape[0] - image.shape[1]
        padding = (difference // 2) + additional_padding
        if difference % 2 == 1:
            image = pad(image, [(additional_padding, additional_padding), (padding + 1, padding)],
                        mode='constant',
                        constant_values=image[0][0])
        else:
            image = pad(image, [(additional_padding, additional_padding), (padding, padding)],
                        mode='constant',
                        constant_values=image[0][0])

    else:
        difference = image.shape[1] - image.shape[0]
        padding = (difference // 2) + additional_padding
        if difference % 2 == 1:
            image = pad(image, [(padding + 1, padding), (additional_padding, additional_padding)],
                        mode='constant',
                        constant_values=image[0][0])
        else:
            image = pad(image, [(padding, padding), (additional_padding, additional_padding)],
                        mode='constant',
                        constant_values=image[0][0])
    return image


def reescale_images(image1, image2):

    if image1.shape != image2.shape:
        # Width
        width_ratio = image1.shape[1] / image2.shape[1]
        if width_ratio < 1:
            width_ratio = 1 / width_ratio
            image1 = rescale(image1, width_ratio)
        else:
            image2 = rescale(image2, width_ratio)

        # Height
        height_difference = image1.shape[0] - image2.shape[0]
        if height_difference < 0:
            height_difference = - height_difference
            padding = int(height_difference / 2)
            image1 = pad(image1, [(padding, padding), (0, 0)],
                         mode='constant',
                         constant_values=image1[0][0])

        else:
            padding = int(height_difference / 2)
            image2 = pad(image2, [(padding, padding), (0, 0)],
                         mode='constant',
                         constant_values=image2[0][0])

        image1 = add_square_padding(image1)
        image2 = add_square_padding(image2)

    return image1, image2


def process_images(source_image_path, dest_image_path):

    image1 = img_as_float32(color.rgb2gray(io.imread(source_image_path)))
    image2 = img_as_float32(color.rgb2gray(io.imread(dest_image_path)))

    image1_array, image2_array = reescale_images(image1, image2)

    image1_array = image1_array.reshape([1, 1] + list(image1_array.shape))
    image2_array = image2_array.reshape([1, 1] + list(image2_array.shape))
    sz = np.array(image1_array.shape)
    spacing = 1. / (sz[2::] - 1)

    return image1_array, image2_array, spacing


def process_source_target_images(source_image_path, dest_image_path):
    image1 = img_as_float32(color.rgb2gray(io.imread(source_image_path)))
    image2 = img_as_float32(color.rgb2gray(io.imread(dest_image_path)))

    image1_array, image2_array = reescale_images(image1, image2)

    return image1_array, image2_array, spacing


def save_warped_image(warped_image, warped_image_path):
    img = warped_image[0][0]
    img = ToPILImage()(img.cpu())  # If done with GPU
    img.save(warped_image_path)


def save_st_image(image, image_path):
    formatted = (image * 255 / np.max(image)).astype('uint8')
    img = Image.fromarray(formatted)
    img.save(image_path)


def show_registration_statistics(history):
    plt.clf()
    e_p, = plt.plot(history['energy'], label='energy')
    s_p, = plt.plot(history['similarity_energy'], label='similarity_energy')
    r_p, = plt.plot(history['regularization_energy'],
                    label='regularization_energy')
    plt.legend(handles=[e_p, s_p, r_p])
    plt.show()


def register_images(image1_array, image2_array, spacing):
    si = SI.RegisterImagePair()
    si.register_images(image1_array, image2_array, spacing,
                       model_name='svf_scalar_momentum_map',
                       nr_of_iterations=50,
                       use_multi_scale=False,
                       visualize_step=None,
                       optimizer_name='sgd',
                       learning_rate=0.001,
                       rel_ftol=1e-7,
                       json_config_out_filename=('test2d_tst.json',
                                                 'test2d_tst_with_comments.json'),
                       params='test2d_tst.json')

    warped_image = si.get_warped_image()
    history = si.get_history()
    # show_registration_statistics(history)
    return warped_image, history


def main():

    orig_path = "../datasets/Diatoms/dataset"
    out_path = "../datasets/Diatoms/dataset_svf"
    parts = ["train", "val", "test"]

    classes = os.listdir(os.path.join(orig_path, parts[0]))

    if not os.path.exists(out_path):
        os.makedirs(out_path)

    for part in parts:
        print('\n', part)
        part_path = os.path.join(orig_path, part)
        os.makedirs(part_path.replace(orig_path, out_path), exist_ok=True)
        for clase in classes:
            print(clase)
            class_path = os.path.join(part_path, clase)
            os.makedirs(class_path.replace(orig_path, out_path), exist_ok=True)
            images = os.listdir(class_path)
            if part in ["val", "test"]:
                for img in images:
                    img_path = os.path.join(class_path, img)
                    shutil.copy(img_path, img_path.replace(
                        orig_path, out_path))
            elif part == "train":
                candidates = []
                for img in images:
                    img_path = os.path.join(class_path, img)
                    candidates.append(img_path)
                # print(candidates)
                combis = list(itertools.combinations(candidates, 2))
                # print(combis)
                for c in combis:
                    source_image_path = c[0]
                    dest_image_path = c[1]

                    try:
                        image1, image2, spacing = process_images(source_image_path, dest_image_path)
                        warped_image, history = register_images(image1, image2, spacing)

                        # The format of the warped image is: 
                        # sourcepath_targetnumber_EnergyIntegerPart-EnergyDecimals_Niteration.jpg
                        warped_image_path = source_image_path.replace(orig_path, out_path)
                        warped_image_path = warped_image_path.replace(".png", dest_image_path[-8:-4] + "_" +
                                                    str(round(history['energy'][-1], 2)).replace('.', '-') + str(history['iter'][-1]) + "it" + ".png")
                        print(warped_image_path)
                        save_warped_image(warped_image, warped_image_path)
                    except Exception as e:
                        print('Exception while processing images:', e)
                        print(source_image_path)
                        print(dest_image_path)
                    exit(0)

if __name__ == "__main__":
    main()
