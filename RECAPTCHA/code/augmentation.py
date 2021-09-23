import os
import random
from PIL import Image
from glob import glob


# transposing the given image left and right
def transpose_img_lr(image):
    image = image.transpose(Image.FLIP_LEFT_RIGHT)
    return image


# transposing the given image top and bottom
def transpose_img_tb(image):
    image = image.transpose(Image.FLIP_TOP_BOTTOM)
    return image


# converting the given image to grayscale
def convert_grayscale(image):
    image = image.convert('1')
    return image


# rotating the given image from 1 degree to 360 degree
def rotate(image):
    r = random.randrange(1, 361)
    image = image.rotate(r)
    return image


def main():

    count = 1

    path = "image_files"
    save_path = "augmentation"

    print("Doing augmentation...\n")

    # if there is not a directory named 'augmentation', make it
    if save_path not in os.listdir():
        os.mkdir(save_path)

    img_amount = len(glob(path + "/" + '*.png'))

    for i in range(1, img_amount + 1):

        image = Image.open(path + "/" + "{0}.png".format(i))
        X_dim, Y_dim = image.size

        new_img_name = "{0}.png".format(count)
        count += 1

        image.save(save_path + "/" + new_img_name)
        image.close()

        FILE_LIST = [new_img_name]

        # (1) transpose left and right
        for i in range(len(FILE_LIST)):

            img = Image.open(save_path + "/" + FILE_LIST[i])
            new_conv_img = "{0}.png".format(count)
            count += 1
            img = transpose_img_lr(img)
            img.save(save_path + "/" + new_conv_img)
            img.close()
            FILE_LIST.append(new_conv_img)

        # (2) transpose top and bottom
        for i in range(len(FILE_LIST)):
            img = Image.open(save_path + "/" + FILE_LIST[i])
            new_conv_img = "{0}.png".format(count)
            count += 1
            img = transpose_img_tb(img)
            img.save(save_path + "/" + new_conv_img)
            img.close()
            FILE_LIST.append(new_conv_img)

        (3) make images to grayscale
        for i in range(len(FILE_LIST)):
            img = Image.open(save_path + "/" + FILE_LIST[i])
            new_conv_img = "{0}.png".format(count)
            count += 1
            img = convert_grayscale(img)
            img.save(save_path + "/" + new_conv_img)
            img.close()
            FILE_LIST.append(new_conv_img)

        # (4) rotate images
        for i in range(len(FILE_LIST)):
            img = Image.open(save_path + "/" + FILE_LIST[i])
            new_conv_img = "{0}.png".format(count)
            count += 1
            img = rotate(img)
            # img = img.resize(X_dim, Y_dim)
            img.save(save_path + "/" + new_conv_img)
            img.close()

    print("Finished.")


if __name__ == "__main__":
    main()
