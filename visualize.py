import numpy as np
from PIL import Image
import os


def plot(file, x_label, y_label, title=''):
    import matplotlib.pyplot as plt
    import math
    results = np.load(file)
    n_epochs = results.shape[0]
    x = np.linspace(1, n_epochs, n_epochs)
    y = results[:, 2]

    plt.plot(x, y)
    plt.title(title)
    new_list = range(math.floor(min(x)) - 1, math.ceil(max(x)) + 2, 2)
    plt.xticks(new_list)

    plt.xlabel(x_label)
    plt.ylabel(y_label)

    plt.savefig(f'plot_{file.split(".")[0]}.png')
    plt.show()


def combine(path_orig, path_sal):
    import cv2 as cv

    names = os.listdir(os.path.join(path_orig, 'images'))

    path_out = os.path.join(path_sal, 'visual')
    if not os.path.isdir(path_out):
        os.makedirs(path_out)

    for name in names:
        img_orig = cv.imread(os.path.join(path_orig, 'images', name))
        img_sal = cv.imread(os.path.join(path_sal, name))
        img_sal[:, :, 0] = 0
        img_sal[:, :, 1] = 0

        out = cv.addWeighted(img_orig, 1.0, img_sal, 1.0, 0.0)
        # cv.imshow('with saliency', out)
        # cv.waitKey(0)
        cv.imwrite(os.path.join(path_out, name), out)

    names = os.listdir(path_out)
    images = []
    for name in names:
        images.append(Image.open(os.path.join(path_out, name)))
    images[0].save(os.path.join(path_out, 'video.gif'), save_all=True, append_images=images[1:])


def stack(pred, gt, start_idx, length, step):
    files = [name for name in os.listdir(pred)[start_idx: start_idx + (step * length): step]]
    images_1 = []
    images_2 = []
    for file in files:
        images_1.append(Image.open(os.path.join(pred, file)))
        images_2.append(Image.open(os.path.join(gt, file)))
    images_stack_1 = np.hstack(images_1)
    images_stack_2 = np.hstack(images_2)
    out = np.vstack(np.asarray([images_stack_1, images_stack_2]))
    out = Image.fromarray(out)
    out.save(os.path.join(pred, 'compare.jpg'))


if __name__ == '__main__':
    # plot('loss_negative.npy', 'Epoka', 'loss')
    # combine('E:/Szkolne/Praca_magisterska/ACLNet-Pytorch/validation/0601',
    #         'E:/Szkolne/Praca_magisterska/projekt/result/negative_model/0601')
    stack('E:/Szkolne/Praca_magisterska/projekt/result/sim_model/0601/visual',
          'E:/Szkolne/Praca_magisterska/projekt/result/sim_model/0601/ground_truth',
          492, 3, 8)
