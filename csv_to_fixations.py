import pandas as pd
import math
import os
import numpy as np
import cv2 as cv


dist = 550
num_frames = len(os.listdir('../GazeBase/Frames/VID_1/images'))


def dva_to_coord(dva, pixel_size):
    rad = (dva * math.pi) / 180
    if rad == 0:
        return 0
    return round((dist * math.tan(rad)) / pixel_size)


def read_csv(round_path):
    dirs = [x for x in os.listdir(round_path) if not x.endswith('.zip')]
    x_df, y_df = [], []
    fixation_images = np.zeros((num_frames, 1050, 1680), np.int16)
    frames_ids = [x[-8:-4] for x in os.listdir('../GazeBase/Frames/VID_1/images')]

    x_pixel_size = 474 / 1680
    y_pixel_size = 297 / 1050
    for d in dirs:
        subject = d[-4:]
        sessions = (1, 2)
        print(f'Reading subject {subject}...')
        for s_num in sessions:
            session = f'S{s_num}'
            task = f'{session}_Video_1'
            csv = f'S_{subject}_S{s_num}_VD1.csv'

            df = pd.read_csv(os.path.join(round_path, d, session, task, csv))
            x_dva = df['x']
            y_dva = df['y']
            x_dva = x_dva.dropna()
            y_dva = y_dva.dropna()
            x = x_dva.apply(lambda row: dva_to_coord(row, x_pixel_size))
            y = y_dva.apply(lambda row: dva_to_coord(row, y_pixel_size))
            x_df.append(x)
            y_df.append(y)
            print(f' dataframes read: {len(x_df)}')

    for i in range(num_frames):
        img = fixation_images[i]
        print(f'Writing fixations for frame {frames_ids[i]}...')
        for x, y in zip(x_df, y_df):
            num_fixations = x.size
            idx = round((num_fixations * i) / num_frames)
            if idx >= num_fixations:
                idx = num_fixations - 1
            if idx not in x.index or idx not in y.index:
                continue
            x_coord = x[idx] + 840
            y_coord = y[idx] + 525
            if x_coord >= 1680 or x_coord < 0 or y_coord >= 1050 or y_coord < 0:
                continue
            img[y_coord, x_coord] = 255

        name = f'{frames_ids[i]}.jpg'
        r = os.path.split(round_path)[-1]
        if not os.path.isdir(os.path.join('../GazeBase', 'Fixations', r)):
            os.makedirs(os.path.join('../GazeBase', 'Fixations', r))
        cv.imwrite(os.path.join('../GazeBase', 'Fixations', r, name), fixation_images[i])
        print(f' Saved image {name}')

    return os.path.join('../GazeBase/Fixations', os.path.split(round_path)[-1])


def blur(round_path):
    for img in os.listdir(round_path):
        name = img.split('/')[-1]
        print(f'Blurring image {name}...')
        im = cv.imread(os.path.join(round_path, img), 0)
        im = cv.GaussianBlur(im, (9, 9), 0)
        _, im = cv.threshold(im, 10, 255, cv.THRESH_TOZERO)
        im = cv.GaussianBlur(im, (5, 5), 0)
        # _, im = cv.threshold(im, 20, 255, cv.THRESH_BINARY)
        nb_blobs, im_with_separated_blobs, stats, _ = cv.connectedComponentsWithStats(im)
        sizes = stats[:, -1]
        sizes = sizes[1:]
        nb_blobs -= 1
        min_size = 150
        im_result = np.zeros((im.shape))
        for blob in range(nb_blobs):
            if sizes[blob] >= min_size:
                im_result[im_with_separated_blobs == blob + 1] = 255
        im_result = cv.GaussianBlur(im_result, (99, 99), 0)
        cv.imwrite(os.path.join(round_path, name), im_result)
        print(f' Saved image {name}')


# rounds = [r for r in os.listdir('../GazeBase') if r.startswith('Round')]
rounds = ['Round_1']
for r in rounds:
    print(f'Processing {r}...\n')
    r_path = os.path.join('../GazeBase', r)
    result_path = read_csv(r_path)
    # blur(result_path)
