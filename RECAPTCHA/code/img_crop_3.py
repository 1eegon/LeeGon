from os import listdir
from os.path import isfile, join
import numpy as np
import cv2

''' data01 → PATH = 'data/data01'   |   data02 → PATH = 'data/data02' 
    data01 : 300×300 px             |   data02 : 450×450 px              '''

PATH = 'data/data01'
dataset = [f for f in listdir(PATH) if isfile(join(PATH, f))]
Images = np.empty(len(dataset), dtype=object)

count = 1

for i in range(0, len(dataset)):
    Images[i] = cv2.imread(join(PATH, dataset[i]))

    ''' Checking whether if the image file opens or not '''
    # cv2.imshow("{0}".format(i), Images[i])
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    ''' Cropped data should be the test dataset '''
    for x in range(0, 3):
        for y in range(0, 3):
            cropped = Images[i][0 + 100 * x : 100 + 100 * x, 0 + 100 * y : 100 + 100 * y]
            cv2.imwrite('cropped/data01/{0}.jpg'.format(i + count), cropped)
            count = count + 1

    count = count - 1
