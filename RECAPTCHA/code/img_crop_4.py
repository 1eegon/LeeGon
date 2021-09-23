from os import listdir
from os.path import isfile, join
import numpy as np
import cv2

''' data01 → PATH = 'data/data01'   |   data02 → PATH = 'data/data02' 
    data01 : 300×300 px             |   data02 : 450×450 px              '''

PATH = 'img'
dataset = [f for f in listdir(PATH) if isfile(join(PATH, f))]
Images = np.empty(len(dataset), dtype=object)

count = 1

print(Images.shape)

for i in range(0, len(dataset)):
    Images[i] = cv2.imread(join(PATH, dataset[i]))
    PADDED = np.pad((Images[i]), ((1, 1), (1, 1), (0, 0)), 'constant', constant_values=0)

    '''Padding checking'''

    # cv2.imshow("Padding", PADDED)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    ''' Checking whether if the image file opens or not '''
    # cv2.imshow("{0}".format(i), Images[i])
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    ''' Cropped data should be the test dataset '''
    for x in range(0, 4):
        for y in range(0, 4):
            cropped = PADDED[0 + 113 * x : 113 + 113 * x, 0 + 113 * y : 113 + 113 * y]
            cv2.imwrite('cropped/{0}.jpg'.format(i + count), cropped)
            count = count + 1

    count = count - 1
