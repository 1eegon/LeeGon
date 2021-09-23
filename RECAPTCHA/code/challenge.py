from selenium import webdriver
from selenium.common.exceptions import NoSuchElementException
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms
from os import listdir
from os.path import isfile, join
import numpy as np
import cv2
import time
import urllib.request
import torch
import torch.nn as nn
import os
import queue

data_folder_path = './Test'
savepath = "C:/Users/eiden/PycharmProjects/Challenge_/Test/download/"

# Korean
img_dict = {'자전거': 0, '보트': 1, '교각': 2, '버스': 3, '차량': 4, '자동차':4, '굴뚝': 5, '횡단보도': 6, '소화전': 7, '오토바이': 8, '산 또는 언덕': 9,
            '야자수': 10, '주차 요금 정산기': 11, '계단': 12, '택시': 13, '트랙터': 14, '신호등': 15}

# English
# img_dict = {'bicycles': 0, 'boats': 1, 'bridges': 2, 'buses': 3, 'cars': 4, 'vehicles': 4, 'chimneys': 5,
#             'crosswalks': 6, 'hydrants': 7, 'motorcycles': 8, 'mountains': 9, 'palm trees': 10, 'parking meters': 11,
#             'stairs': 12, 'taxis': 13, 'tractors': 14, 'traffic lights': 15}

# def Classify(*sequential, **named):
#     enums = dict(zip(sequential, range(len(sequential))), **named)
#     return type('Enum', (), enums)


# (2) Cropped func : to crop img (3 by 3) or (4 by 4)
def Cropped(kind):
    dataset = [f for f in listdir(savepath) if isfile(join(savepath, f))]
    Images = np.empty(len(dataset), dtype=object)

    count = 10

    if kind == 3:
        for i in range(0, len(dataset)):
            Images[i] = cv2.imread(join(savepath, dataset[i]))

            for x in range(0, 3):
                for y in range(0, 3):
                    cropped = Images[i][0 + 100 * x: 100 + 100 * x, 0 + 100 * y: 100 + 100 * y]
                    cv2.imwrite('Test/test/img/{0}.jpg'.format(i + count), cropped)
                    count = count + 1

            count = count - 1

    elif kind == 4:
        for i in range(0, len(dataset)):
            Images[i] = cv2.imread(join(savepath, dataset[i]))
            PADDED = np.pad((Images[i]), ((1, 1), (1, 1), (0, 0)), 'constant', constant_values=0)

            ''' Cropped data should be the test dataset '''
            for x in range(0, 4):
                for y in range(0, 4):
                    cropped = PADDED[0 + 113 * x: 113 + 113 * x, 0 + 113 * y: 113 + 113 * y]
                    cv2.imwrite('Test/test/img/{0}.jpg'.format(i + count), cropped)
                    count = count + 1

            count = count - 1


# (3) Test images using given training-model

def Test(kind):
    cuda = torch.cuda.is_available()
    device = 'cpu' if not cuda else 'cuda'

    image_amount = 16
    your_model_path = './0901_googlenet_re.pt'

    if kind == 3:
        batch_size_num = 3
        # your_model_path = './train_0819_googlenet_recaptcha_13.pt'

    elif kind == 4:
        # image_amount = 12
        batch_size_num = 4
        # your_model_path = './train_0821_googlenet_recaptcha_12.pt'

    ''' Loaing model '''
    model = models.googlenet(pretrained=True)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, image_amount)
    model.load_state_dict(torch.load(your_model_path, map_location=torch.device('cpu')))
    model.to(device)
    # print('Model loaded')

    # Load evaluation image set
    transform_list = [transforms.ToTensor(), transforms.Resize(224),
                      transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]
    _transforms = transforms.Compose(transform_list)

    image_datasets = {x: datasets.ImageFolder(os.path.join(data_folder_path, x), _transforms) for x in ['test']}
    dataloaders = {
        x: torch.utils.data.DataLoader(image_datasets[x], batch_size=batch_size_num, shuffle=False, num_workers=4) for x
        in ['test']}

    # print('Evaluating...')

    model.eval()

    store_tensor = np.zeros((kind, kind))

    with torch.no_grad():
        for i, (inputs, labels) in enumerate(dataloaders['test']):
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)

            # print("preds", preds)

            # Tensor to Numpy
            preds_np = preds.cpu().numpy()
            store_tensor[i] = preds_np

    # store_tensor = torch.from_numpy(store_tensor)

    return store_tensor

def Func(driver) :
    dir = data_folder_path + "/test/img/"
    test = os.listdir(dir)
    for item in test:
        if item.endswith(".jpg"):
            os.remove(os.path.join(dir, item))

    time.sleep(1)

    frame = driver.find_elements_by_tag_name("iframe")[2]
    driver.switch_to.frame(frame)

    # find topic
    topic = driver.find_element_by_css_selector('#rc-imageselect > div.rc-imageselect-payload > '
                                                'div.rc-imageselect-instructions > div.rc-imageselect-desc-wrapper > '
                                                'div > strong').text
    print(topic)
    print(img_dict[topic])

    imgUrl = driver.find_element_by_xpath(
        "//*[@id='rc-imageselect-target']/table/tbody/tr[1]/td[1]/div/div[1]/img").get_attribute("src")
    urllib.request.urlretrieve(imgUrl, savepath + "payload.jpg")
    kind = int(driver.find_element_by_xpath(
        "//*[@id='rc-imageselect-target']/table/tbody/tr[1]/td[1]/div/div[1]/img").get_attribute("class")[-1])


    ''' 2. 사진 분할하기 '''
    Cropped(kind)
    print(kind)

    return kind, topic

def Clear():
    dir = data_folder_path + "/test/img/"
    test = os.listdir(dir)
    for item in test:
        if item.endswith(".jpg"):
            os.remove(os.path.join(dir, item))

def main():
    ''' 1. 데모사이트에서 이미지 다운받기 '''

    driver = webdriver.Chrome()
    demo_path = "https://www.google.com/recaptcha/api2/demo"

    driver.get(demo_path)

    mainwindow = driver.current_window_handle

    frame = driver.find_element_by_tag_name("iframe")
    driver.switch_to.frame(frame)

    driver.find_element_by_id("recaptcha-anchor").click()

    # checkbox bug
    try:
        driver.switch_to.window(mainwindow)
    except RuntimeError:
        driver.close()

    # driver.switch_to.window(mainwindow)

    # frame changing-term
    time.sleep(1)

    flag = False
    First=None
    Second=None
    Third=None
    Fourth=None

    q=queue.Queue()

    while (flag == False):

        if (First is None) and (Second is None) and (Third is None) and (Fourth is None) :
            kind, topic = Func(driver)
            img_tensor = Test(kind)
        elif First is not None :
            kind, topic = Func(driver)
            img_tensor = Test(kind)
        elif (Second is not None) or (Fourth is not None):
            frame = driver.find_elements_by_tag_name("iframe")[2]
            driver.switch_to.frame(frame)

            driver.find_element_by_xpath("//*[@id='recaptcha-reload-button']").click()

            driver.switch_to.window(mainwindow)

            kind, topic = Func(driver)
            img_tensor = Test(kind)
        elif Third is not None :
            frame = driver.find_elements_by_tag_name("iframe")[2]
            driver.switch_to.frame(frame)

            while q.qsize()>0 :
                dir=data_folder_path+"/test/img/"
                tup=q.get()
                i=tup[0]
                j=tup[1]
                num=10+3*i+j

                imgUrl = driver.find_element_by_xpath(
                    "//*[@id='rc-imageselect-target']/table/tbody/tr[{0}]/td[{1}]/div/div[1]/img".format(i+1, j+1)).get_attribute("src")
                urllib.request.urlretrieve(imgUrl, dir + "{}.jpg".format(num))

                img_tensor = Test(kind)
                if img_dict[topic]==img_tensor[i][j]:
                    img_tensor[i][j] = -1
                    q.put((i, j))
                    driver.find_element_by_xpath('/html/body/div/div/div[2]/div[2]/div/table/tbody/tr[{0}]/td[{1}]'.
                                                 format(i + 1, j + 1)).click()
                    time.sleep(3)

        if Third is None :
            while img_dict[topic] in img_tensor[:][:]:
                for i in range(kind):
                    for j in range(kind):
                        if img_tensor[i][j] == img_dict[topic]:
                            img_tensor[i][j] = -1
                            q.put((i,j))
                            print(img_tensor)
                            driver.find_element_by_xpath('/html/body/div/div/div[2]/div[2]/div/table/tbody/tr[{0}]/td[{1}]'.
                                                         format(i + 1, j + 1)).click()
                            time.sleep(3)

                # 확인 버튼 누르기

        driver.find_element_by_css_selector('#recaptcha-verify-button').click()

        # 다시 시도해 주세요 -> 첨부터 다시
        First = driver.find_element_by_xpath("/html/body/div/div/div[2]/div[3]").get_attribute("tabindex")
        # 해당되는 이미지를 모두 선택하세요 -> 새래고침안되면 그냥 새로고침 / 새로고침되는경우로, 그 사진만 다시 다운
        Second = driver.find_element_by_xpath("/html/body/div/div/div[2]/div[4]").get_attribute("tabindex")
        # 새 이미지도 확인해 보세요 -> 새로고침되는경우로, 그사진만 다시 다운
        Third = driver.find_element_by_xpath("/html/body/div/div/div[2]/div[5]").get_attribute("tabindex")
        # 개체 주변을 선택하거나, 개체가 없으면 새로고침하세요.
        Fourth = driver.find_element_by_xpath("/html/body/div/div/div[2]/div[6]").get_attribute("tabindex")

        if Third is not None and q.qsize()==0:
            Third = None

        # frame switching
        driver.switch_to.window(mainwindow)
        frame = driver.find_element_by_tag_name("iframe")
        driver.switch_to.frame(frame)

                # frame changing-term
        time.sleep(3)

        check_box = driver.find_element_by_css_selector('#recaptcha-anchor').get_attribute('aria-checked')
        print(check_box)

        if check_box == 'true':
            print("DONE.")
            flag=True
            time.sleep(5)
            driver.close()

        else:
            driver.switch_to.window(mainwindow)
            time.sleep(1)
            print("???")
            time.sleep(3)



if __name__ == '__main__':
    Clear()
    main()

    # time.sleep(10)
    # driver.Chrome().close()

