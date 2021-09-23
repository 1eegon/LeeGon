import torch
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms
import csv
import torch.nn as nn
from torch.utils.data import Dataset
import os
from typing import List
from PIL import Image

# PATH
your_model_path = './0830_googlenet_both.pt'

# image class amount (Generally 16 : From Bicycle to Traffic light)
image_amount = 16

# batch size
batch_size_num = 4

normalize = False


def main():

    cuda = torch.cuda.is_available()
    device = 'cpu' if not cuda else 'cuda'

    '''     Loading model   '''

    '''    googlenet   '''

    model=models.googlenet(pretrained=True)
    num_ftrs=model.fc.in_features
    model.fc=nn.Linear(num_ftrs, image_amount)

    '''                 '''

    '''     inception v3     '''
    # model = models.inception_v3(pretrained=True)
    # model.aux_logits = False
    #
    # num_ftrs = model.AuxLogits.fc.in_features
    # model.AuxLogits.fc = nn.Linear(num_ftrs, image_amount)
    #
    # num_ftrs = model.fc.in_features
    # model.fc = nn.Linear(num_ftrs, image_amount)
    '''                     '''

    '''     resnet34  '''
    # model = models.resnet34(pretrained=True)
    # num_ftrs = model.fc.in_features
    # model.fc = nn.Linear(num_ftrs, image_amount)
    '''             '''

    model.load_state_dict(torch.load(your_model_path))
    model.to(device)
    print('Model loaded')

    '''     Loading evaluation dataset      '''
    transform_list = [transforms.ToTensor()]

    # if normalization is applied in your training, you can utilize the codes below.
    # transform_list.append(transforms.Resize(224))
    transform_list.append(transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]))

    _transforms = transforms.Compose(transform_list)

    data_folder_path = './data'

    image_datasets = {x: datasets.ImageFolder(os.path.join(data_folder_path, x),
                                              _transforms)
                      for x in ['test']}

    dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=batch_size_num,
                                                  shuffle=True, num_workers=4)
                   for x in ['test']}

    class_names = image_datasets['test'].classes

    ''' evaluation '''
    print('Evaluating...')

    correct_pred = {classname: 0 for classname in class_names}
    total_pred = {classname: 0 for classname in class_names}

    model.eval()

    correct = 0
    total = 0
    # 학습 중이 아니므로, 출력에 대한 변화도를 계산할 필요가 없습니다
    with torch.no_grad():
        for i, (inputs, labels) in enumerate(dataloaders['test']):
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)

            total += labels.size(0)
            correct += (preds == labels).sum().item()

    print('Accuracy of the network on the {:} test images: {:.1f} %'.format(total, 100 * correct / total))

    # 변화도는 여전히 필요하지 않습니다
    with torch.no_grad():
        for i, (inputs, labels) in enumerate(dataloaders['test']):
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)

            # 각 분류별로 올바른 예측 수를 모읍니다
            for label, prediction in zip(labels, preds):
                if label == prediction:
                    correct_pred[class_names[label]] += 1
                total_pred[class_names[label]] += 1

    # 각 분류별 정확도(accuracy)를 출력합니다
    for classname, correct_count in correct_pred.items():
        accuracy = 100 * float(correct_count) / total_pred[classname]
        print("Accuracy for class {:5s} is: {:.1f} %".format(classname, accuracy))

    print('Done!')
    # print('Results saved at : ', os.path.join(os.getcwd(), filename))


if __name__ == '__main__':
    main()
