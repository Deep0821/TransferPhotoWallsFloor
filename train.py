import torch
import torch.nn as nn
import segmentation_models_pytorch as smp
import pickle
import numpy as np
import os
import cv2
import pandas as pd
import time
import scipy.io as io
import torchvision.transforms as T
from torch.utils.data import Dataset
from PIL import Image
from datetime import datetime
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms as transforms
# from efficientnet_pytorch import EfficientNet
import torchvision.models as models
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt
torch.cuda.empty_cache()


torch.manual_seed(0)
device = torch.device("cuda")


def display_label(label, label_pred):
    label, label_pred = label.detach().cpu().numpy(), label_pred.detach().cpu().numpy()
    print(label.shape)
    print(label_pred.shape)
    new_label = np.argmax(label[0], axis = 0)
    new_image = [new_label for _ in range(3)]
    new_image = np.transpose(new_image, (1, 2, 0))
    new_label_pred = np.argmax(label_pred[0], axis=0)
    new_image_pred = [new_label_pred for _ in range(3)]
    new_image_pred = np.transpose(new_image_pred, (1, 2, 0))
    print(new_image_pred.shape)
    print(np.concatenate([new_image*20, new_image_pred*20], axis=1).shape)
    cv2.imshow("", np.concatenate([(new_image*20).astype(np.uint8), (new_image_pred*20).astype(np.uint8)], axis=1))
    cv2.waitKey(0)


class CustomImageDataset(Dataset):
    def __init__(self, image_folder_dir, label_folder_dir):
        img_paths = []
        labels = []
        for img_path in os.listdir(label_folder_dir):
            full_path = image_folder_dir + "/" + img_path[:-3] + "jpg"
            label_full_path = label_folder_dir + "/" + img_path
            img_paths.append(full_path)
            labels.append(label_full_path)

        self.img_paths = img_paths
        self.labels = labels


    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_path, label = self.img_paths[idx], self.labels[idx]
        # print(label)
        # print(img_path, label)

        image = Image.open(img_path).convert("RGB")
        image = T.PILToTensor()(image)
        image = T.Resize((224, 224))(image)
        image = image.float()
        image = image / 255
        img_shape = image.shape[1:]

        mat = io.loadmat(label)
        img_net = np.array([mat["layout"] for j in range(3)])
        img_net = np.transpose(img_net, (1, 2, 0))
        # img_net = (img_net - np.mean(img_net)) * 255 / (np.max(img_net) - np.min(img_net))
        img_net = img_net.astype(np.uint8)
        label_image = Image.fromarray(img_net)

        label_image = T.Resize(img_shape)(label_image)
        label_image = T.PILToTensor()(label_image)

        # label_image = label_image.float()
        label_image = np.array(label_image)

        label_image -= 1

        label_image = label_image[0]
        new_image = np.array([(label_image == 0) * 1])

        for j in range(1, 5):
            new_image = np.concatenate([new_image, [(label_image == j) * 1]], axis=0)

        # label_image = cv2.cvtColor(label_image, cv2.COLOR_RGB2GRAY)

        #
        # cv2.imshow("", label_image)
        # cv2.waitKey(0)
        # assert 0
        # image = T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))(image)
        # print(image.shape)
        # print(new_image.shape)
        return image, new_image.astype(np.float32), img_path


model = smp.PSPNet(
    encoder_name="resnet34",        # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
    encoder_weights="imagenet",     # use `imagenet` pre-trained weights for encoder initialization
    in_channels=3,                  # model input channels (1 for gray-scale images, 3 for RGB, etc.)
    classes=5,                      # model output channels (number of classes in your dataset)
)

# model.load_state_dict(torch.load("model_weights_7_pspnet"))
# model.to(device)


pics = CustomImageDataset(r"C:\Users\USER\Desktop\lsun_dataset_c\images", \
       r'C:\Users\USER\Desktop\lsun_dataset_c\masks_mat')
n_epochs = 100
batch_size_train = 4
batch_size_test = 1
learning_rate = 0.0005
num_classes = 2

train_size = int(0.8 * len(pics))
test_size = len(pics) - train_size
train_dataset, test_dataset = torch.utils.data.random_split(pics, [train_size, test_size])
classes = ('wood', 'carpet')
train_dataset = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size_train, shuffle=True, drop_last=True)
test_dataset = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size_test, drop_last=True)
# model = EfficientNet.from_pretrained("efficientnet-b3")
# model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet152', pretrained=True)
# for layer in model.parameters():
#     layer.requires_grad = False


# model.load_state_dict(torch.load("model_weights_9_only_new1"))
# model.load_state_dict(torch.load(f"model_weights_9_resnet151_1"))
criterion = nn.CrossEntropyLoss()

optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, 10)
no_of_layers = 0
conv_layers = []
weights = []
model.eval()
model_children = list(model.modules())



# print(conv_layers)
#
# data1 = iter(test_dataset)
# img1, label1 = next(data1)
# img1, label1 = img1.cuda(), label1.cuda()
# activations = []
# results = [conv_layers[0](img1)]



# for num, layer in enumerate(conv_layers[1:]):
#     print(num)
#     results.append(layer(results[-1]).cpu())
# print(np.array(results).shape)
# assert 0
# optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.001)
#scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=400, gamma=0.7)
# image = Image.open('/home/vigen/Desktop/classification_improve_dataset/new_full_dataset/carpet/370.png')
# image = T.PILToTensor()(image)
# image = T.Resize((224, 224))(image)
# image = image.float()ls
# image = T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))(image)
# image = image[None, :]
# image = image.to(device)
# model.eval()
# print(model(image))
# assert 0

writer = SummaryWriter()
# model.load_state_dict(torch.load("model_weights_temp_6_new9_eff"))
show_interval = 4
for epoch in range(n_epochs):
    true_count = 0
    samples_count = 0
    true_count_test = 0
    samples_count_test = 0
    test_dataset_loader = iter(test_dataset)
    torch.save(model.state_dict(), f"model_weights_{epoch}_pspnet")
    # for num, (name, layers) in enumerate(model.fc.named_parameters()):
    #     writer.add_histogram(name+str(epoch), layers, num)
    for num, (image, label, name) in enumerate(train_dataset):
        # print(name[0])
        image, label = image.cuda(), label.cuda()
        # print(image, label)
        #image = image[None, :]
        pred = model(image)
        # display_label(label, pred)
        loss = criterion(pred, label)
        # print(pred.shape)
        # print(label.shape)
        l1_penalty = torch.nn.L1Loss(size_average=False)
        reg_loss = 0
        for param in model.parameters():
            reg_loss += torch.norm(param)

        loss += 0.001 * reg_loss
        l1=torch.tensor([torch.sum(torch.abs(layer)) for layer in model.parameters()])
        loss += 0.001*torch.sum(l1)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()
        #scheduler.step()


        predicted = torch.argmax(pred, 1)


        label = torch.argmax(label, 1)
        diff = predicted - label

        np_label = label.cpu().numpy()

        wood_count = np.where(np_label == 1)[0]
        carpet_count = np.where(np_label == 0)[0]

        # wood_count = np.where(np_label == 1)
        # carpet_count = np.where(np_label == 0)

        count = len(diff[diff == 0])
        # print(f"Sample True rate: {count/(batch_size_train*224*224)}")
        true_count += count
        samples_count += batch_size_train*224*224

        if num % show_interval == 0 and num != 0:
            # if num >= 408:
            #     break
            with torch.no_grad():
                test_image, test_label, name2 = next(test_dataset_loader)
                test_image, test_label = test_image.cuda(), test_label.cuda()
                pred_test = model(test_image)
                loss_test = criterion(pred_test, test_label)
                predicted = torch.argmax(pred_test, 1)
                test_label = torch.argmax(test_label, 1)
                diff = predicted - test_label
                count = len(diff[diff == 0])

                true_count_test += count
                samples_count_test += batch_size_test * 224 * 224

                # wood_count = np.where(np_label == 1)
                # carpet_count = np.where(np_label == 0)


                writer.add_scalar("loss_train", loss_test, num)
                writer.add_scalar("loss_test", loss_test, num)
                writer.add_scalar("accuracy_train", true_count/samples_count, num)
                writer.add_scalar("accuracy_test", true_count_test / samples_count_test, num)
                print("Epoch: ", epoch, "iter: ", num)
                print(f"Accuracy: {true_count/samples_count}  Loss : {loss}")
                print(f"Test Accuracy: {true_count_test / samples_count_test}  Test Loss : {loss_test}")


torch.save(model.state_dict(), "model_weights1_new_eff")
for num, (name, layers) in enumerate(model.fc.named_parameters()):
    writer.add_histogram(name, layers, num)

writer.close()
