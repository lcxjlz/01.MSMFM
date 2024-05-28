import os
import sys
import json
import pickle
import random
import shutil
import scipy.misc
import cv2
from matplotlib import pyplot as plt
from torchvision import transforms

from Resnet import *
from model_new import *
import numpy as np
import torch
from tqdm import tqdm
from focal_loss import FocalLoss
from unet import UNet
from convnext import convnext_base
import math


def load_model(model_name, pretrain=True, require_grad=True, class_num=13, in_c=3, use_ent=True):
    print('==> Building model..')
    if model_name == 'resnet50_pmg':
        # net = resnet50(pretrained=pretrain, in_c=in_c)
        net = convnext_base(num_classes=class_num)
        for param in net.parameters():
            param.requires_grad = require_grad
    else:
        net = UNet(in_channels=in_c, num_classes=class_num)
    # net = MGCM(net, 1024, class_num, use_ent)
    net = MSConv(net, 512, class_num, use_ent)
    # net = PMGN(net, 1024, class_num, use_ent)
    return net


def write_pickle(list_info: list, file_name: str):
    with open(file_name, 'wb') as f:
        pickle.dump(list_info, f)


def read_pickle(file_name: str) -> list:
    with open(file_name, 'rb') as f:
        info_list = pickle.load(f)
        return info_list


def train_one_epoch(model, optimizer, data_loader, device, epoch, scheduler, class_num=4):
    model.train()
    # loss_function = FocalLoss(class_num=class_num, device=device)
    loss_function = torch.nn.CrossEntropyLoss()
    accu_loss = torch.zeros(1).to(device)  # 累计损失
    accu_num = torch.zeros(1).to(device)  # 累计预测正确的样本数
    optimizer.zero_grad()

    sample_num = 0
    data_loader = tqdm(data_loader, file=sys.stdout, ncols=85)
    for step, data in enumerate(data_loader):
        images, ent, labels, _ = data
        sample_num += images.shape[0]
        # inputs = torch.cat((images, ent), dim=1)
        # optimizer.zero_grad()
        # inputs1 = jigsaw_generator(inputs, 8)
        # images, ent = inputs1[:, 0:-1], inputs1[:, -1].unsqueeze(1)
        # output_1, _, _, _ = model(images.to(device), ent.to(device))
        # loss1 = loss_function(output_1, labels.to(device))
        # loss1.backward()
        # optimizer.step()
        #
        # optimizer.zero_grad()
        # inputs1 = jigsaw_generator(inputs, 4)
        # images, ent = inputs1[:, 0:-1], inputs1[:, -1].unsqueeze(1)
        # _, output_2, _, _ = model(images.to(device), ent.to(device))
        # loss2 = loss_function(output_2, labels.to(device))
        # loss2.backward()
        # optimizer.step()
        # optimizer.zero_grad()
        # inputs1 = jigsaw_generator(inputs, 2)
        # images, ent = inputs1[:, 0:-1], inputs1[:, -1].unsqueeze(1)
        # _, _, output_3, _ = model(images.to(device), ent.to(device))
        # loss3 = loss_function(output_3, labels.to(device))
        # loss3.backward()
        # optimizer.step()
        #
        optimizer.zero_grad()
        output_concat = model(images.to(device), ent.to(device))
        loss = loss_function(output_concat, labels.to(device))
        loss.backward()
        optimizer.step()
        pred_classes = torch.max(output_concat, dim=1)[1]
        accu_num += torch.eq(pred_classes, labels.to(device)).sum()
        # loss = loss1 + loss2 + loss3 + concat_loss
        accu_loss += loss.detach()

        data_loader.desc = "[train epoch {}] loss: {:.3f}, acc: {:.3f}, lr: {:.2e}".format(
            epoch + 1,
            accu_loss.item() / (step + 1),
            accu_num.item() / sample_num,
            optimizer.param_groups[0]["lr"]
        )
        scheduler.step()
        if not torch.isfinite(loss):
            print('WARNING: non-finite loss, ending training ', loss)
            sys.exit(1)
    return accu_loss.item() / (step + 1), accu_num.item() / sample_num


@torch.no_grad()
def evaluate(model, data_loader, device, epoch, class_num=4):
    # loss_function = FocalLoss(class_num=class_num, device=device)
    loss_function = torch.nn.CrossEntropyLoss()
    model.eval()

    accu_num = torch.zeros(1).to(device)  # 累计预测正确的样本数
    accu_loss = torch.zeros(1).to(device)  # 累计损失

    sample_num = 0
    data_loader = tqdm(data_loader, file=sys.stdout, ncols=85)
    for step, data in enumerate(data_loader):
        images, ent, labels, _ = data
        sample_num += images.shape[0]

        output_concat = model(images.to(device), ent.to(device))
        # outputs_com = output_1 + output_2 + output_3 + output_concat
        pred_classes = torch.max(output_concat, dim=1)[1]
        accu_num += torch.eq(pred_classes, labels.to(device)).sum()

        loss = loss_function(output_concat, labels.to(device))
        accu_loss += loss

        data_loader.desc = "[valid epoch {}] loss: {:.3f}, acc: {:.3f}".format(epoch + 1,
                                                                               accu_loss.item() / (step + 1),
                                                                               accu_num.item() / sample_num)

    return accu_loss.item() / (step + 1), accu_num.item() / sample_num


def test(net, data_loader, device, epoch, **kwargs):
    net.eval()
    use_cuda = torch.cuda.is_available()
    test_loss = 0
    correct = 0
    correct_com = 0
    total = 0
    idx = 0
    loss_function = torch.nn.CrossEntropyLoss()
    for batch_idx, (inputs, ent, targets) in enumerate(data_loader):
        idx = batch_idx
        if use_cuda:
            inputs, ent, targets = inputs.to(device), ent.to(device), targets.to(device)
        # inputs, targets = Variable(inputs, volatile=True), Variable(targets)
        with torch.no_grad():
            output_1, output_2, output_3, output_concat = net(inputs, ent)
            outputs_com = output_1 + output_2 + output_3 + output_concat

            loss = loss_function(output_concat, targets)

            test_loss += loss.item()
            _, predicted = torch.max(output_concat.data, 1)
            _, predicted_com = torch.max(outputs_com.data, 1)
            total += targets.size(0)
            correct += predicted.eq(targets.data).cpu().sum()
            correct_com += predicted_com.eq(targets.data).cpu().sum()

        if batch_idx % 1 == 0:
            print('Step: %d | Loss: %.3f | Acc: %.3f%% (%d/%d) |Combined Acc: %.3f%% (%d/%d)' % (
                batch_idx, test_loss / (batch_idx + 1), 100. * float(correct) / total, correct, total,
                100. * float(correct_com) / total, correct_com, total))

    test_acc = 100. * float(correct) / total
    test_acc_en = 100. * float(correct_com) / total
    test_loss = test_loss / (idx + 1)
    return test_acc, test_acc_en, test_loss


def train(net, optimizer, data_loader, device, epoch, batch_size, nb_epoch, lr, **kwargs):
    print('\nEpoch: %d' % epoch)
    net.train()
    CELoss = torch.nn.CrossEntropyLoss()
    train_loss = 0
    train_loss1 = 0
    train_loss2 = 0
    train_loss3 = 0
    train_loss4 = 0
    correct = 0
    total = 0
    idx = 0
    for batch_idx, (inputs, ent, targets) in enumerate(data_loader):
        idx = batch_idx
        if inputs.shape[0] < batch_size:
            continue
        inputs, ent, targets = inputs.to(device), ent.to(device), targets.to(device)
        # update learning rate
        for nlr in range(len(optimizer.param_groups)):
            optimizer.param_groups[nlr]['lr'] = cosine_anneal_schedule(epoch, nb_epoch, lr[nlr])
        inputs = torch.cat((inputs, ent), dim=1)
        # Step 1
        optimizer.zero_grad()
        inputs1 = jigsaw_generator(inputs, 8)
        output_1, _, _, _ = net(inputs1)
        loss1 = CELoss(output_1, targets) * 1
        loss1.backward()
        optimizer.step()

        # Step 2
        optimizer.zero_grad()
        inputs2 = jigsaw_generator(inputs, 4)
        _, output_2, _, _ = net(inputs2)
        loss2 = CELoss(output_2, targets) * 1
        loss2.backward()
        optimizer.step()

        # Step 3
        optimizer.zero_grad()
        inputs3 = jigsaw_generator(inputs, 2)
        _, _, output_3, _ = net(inputs3)
        loss3 = CELoss(output_3, targets) * 1
        loss3.backward()
        optimizer.step()

        # Step 4
        optimizer.zero_grad()
        _, _, _, output_concat = net(inputs)
        concat_loss = CELoss(output_concat, targets) * 2
        concat_loss.backward()
        optimizer.step()

        #  training log
        _, predicted = torch.max(output_concat.data, 1)
        total += targets.size(0)
        correct += predicted.eq(targets.data).cpu().sum()

        train_loss += (loss1.item() + loss2.item() + loss3.item() + concat_loss.item())
        train_loss1 += loss1.item()
        train_loss2 += loss2.item()
        train_loss3 += loss3.item()
        train_loss4 += concat_loss.item()

        if batch_idx % 1 == 0:
            print(
                'Step: %d | Loss1: %.3f | Loss2: %.5f | Loss3: %.5f | Loss_concat: %.5f | Loss: %.3f | Acc: %.3f%% (%d/%d)' % (
                    batch_idx, train_loss1 / (batch_idx + 1), train_loss2 / (batch_idx + 1),
                    train_loss3 / (batch_idx + 1), train_loss4 / (batch_idx + 1), train_loss / (batch_idx + 1),
                    100. * float(correct) / total, correct, total))
    train_acc = 100. * float(correct) / total
    train_loss = train_loss / (idx + 1)
    return train_acc, train_loss


def jigsaw_generator(images, n):
    l = []
    for a in range(n):
        for b in range(n):
            l.append([a, b])
    block_size = 224 // n
    rounds = n ** 2
    random.shuffle(l)
    jigsaws = images.clone()
    for i in range(rounds):
        x, y = l[i]
        temp = jigsaws[..., 0:block_size, 0:block_size].clone()
        jigsaws[..., 0:block_size, 0:block_size] = jigsaws[..., x * block_size:(x + 1) * block_size,
                                                   y * block_size:(y + 1) * block_size].clone()
        jigsaws[..., x * block_size:(x + 1) * block_size, y * block_size:(y + 1) * block_size] = temp

    return jigsaws


def cosine_anneal_schedule(t, nb_epoch, lr):
    cos_inner = np.pi * (t % (nb_epoch))  # t - 1 is used when t has 1-based indexing.
    cos_inner /= (nb_epoch)
    cos_out = np.cos(cos_inner) + 1

    return float(lr / 2 * cos_out)


def create_lr_scheduler(optimizer,
                        num_step: int,
                        epochs: int,
                        warmup=True,
                        warmup_epochs=1,
                        warmup_factor=1e-3,
                        end_factor=1e-6):
    assert num_step > 0 and epochs > 0
    if warmup is False:
        warmup_epochs = 0

    def f(x):
        """
        根据step数返回一个学习率倍率因子，
        注意在训练开始之前，pytorch会提前调用一次lr_scheduler.step()方法
        """
        if warmup is True and x <= (warmup_epochs * num_step):
            alpha = float(x) / (warmup_epochs * num_step)
            # warmup过程中lr倍率因子从warmup_factor -> 1
            return warmup_factor * (1 - alpha) + alpha
        else:
            current_step = (x - warmup_epochs * num_step)
            cosine_steps = (epochs - warmup_epochs) * num_step
            # warmup后lr倍率因子从1 -> end_factor
            return ((1 + math.cos(current_step * math.pi / cosine_steps)) / 2) * (1 - end_factor) + end_factor

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=f)


def feature_map_to_image(feature_map, output_folder, prefix='feature_map'):
    os.makedirs(output_folder, exist_ok=True)
    feature_map = feature_map.cpu().detach().numpy()
    # 将特征图标准化到0-255范围内
    feature_map = cv2.normalize(feature_map, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    # 创建一个空的三通道彩色图像
    color_image = np.zeros((feature_map.shape[1], feature_map.shape[2], 3), dtype=np.uint8)

    # 将特征图的每个通道复制到彩色图像的每个通道上
    for i in range(3):
        color_image[:, :, i] = feature_map[:, :, 0]
    # 遍历特征图
    for i, feature in enumerate(feature_map):
        # 将每个特征图转换为图片格式
        img_path = os.path.join(output_folder, f"{prefix}_{i}.png")
        cv2.imwrite(img_path, color_image)

        print(f"Feature map {i} saved as {img_path}")


#  可视化特征图
def show_feature_map(
        feature_map,
        feature_map_save):  # feature_map=torch.Size([1, 64, 55, 55]),feature_map[0].shape=torch.Size([64, 55, 55])
    # feature_map[2].shape     out of bounds
    feature_map = feature_map.squeeze(0)  # 压缩成torch.Size([64, 55, 55])

    # 以下4行，通过双线性插值的方式改变保存图像的大小
    feature_map = feature_map.view(1, feature_map.shape[0], feature_map.shape[1], feature_map.shape[2])  # (1,64,55,55)
    upsample = torch.nn.UpsamplingBilinear2d(size=(256, 256))  # 这里进行调整大小
    feature_map = upsample(feature_map)
    feature_map = feature_map.view(feature_map.shape[1], feature_map.shape[2], feature_map.shape[3])

    feature_map_num = feature_map.shape[0]  # 返回通道数
    row_num = int(np.ceil(np.sqrt(feature_map_num)))  # 8
    plt.figure()
    for index in range(1, feature_map_num + 1):  # 通过遍历的方式，将64个通道的tensor拿出
        # plt.imshow(feature_map[index - 1], cmap='gray')  # feature_map[0].shape=torch.Size([55, 55])
        plt.imshow(transforms.ToPILImage()(feature_map[index - 1]))  # feature_map[0].shape=torch.Size([55, 55])
        plt.axis('off')
        plt.savefig(feature_map_save + '//' + str(index) + ".png")  # 图像保存
