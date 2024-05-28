import csv
import os
import argparse
import datetime
import torch
import torch.optim as optim
import torchvision
# from torch import nn
# from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
from predict_csv import predict_out_mini
from my_dataset import MyDataSet
from model import NewConv as create_model
from utils import *
import wandb


def main(args):
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"using {device} {torch.cuda.get_device_name(0)} device.")
    if os.path.exists("./weights/") is False:
        os.makedirs("./weights/")
    img_size = 224
    data_transform = {
        "train": transforms.Compose([
            transforms.Resize(int(img_size * 1.25)),
            transforms.RandomRotation(90),
            transforms.RandomResizedCrop(img_size, (0.5, 1)),
            transforms.RandomHorizontalFlip(),
            transforms.
            transforms.ColorJitter(),
            transforms.ToTensor(),
        ]),
        "test": transforms.Compose([transforms.Resize(int(img_size * 1.25)),
                                    transforms.CenterCrop(img_size),
                                    transforms.ToTensor(),
                                    ])}

    # 实例化训练数据集
    train_dataset = MyDataSet(root=os.path.join(args.data_path, 'train'),
                              transform=data_transform['train'])

    # 实例化验证数据集
    test_dataset = MyDataSet(root=os.path.join(args.data_path, 'test'),
                             transform=data_transform['test'])

    batch_size = args.batch_size
    nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 6])  # number of workers
    print('Using {} dataloader workers every process'.format(nw))
    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=batch_size,
                                               shuffle=True,
                                               pin_memory=True,
                                               num_workers=nw
                                               )

    val_loader = torch.utils.data.DataLoader(test_dataset,
                                             batch_size=batch_size,
                                             shuffle=False,
                                             pin_memory=True,
                                             num_workers=nw
                                             )

    # model = create_model(in_channels=3, num_classes=args.num_classes, classifier=True, img_size=img_size,
    #                      transformer=True).to(device)
    if args.resume:
        model = torch.load(args.model_path)
        for name, para in model.named_parameters():
            # 除head外，其他权重全部冻结
            if 'classifier_concat' in name:
                para.requires_grad_(True)
                print("training {}".format(name))
            else:
                para.requires_grad_(False)
    else:
        model = load_model(model_name='resnet50_pmg', pretrain=True, require_grad=True, class_num=args.num_classes,
                           in_c=3, use_ent=True)
    model.to(device)
    # optimizer = optim.AdamW(model.parameters(), lr=args.lr)
    optimizer = optim.RMSprop(model.parameters(), lr=args.lr)
    # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', factor=0.5, patience=10, eps=1e-7)
    scheduler = create_lr_scheduler(optimizer, len(train_loader), args.epochs,
                                       warmup=True, warmup_epochs=1)
    best_acc = 0.75
    nowTime = datetime.datetime.now().strftime('%Y-%m-%d-%H:%M:%S')
    with open('./results.txt', 'a') as file:
        file.write('start at {}\n'.format(nowTime))
    for epoch in range(args.epochs):
        # train
        train_loss, train_acc = train_one_epoch(model=model,
                                                optimizer=optimizer,
                                                data_loader=train_loader,
                                                device=device,
                                                epoch=epoch,
                                                scheduler=scheduler,
                                                class_num=args.num_classes)

        # validate
        val_loss, val_acc = evaluate(model=model,
                                     data_loader=val_loader,
                                     device=device,
                                     epoch=epoch,
                                     class_num=args.num_classes
                                     )
        # scheduler.step(val_acc)
        wandb.log({"train_acc": train_acc, "train_loss": train_loss, "val_acc": val_acc, "val_loss": val_loss})
        if epoch + 1 == args.epochs:
            model.cpu()
            torch.save(model,
                       "./weights/model_final_class{}_{}.pth".format(args.num_classes, str(nowTime)))
            model.to(device)
        if best_acc < val_acc:
            try:
                # 使用os.remove()函数删除文件
                os.remove(best_save)
            except FileNotFoundError:
                print(f"文件未找到。")
            except Exception as e:
                print(f"删除文件时发生错误：{str(e)}")
            with open('./results_best.txt', 'a') as file:
                file.write('epoch %d, train_acc = %.5f, train_loss = %.6f,test_acc = %.5f, test_loss = %.6f\n' % (
                    epoch, train_acc, train_loss, val_acc, val_loss))
            model.cpu()
            best_save = "./weights/model_class{}_acc{:.3f}.pth".format(args.num_classes, val_acc)
            torch.save(model, best_save)
            model.to(device)
            best_acc = val_acc
        # if val_acc == 0.999 or optimizer.param_groups[0]["lr"] <= 1e-8:
        #     model.cpu()
        #     torch.save(model, "./weights/model_final_{}_{}.pth".format(os.path.basename(args.data_path)[-1],str(nowTime)))
        #     model.to(device)
        #     break


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_classes', type=int, default=200)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch-size', type=int, default=4)
    parser.add_argument('--lr', type=float, default=1e-5)
    parser.add_argument('--data-path', type=str,
                        default=r"E:\CUB")
    parser.add_argument('--device', default='cuda:0', help='device id (i.e. 0 or 0,1 or cpu)')
    parser.add_argument('--resume', type=bool, default=False)
    parser.add_argument('--model_path', type=str,
                        default='./weights/model_best_class2_acc0.953.pth')
    opt = parser.parse_args()
    wandb.init(
        # set the wandb project where this run will be logged
        project="test-project",

        # track hyperparameters and run metadata
        config={
            "learning_rate": 1e-4,
            "architecture": "resnet with multi-feature",
            "dataset": "personal",
            "epochs": 150,
        }
    )
    main(opt)
