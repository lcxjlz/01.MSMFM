import os
import sys

import matplotlib
import numpy as np
import pandas as pd
from torch import nn
from torchvision import transforms
from my_dataset import MyDataSet

sys.path.insert(0, os.getcwd())
import matplotlib.pyplot as plt

from torch.utils.data import DataLoader
from utils import *
from sklearn.manifold import TSNE
import seaborn as sns

pinyin_to_english = {
    'ALL-a': 'ALL',
    'AML-a': 'AML',
    'APL-a': 'APL',
    'CLL-a': 'CLL',
    'CMLYY-a': 'CML',
    'CMML-a': 'CMML',
    'MDS-a': 'MDS',
    'danhe-a': 'Monocytes',
    'lingba-a': 'Lymphocytes',
    'shijian-a': 'Basophils',
    'shisuan-a': 'Eosinophils',
    'zhongxing-a': 'Neutrophils',
    'ALL': 'ALL',
    'ALL-PH': 'PH+ALL',
    'AML': 'AML',

}

def plot_tsne(tsne_features=None, labels=None, classes=None):
    x_min, x_max = np.min(tsne_features, 0), np.max(tsne_features, 0)
    tsne_features = (tsne_features - x_min) / (x_max - x_min)  # 对数据进行归一化处理
    # 一个类似于表格的数据结构
    df = pd.DataFrame()
    df["y"] = labels
    df["comp1"] = tsne_features[:, 0]
    df["comp2"] = tsne_features[:, 1]

    # 颜色是根据标签的大小顺序进行赋色.
    hex = ['b', 'g', 'r', 'c', 'm', 'y', 'k', '#FF5733', '#007ACC', '#FFA500', '#8B008B']
    data_label = []
    for i in range(len(classes)):
        print(classes[i])
    for v in df.y.tolist():
        data_label.append(classes[v])

    df["value"] = data_label

    # hue=df.y.tolist()
    # hue:根据y列上的数据种类，来生成不同的颜色；
    # style:根据y列上的数据种类，来生成不同的形状点；
    # s:指定显示形状的大小
    sns.scatterplot(x=df.comp1.tolist(), y=df.comp2.tolist(), hue=df.value.tolist(), style=df.value.tolist(),
                    palette=sns.color_palette(hex, len(classes)),
                    markers={
                        "ALL": ".", "AML": ".", "CLL": ".", "CML": ".", "CMML": ".",
                        "MDS": ".", "Monocytes": ".", "Lymphocytes": ".", "Basophils": ".",
                        "Eosinophils": ".", "Neutrophils": "."
                    },
                    # s = 10,
                    data=df).set(title="")  # T-SNE projection

    # 指定图注的位置 "lower right"
    plt.legend(loc="lower right")
    # 不要坐标轴
    plt.axis("off")
    # 保存图像
    plt.savefig('./tsne-2.jpg', format="jpg", dpi=300)
    plt.show()


def main(model_path=None, data_path=None):
    """
    创建评估文件夹、metrics文件、混淆矩阵文件
    """
    # dirname = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
    # save_dir = os.path.join('eval_results', dirname)
    # metrics_output = os.path.join(save_dir, 'metrics_output.csv')
    # prediction_output = os.path.join(save_dir, 'prediction_results.csv')
    # os.makedirs(save_dir)
    """
    制作测试集并喂入Dataloader
    """
    use_cuda = torch.cuda.is_available()
    print(use_cuda)
    # GPU
    device = torch.device("cuda:0")
    net = torch.load(model_path, map_location=device)
    net.to(device)
    net.eval()
    net.fc = nn.Identity()
    # cudnn.benchmark = True

    transform_test = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])
    test_dataset = MyDataSet(data_path, transform_test)
    print(test_dataset.classes)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=16, shuffle=False, num_workers=1)
    """
    计算Precision、Recall、F1 Score、Confusion matrix
    """
    # tsne_data = np.zeros(len(test_dataset), 1024 * 3)
    with torch.no_grad():
        preds, targets, image_paths = [], [], []
        with tqdm(total=len(test_dataset) // 16) as pbar:
            for _, batch in enumerate(test_loader):
                images, ent, target, path = batch
                output_concat = net(images.to(device),ent.to(device))
                # outputs_com = output_1 + output_2 + output_3 + output_concat
                # outputs = torch.softmax(outputs_com.data, 1)
                # preds.append(outputs)
                # targets.append(target.to(device))
                # image_paths.extend(images)

                output_concat = np.array(output_concat.cpu())
                target = np.array(target)
                if not 'tsne_data' in vars():
                    tsne_data = output_concat
                    labels = target
                tsne_data = np.append(tsne_data, output_concat, axis=0)
                labels = np.append(labels, target, axis=0)
                pbar.update(1)
    print(tsne_data.shape, labels.shape)
    print('Begining......')  # 时间会较长，所有处理完毕后给出finished提示
    tsne_2D = TSNE(n_components=2, init='pca', random_state=0)  # 调用TSNE
    result_2D = tsne_2D.fit_transform(tsne_data)
    print('Finished......')
    matplotlib.use('TkAgg')
    names = [pinyin_to_english[name] for name in test_dataset.classes]
    plot_tsne(result_2D, labels, names)
    # eval_results = evaluate(torch.cat(preds), torch.cat(targets), metrics,
    #                         metric_options)
    #
    # APs = plot_ROC_curve(torch.cat(preds), torch.cat(targets), classes_names, save_dir)
    # get_metrics_output(eval_results, metrics_output, classes_names, indexs, APs)
    # get_prediction_output(torch.cat(preds), torch.cat(targets), image_paths, classes_names, indexs, prediction_output)


if __name__ == "__main__":
    main(model_path='./weights/model_best_class2_acc0.950.pth', data_path='../../../blood-cell-class/class-test-class2-a')
