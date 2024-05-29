import torch
from torch import nn
from until import ResizeFocus, BasicConv, EntAttMix, EntBlock, ClsTokenMix


class MSConv(nn.Module):
    def __init__(self, model, feature_size, classes_num, use_ent=True, use_resize=True, use_vit=False, use_mix=True):
        super(MSConv, self).__init__()
        self.use_resize = use_resize
        self.use_vit = use_vit
        self.use_mix = use_mix
        self.use_ent = use_ent

        self.features = model
        self.num_ftrs = 1024 * 1 * 1
        self.elu = nn.ELU(inplace=True)
        self.mix = ClsTokenMix()

        output_len = feature_size * 4
        self.resize = ResizeFocus(2, 1)
        self.entblock1 = EntBlock(1, 128, 4)
        self.entblock2 = EntBlock(128, 256, 2)
        self.entblock3 = EntBlock(256, 512, 2)
        self.cls_token = ClsBlock(512, 512)
        self.attntion = EntAttMix(512, 512)

        self.conv_block1 = nn.Sequential(
            BasicConv(self.num_ftrs // 4, feature_size, kernel_size=1, stride=1, padding=0, relu=True),
            BasicConv(feature_size, self.num_ftrs // 2, kernel_size=3, stride=1, padding=1, relu=True)
        )

        self.conv_block2 = nn.Sequential(
            BasicConv(self.num_ftrs // 2, feature_size, kernel_size=1, stride=1, padding=0, relu=True),
            BasicConv(feature_size, self.num_ftrs // 2, kernel_size=3, stride=1, padding=1, relu=True)
        )

        self.conv_block3 = nn.Sequential(
            BasicConv(self.num_ftrs, feature_size, kernel_size=1, stride=1, padding=0, relu=True),
            BasicConv(feature_size, self.num_ftrs // 2, kernel_size=3, stride=1, padding=1, relu=True)
        )

        # self.max1 = nn.MaxPool2d(kernel_size=7, stride=7)
        # self.max2 = nn.MaxPool2d(kernel_size=7, stride=7)
        # self.max3 = nn.MaxPool2d(kernel_size=7, stride=7)
        self.resize0 = ResizeFocus(1, 3)
        self.resize1 = ResizeFocus(2, 3)
        self.resize2 = ResizeFocus(4, 3)

        self.classifier_concat = nn.Sequential(
            nn.BatchNorm1d(output_len),
            nn.Linear(output_len, feature_size),
            nn.BatchNorm1d(feature_size),
            nn.ELU(inplace=True),
            nn.Linear(feature_size, classes_num),
        )

    def forward(self, x, ent):
        x = self.resize0(x)
        x1 = self.resize1(x)
        x2 = self.resize2(x)
        _, _, xf3, _, _ = self.features(x2)
        _, _, _, xf4, _ = self.features(x1)
        _, _, _, _, xf5 = self.features(x)

        xl1 = self.conv_block1(xf3)
        xl2 = self.conv_block2(xf4)
        xl3 = self.conv_block3(xf5)

        # xc1 = self.max1(xl1)
        # xc1 = xc1.view(xc1.size(0), -1)
        # xc2 = self.max2(xl2)
        # xc2 = xc2.view(xc2.size(0), -1)
        # xc3 = self.max3(xl3)
        # xc3 = xc3.view(xc3.size(0), -1)

        x_concat, x3 = self.mix(xl1, xl2, xl3)

        ent = self.resize(ent)
        ent = self.entblock1(ent)
        ent = self.entblock2(ent)
        ent = self.entblock3(ent)
        ent = self.cls_token(ent)
        ent = self.attntion(x3, ent)

        x_concat = torch.cat((x_concat, ent), -1)
        x_concat = self.classifier_concat(x_concat)
        return x_concat