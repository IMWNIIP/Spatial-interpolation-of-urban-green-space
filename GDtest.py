#导入运行所需包
from __future__ import print_function
import argparse
import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
from torch.autograd import Variable



#设置参数
nc = 1 #模型输入通道数
L_ele = 4.49 # The Lowest and Highest elevation for dem re-normalization
H_ele = 9.69
batchSize = 64
ngf = batchSize #函数输出通道数
ndf = batchSize
imageSize = 32
nthread = 0 #数据读取线程数
ncp = 100 #控制点个数
nk = 1 #判别器对生成器的判别次数
niter = 20 #循环数
lr = 0.0002
beta1 = 0.5 #adam方法平均数
cuda = True
outf = './output_test' #输出文件夹
datafile = 'F:/training_data/PH_Cut/64_test'
manualSeed = 999 #随机种子
netDdir = 'F:/training_data/PH_Cut/output64_100/nets/netD_epoch_10000.pth' #预训练模型地址
netGdir = 'F:/training_data/PH_Cut/output64_100/nets/netG_epoch_10000.pth'
npre = 0
logfile = './errlogtestcpt64100.txt'



if manualSeed is None:
    manualSeed = random.randint(1, 10000)
print("Random Seed: ", manualSeed)
random.seed(manualSeed)
torch.manual_seed(manualSeed)
if cuda:
    torch.cuda.manual_seed_all(manualSeed)

cudnn.benchmark = True

#生成器设置
class Generator(nn.Module):
    def __init__(self, nc, ngf):
        super(Generator, self).__init__()
        self.layer1 = nn.Sequential(nn.Conv2d(nc, ngf, kernel_size=4, stride=2, padding=1),
                                    nn.BatchNorm2d(ngf),
                                    nn.LeakyReLU(0.2, inplace=True))
        # 16 x 16 x 64
        self.layer2 = nn.Sequential(nn.Conv2d(ngf, ngf * 2, kernel_size=4, stride=2, padding=1),
                                    nn.BatchNorm2d(ngf * 2),
                                    nn.LeakyReLU(0.2, inplace=True))
        # 8 x 8 x 128

        self.layer3 = nn.Sequential(nn.Conv2d(ngf * 2, ngf * 4, kernel_size=4, stride=2, padding=1),
                                    nn.BatchNorm2d(ngf * 4),
                                    nn.LeakyReLU(0.2, inplace=True))
        # 4 x 4 x 256
        # 4 x 4 x 256
        self.layer4 = nn.Sequential(nn.ConvTranspose2d(ngf * 4, ngf * 2, kernel_size=4, stride=2, padding=1),
                                    nn.BatchNorm2d(ngf * 2),
                                    nn.ReLU())
        # 8 x 8 x 128
        self.layer5 = nn.Sequential(nn.ConvTranspose2d(ngf * 2, ngf, kernel_size=4, stride=2, padding=1),
                                    nn.BatchNorm2d(ngf),
                                    nn.ReLU())
        # 16 x 16 x 64
        self.layer6 = nn.Sequential(nn.ConvTranspose2d(ngf, nc, kernel_size=4, stride=2, padding=1),
                                    nn.Tanh())
        # 32 x 32 x 1

    def forward(self, _cpLayer):
        out = self.layer1(_cpLayer)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.layer5(out)
        out = self.layer6(out)
        #out = torch.squeeze(out)
        return out

#判别器设置
class Discriminator(nn.Module):
    def __init__(self, nc, ndf):
        super(Discriminator, self).__init__()
        self.layer1_image = nn.Sequential(nn.Conv2d(nc, int(ndf / 2), kernel_size=4, stride=2, padding=1),
                                          # nn.BatchNorm2d(ndf/2),
                                          nn.LeakyReLU(0.2, inplace=True))
        # 16 x 16
        self.layer1_cp = nn.Sequential(nn.Conv2d(nc, int(ndf / 2), kernel_size=4, stride=2, padding=1),
                                       # nn.BatchNorm2d(ndf/2),
                                       nn.LeakyReLU(0.2, inplace=True))
        # 16 x 16
        self.layer2 = nn.Sequential(nn.Conv2d(ndf, ndf * 2, kernel_size=4, stride=2, padding=1),
                                    nn.BatchNorm2d(ndf * 2),
                                    nn.LeakyReLU(0.2, inplace=True))
        # 8 x 8

        self.layer3 = nn.Sequential(nn.Conv2d(ndf * 2, ndf * 4, kernel_size=4, stride=2, padding=1),
                                    nn.BatchNorm2d(ndf * 4),
                                    nn.LeakyReLU(0.2, inplace=True))
        # 4 x 4

        self.layer4 = nn.Sequential(nn.Conv2d(ndf * 4, 1, kernel_size=4, stride=1, padding=0),
                                    nn.Sigmoid())
        # 1

    def forward(self, dem, _cpLayer):
        out_1 = self.layer1_image(dem)
        out_2 = self.layer1_cp(_cpLayer)
        out = self.layer2(torch.cat((out_1, out_2), 1))
        out = self.layer3(out)
        out = self.layer4(out)
        out = torch.squeeze(out)
        return out




###############   DATASET   ##################
dataset = dset.ImageFolder(root=str(datafile), transform=transforms.Compose([
            transforms.Resize([imageSize,imageSize]),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            #transforms.Grayscale(1)  # 添加这一行以转换图像为灰度图像
        ]))

loader = torch.utils.data.DataLoader(dataset=dataset,
                                     batch_size=batchSize,
                                     shuffle=True, drop_last=True, num_workers=nthread)

#print(dataset[0][0].shape)

###############   MODEL and initialization  ####################

netD = Discriminator(nc, ndf)
netG = Generator(nc, ngf)
if (netG != '' and netD != ''):

    netG.load_state_dict(torch.load(str(netGdir)))
    netD.load_state_dict(torch.load(str(netDdir)))
if (cuda):
    netD.cuda()
    netG.cuda()


###############   self-defined FUNCTION   ####################
# def ControlPointsImage(dems, ncp):  # Uniform sampling
#     y_cpLayer = torch.FloatTensor(opt.batchSize, nc, opt.imageSize, opt.imageSize).zero_()
#     y_cpLayer = Variable(y_cpLayer)
#     if (opt.cuda):
#         y_cpLayer = y_cpLayer.cuda()
#     cp = []
#     x_index = []
#     y_index = []
#     step = np.floor(opt.imageSize / np.sqrt(ncp))
#     pad = opt.imageSize % np.floor(np.sqrt(ncp))
#     # print step,pad
#     for i in range(0, int(np.floor(np.sqrt(ncp)))):
#         x_index.append(pad + i * step)
#         y_index.append(pad + i * step)
#     for n in range(0, opt.batchSize):
#         cp.append([])
#         for i in x_index:
#             for j in y_index:
#                 cp[n].append([dems[n, 0, i, j], i, j])  # extract dem control point function, one channel
#     for i in range(0, opt.batchSize):
#         for _cp in cp[i]:
#             h = _cp[0]
#             x = _cp[1]
#             y = _cp[2]
#             y_cpLayer[i, 0, x, y] = h
#     return y_cpLayer


def ControlPointsImage_random(dems, ncp):  # Random sampling
    y_cpLayer = torch.FloatTensor(batchSize, nc, imageSize, imageSize).zero_()
    y_cpLayer = Variable(y_cpLayer)
    if (cuda):
        y_cpLayer = y_cpLayer.cuda()
    cp = []
    for n in range(0, batchSize):
        cp.append([])
        for ite in range(0, ncp):
            i = random.randint(0, imageSize - 1)
            j = random.randint(0, imageSize - 1)
            cp[n].append([dems[n, 0, i, j], i, j])  # extract dem control point function
    for i in range(0, batchSize):
        for _cp in cp[i]:
            h = _cp[0]
            x = _cp[1]
            y = _cp[2]
            y_cpLayer[i, 0, x, y] = h
    return y_cpLayer


###########   LOSS & OPTIMIZER   ##########
criterion = nn.BCELoss(reduction='mean')
optimizerD = torch.optim.Adam(netD.parameters(), lr=lr, betas=(beta1, 0.999))
optimizerG = torch.optim.Adam(netG.parameters(), lr=lr, betas=(beta1, 0.999))

criterionDEM = nn.MSELoss(reduction='mean')

##########   GLOBAL VARIABLES   ###########
cpLayer = torch.FloatTensor(batchSize, nc, imageSize,imageSize).zero_()  # all ncp control points in one layer bs*nc*imagesize*imagesize for D
dems = torch.FloatTensor(batchSize, nc, imageSize, imageSize)  # real data for D
cpLayer = Variable(cpLayer)
dems = Variable(dems)

reallabel = torch.FloatTensor(batchSize)
fakelabel = torch.FloatTensor(batchSize)
real_label = 1
fake_label = 0
reallabel = Variable(reallabel)
fakelabel = Variable(fakelabel)
reallabel.data.resize_(batchSize).fill_(real_label)
fakelabel.data.resize_(batchSize).fill_(fake_label)

if (cuda):
    cpLayer = cpLayer.cuda()
    dems = dems.cuda()
    reallabel = reallabel.cuda()
    fakelabel = fakelabel.cuda()

if __name__ == "__main__":
########### Training   ###########
    netD.eval()
    netG.eval()
    with torch.no_grad():
        for epoch in range(npre + 1, npre + niter + 1):
            for i, (images, _) in enumerate(loader):
                errlog = open(logfile, 'a')
                ########### fDx ###########
                dems.data.copy_(images[:, 0, :, :].view(dems.shape))  # 使用dems.data.copy_()将第一通道的数据复制到dems的数据结构中
                cpLayer = ControlPointsImage_random(dems, ncp)
                for k in range(0, nk):


                    # train with real data, resize real because last batch may has less than
                    output = netD(dems, cpLayer)  # input real images6450 and cpLayer both bs*nc*imagesize*imagesize
                    errD_real = criterion(output, reallabel)


                    # train with fake data
                    fake = netG(cpLayer)
                    # detach gradients here so that gradients of G won't be updated
                    output = netD(fake.detach(), cpLayer)  # input fake images6450 and cpLayer both bs*nc*imagesize*imagesize
                    errD_fake = criterion(output, fakelabel)

                    errD = errD_fake + errD_real


                ########### fGx ###########

                output = netD(fake, cpLayer)
                errG = criterion(output, reallabel)

                if (i == 0):
                    ########### Logging #########
                    vutils.save_image(fake.data, '%s/cpt64100/epoch_%03d_batch_%03d_fake.png' % (outf, epoch, i),
                                      normalize=True)
                    vutils.save_image(dems.data, '%s/cpt64100/epoch_%03d_batch_%03d_real.png' % (outf, epoch, i),
                                      normalize=True)
                    vutils.save_image(cpLayer.data, '%s/cpt64100/epoch_%03d_batch_%03d_cp.png' % (outf, epoch, i),
                                     normalize=True)
                # if (i % 10 == 0):
                    ########### Logging #########
                    dems.data.copy_(L_ele + (dems.data / 2 + 0.5) * (H_ele - L_ele))
                    fake.data.copy_(L_ele + (fake.data / 2 + 0.5) * (H_ele - L_ele))
                    errDem = criterionDEM(fake, dems)


                    errlog.write('[%d/%d][%d/%d],Loss_D: %.4f,Loss_G: %.4f,MSE: %.2f\n'
                                 % (epoch, npre + niter, i, len(loader),
                                    errD.item(), errG.item(), errDem.item()))
                    print('[%d/%d][%d/%d],Loss_D: %.4f,Loss_G: %.4f,MSE: %.2f'
                          % (epoch, npre + niter, i, len(loader),
                             errD.item(), errG.item(), errDem.item()))
                errlog.close()

