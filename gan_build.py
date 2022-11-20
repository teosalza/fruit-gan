from utils import *
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
import os
import torch
import torchvision.utils as vutils
from CGanDiscr import Discriminator as cganD
from CGanGen import Generator as cganG
from random import randrange
from tqdm import tqdm



class   Model(object):
    def __init__(self,name,device,data_loader,classes,channels,img_size,latent_dim):
        self.name = name
        self.device = device
        self.data_loader = data_loader
        self.classes = classes
        self.channels = channels
        self.img_size = img_size
        self.latent_dim = latent_dim

        if self.name == 'cgan':
            self.netG = cganG(self.classes, self.channels,
            self.img_size, self.latent_dim)
            self.netG.to(self.device)

        if self.name == 'cgan':
            self.netD = cganD(self.classes, self.channels,
            self.img_size, self.latent_dim)
            self.netD.to(self.device)
            self.optim_G = None
            self.optim_D = None

    def create_optim(self, lr, alpha=0.5    , beta=0.999):
        self.optim_G = torch.optim.Adam(
            filter(lambda p: p.requires_grad,self.netG.parameters()),
            lr=lr,
            betas=(alpha, beta))
        self.optim_D = torch.optim.Adam(
            filter(lambda p: p.requires_grad,self.netD.parameters()),
            lr=lr,
            betas=(alpha, beta))

    def train(self,epochs,log_interval=100,out_dir='',verbose=True):
        self.netG.train()
        self.netD.train()
        viz_noise = torch.randn(self.data_loader.batch_size,self.latent_dim, device=self.device)
        nrows = self.data_loader.batch_size // 8
        viz_label_test = torch.LongTensor(np.array([num for _ in range(nrows) for num in range(self.classes)]))
        viz_label = torch.LongTensor(np.array([randrange(0,self.classes) for i in range(self.data_loader.batch_size)])).to(self.device)
        # viz_label = viz_label.to(torch.int16).to(self.device)
        # viz_label = (viz_label[:,None]).to(self.device)

        for epoch in tqdm(range(epochs)):
            for batch_idx, (data, target) in enumerate(self.data_loader):
                # data = data.squeeze()
                # target = target[:,None]
                data, target = data.to(self.device), target.to(self.device)
                data = data.type(torch.float32)
                batch_size = data.size(0)
                real_label = torch.full((batch_size, 1), 1.,device=self.device)
                fake_label = torch.full((batch_size, 1), 0.,device=self.device)

                # Train G
                self.netG.zero_grad()
                z_noise = torch.randn(batch_size, self.latent_dim,device=self.device)
                x_fake_labels = torch.randint(0, self.classes,(batch_size,), device=self.device)
                # x_fake_labels = x_fake_labels[:,None]
                x_fake = self.netG(z_noise, x_fake_labels)
                y_fake_g = self.netD(x_fake, x_fake_labels)
                g_loss = self.netD.loss(y_fake_g, real_label).to(self.device)
                g_loss.backward()
                self.optim_G.step()

                # Train D
                self.netD.zero_grad()
                y_real = self.netD(data, target)
                d_real_loss = self.netD.loss(y_real, real_label)
                y_fake_d = self.netD(x_fake.detach(), x_fake_labels).to(self.device)
                d_fake_loss = self.netD.loss(y_fake_d, fake_label)
                d_loss = (d_real_loss + d_fake_loss) / 2
                d_loss.backward()
                self.optim_D.step()

                if verbose and batch_idx % log_interval == 0 and batch_idx > 0:
                    print('Epoch {} [{}/{}] loss_D: {:.4f} loss_G:{:.4f}'.format(
                        epoch, batch_idx, 
                        len(self.data_loader),
                        d_loss.mean().item(),
                        g_loss.mean().item()))
                    vutils.save_image(data, os.path.join(out_dir,'real_samples.png'), normalize=True)
                    with torch.no_grad():
                        viz_sample = self.netG(viz_noise, viz_label)
                        vutils.save_image(viz_sample, os.path.join(out_dir,'fake_samples_{}.png'.format(epoch)), nrow=8, normalize=True)


            #end epoch save the model
            if epoch % 10 == 0:
                torch.save(self.netG.state_dict(), os.path.join("fuit-gan\\saved_models",
                            'netG_{}.pth'.format(epoch)))
                torch.save(self.netD.state_dict(), os.path.join("fuit-gan\\saved_models",
                            'netD_{}.pth'.format(epoch)))                             
            # self.save_to(path=out_dir, name=self.name, verbose=False)
        torch.save(self.netG.state_dict(), os.path.join("fuit-gan\\saved_models",
                            'netG_final.pth'))
        torch.save(self.netD.state_dict(), os.path.join("fuit-gan\\saved_models",
                            'netD_final.pth'))        







