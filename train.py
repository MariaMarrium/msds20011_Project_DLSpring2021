import os
import re
import glob
import torch
import argparse
from PIL import Image
from torchvision.utils import make_grid, save_image
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
from torch.utils.data import Dataset
from torchvision import transforms
from torch.utils.data import DataLoader
from cycle_gan import Discriminator, Generator, get_gen_loss, get_disc_loss
from utils import visualizer


class ImageDataset(object):
    def __init__(self, root, A, B, transform=None):
        self.transform = transform
        self.domain_A = sorted(
            glob.glob(os.path.join(root, '%s' % A) + '/*.*'))
        self.domain_B = sorted(
            glob.glob(os.path.join(root, '%s' % B) + '/*.*'))
        print(len(self.domain_A), len(self.domain_B))
        if len(self.domain_A) > len(self.domain_B):
            self.domain_A, self.domain_B = self.domain_B, self.domain_A
        self.new_perm()
        assert len(self.domain_A) > 0

    def new_perm(self):
        self.randperm = torch.randperm(len(self.domain_B))[:len(self.domain_A)]

    def __getitem__(self, index):
        item_A = self.transform(Image.open(
            self.domain_A[index % len(self.domain_A)]))
        item_B = self.transform(Image.open(
            self.domain_B[self.randperm[index]]))
        if item_A.shape[0] != 3:
            item_A = item_A.repeat(3, 1, 1)
        if item_B.shape[0] != 3:
            item_B = item_B.repeat(3, 1, 1)
        if index == len(self) - 1:
            self.new_perm()
        # Old versions of PyTorch didn't support normalization for different-channeled images
        return (item_A - 0.5) * 2, (item_B - 0.5) * 2

    def __len__(self):
        return min(len(self.domain_A), len(self.domain_B))

def save_tensor_images(image_tensor, num_images=25, size=(1, 28, 28), fname='1.jpg'):
  '''
  Function for visualizing images: Given a tensor of images, number of images, and
  size per image, plots and prints the images in an uniform grid.
  '''
  image_tensor = (image_tensor + 1) / 2
  image_shifted = image_tensor
  image_unflat = image_shifted.detach().cpu().view(-1, *size)
  #image_grid = make_grid(image_unflat[:num_images], nrow=5)
  save_image(image_unflat, fname)


def weights_init(m):
    if isinstance(m, torch.nn.Conv2d) or isinstance(m, torch.nn.ConvTranspose2d):
        torch.nn.init.normal_(m.weight, 0.0, 0.02)
    if isinstance(m, torch.nn.BatchNorm2d):
        torch.nn.init.normal_(m.weight, 0.0, 0.02)
        torch.nn.init.constant_(m.bias, 0)

def train(epochs, adv_criterion, recon_criterion, target_shape=128, display_step=200, save_model=True):
    adv_criterion = torch.nn.MSELoss()
    recon_criterion = torch.nn.L1Loss()
    mean_generator_loss = 0
    mean_discriminator_loss = 0

    cur_step = 0
    generator_loss = []
    disc_loss = []

    for epoch in range(epochs):
        for real_A, real_B in tqdm(dataloader):
            # image_width = image.shape[3]
            real_A = torch.nn.functional.interpolate(real_A, size=target_shape)
            real_B = torch.nn.functional.interpolate(real_B, size=target_shape)

            cur_batch_size = len(real_A)
            real_A = real_A.to(device)
            real_B = real_B.to(device)
            # Update discriminator A
            disc_A_opt.zero_grad()  # Zero out the gradient before backpropagation

            with torch.no_grad():
                fake_A = gen_BA(real_B)
            disc_A_loss = get_disc_loss(real_A, fake_A, disc_A, adv_criterion)
            disc_A_loss.backward(retain_graph=True)  # Update gradients
            disc_A_opt.step()  # Update optimizer

            ### Update discriminator B ###
            disc_B_opt.zero_grad()  # Zero out the gradient before backpropagation
            with torch.no_grad():
                fake_B = gen_AB(real_A)
            disc_B_loss = get_disc_loss(real_B, fake_B, disc_B, adv_criterion)
            disc_B_loss.backward(retain_graph=True)  # Update gradients
            disc_B_opt.step()  # Update optimizer

            ### Update generator ###
            gen_opt.zero_grad()
            gen_loss, fake_A, fake_B = get_gen_loss(
                real_A, real_B, gen_AB, gen_BA, disc_A, disc_B, adv_criterion, recon_criterion, recon_criterion
            )
            gen_loss.backward()  # Update gradients
            gen_opt.step()  # Update optimizer

            # Keep track of the average discriminator loss
            mean_discriminator_loss += disc_A_loss.item() / display_step
            # Keep track of the average generator loss
            mean_generator_loss += gen_loss.item() / display_step
            ### Visualization code ###
            if cur_step % display_step == 0:
                print(
                    f"Epoch {epoch}: Step {cur_step}: Generator (U-Net) loss: {mean_generator_loss}, Discriminator loss: {mean_discriminator_loss}")
                save_tensor_images(real_A, size=(
                    args.dim_A, target_shape, target_shape), fname=f'real_A{epoch}_{cur_step}.jpg')
                save_tensor_images(real_B, size=(
                    args.dim_A, target_shape, target_shape), fname=f'real_B{epoch}_{cur_step}.jpg')
                save_tensor_images(fake_A, size=(
                    args.dim_A, target_shape, target_shape), fname=f'fake_A{epoch}_{cur_step}.jpg')
                save_tensor_images(fake_B, size=(
                    args.dim_A, target_shape, target_shape), fname=f'fake_B{epoch}_{cur_step}.jpg')
                
                # show_tensor_images(torch.cat([fake_B, fake_A]), size=(
                #     args.dim_B, target_shape, target_shape))
                # mean_generator_loss = 0
                mean_discriminator_loss = 0
                # You can change save_model to True if you'd like to save the model
                if save_model:
                    torch.save({
                        'gen_AB': gen_AB.state_dict(),
                        'gen_BA': gen_BA.state_dict(),
                        'gen_opt': gen_opt.state_dict(),
                        'disc_A': disc_A.state_dict(),
                        'disc_A_opt': disc_A_opt.state_dict(),
                        'disc_B': disc_B.state_dict(),
                        'disc_B_opt': disc_B_opt.state_dict()
                    }, f"a{cur_step}.pth")
            cur_step += 1
        generator_loss.append(mean_generator_loss)
        disc_loss.append(mean_discriminator_loss)
    return generator_loss, disc_loss


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str,
                        default='joy2sadness', help='model you want to train')
    parser.add_argument('--dataroot', type=str,
                        default='dataset/Emotion6/images', help='dataset directory path')
    parser.add_argument('--batch_size', type=int,
                        default=1, help='batch size')
    parser.add_argument('--epochs', type=int,
                        default=20, help='tarining epochs')
    parser.add_argument('--lr', type=int,
                        default=0.001, help='tarining epochs')
    parser.add_argument('--model_path', type=str,
                        default='checkpoints', help='tarining epochs')
    parser.add_argument('--target_size', type=int,
                        default=128, help='image resize')
    parser.add_argument('--load_size', type=int,
                        default=128, help='image load size')
    parser.add_argument('--dim_A', type=int,
                        default=3, help='domain A image dimesions')
    parser.add_argument('--dim_B', type=int,
                        default=3, help='domain B image dimesions')
    parser.add_argument('--betas', type=tuple,
                        default=(0.5, 0.999), help='running average co-efficient')
    parser.add_argument('--loss', type=str,
                        default='mse', help='loss function')
    
    args = parser.parse_args()
    emotions = re.findall('[A-Za-z]+', args.model)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    transform = transforms.Compose([
        transforms.Resize((args.target_size, args.target_size)),
        transforms.ToTensor()
    ])  

    print(args.dataroot)
    dataset = ImageDataset(
        args.dataroot, transform=transform, A=emotions[0], B=emotions[1])

    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
    gen_AB = Generator(args.dim_A, args.dim_B).to(device)
    gen_BA = Generator(args.dim_B, args.dim_A).to(device)
    gen_opt = torch.optim.Adam(
        list(gen_AB.parameters()) + list(gen_BA.parameters()), lr=args.lr, betas=args.betas)
    disc_A = Discriminator(args.dim_A).to(device)
    disc_A_opt = torch.optim.Adam(
        disc_A.parameters(), lr=args.lr, betas=args.betas)
    disc_B = Discriminator(args.dim_B).to(device)
    disc_B_opt = torch.optim.Adam(
        disc_B.parameters(), lr=args.lr, betas=args.betas)

    gen_AB = gen_AB.apply(weights_init)
    gen_BA = gen_BA.apply(weights_init)
    disc_A = disc_A.apply(weights_init)
    disc_B = disc_B.apply(weights_init)

    adv_criterion = torch.nn.MSELoss()
    recon_criterion = torch.nn.L1Loss()
    
    generator_loss, disc_loss = train(args.epochs, adv_criterion, recon_criterion)
