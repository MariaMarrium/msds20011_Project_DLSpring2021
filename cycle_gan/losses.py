import torch
import torch.nn.functional as F


class GPLoss(torch.nn.Module):
    def __init__(self):
        super(GPLoss, self).__init__()
        self.trace = SPLoss()

    def get_image_gradients(self, input):
        f_v_1 = F.pad(input, (0, -1, 0, 0))
        f_v_2 = F.pad(input, (-1, 0, 0, 0))
        f_v = f_v_1-f_v_2

        f_h_1 = F.pad(input, (0, 0, 0, -1))
        f_h_2 = F.pad(input, (0, 0, -1, 0))
        f_h = f_h_1-f_h_2

        return f_v, f_h

    def __call__(self, input, reference):
        # comment these lines when you inputs and outputs are in [0,1] range already
        input = (input+1)/2
        reference = (reference+1)/2

        input_v, input_h = self.get_image_gradients(input)
        ref_v, ref_h = self.get_image_gradients(reference)

        trace_v = self.trace(input_v, ref_v)
        trace_h = self.trace(input_h, ref_h)
        return trace_v + trace_h


class CPLoss(torch.nn.Module):
    def __init__(self, rgb=True, yuv=True, yuvgrad=True):
        super(CPLoss, self).__init__()
        self.rgb = rgb
        self.yuv = yuv
        self.yuvgrad = yuvgrad
        self.trace = SPLoss()
        self.trace_YUV = SPLoss()

    def get_image_gradients(self, input):
        f_v_1 = F.pad(input, (0, -1, 0, 0))
        f_v_2 = F.pad(input, (-1, 0, 0, 0))
        f_v = f_v_1-f_v_2

        f_h_1 = F.pad(input, (0, 0, 0, -1))
        f_h_2 = F.pad(input, (0, 0, -1, 0))
        f_h = f_h_1-f_h_2

        return f_v, f_h

    def to_YUV(self, input):
        return torch.cat((0.299*input[:, 0, :, :].unsqueeze(1)+0.587*input[:, 1, :, :].unsqueeze(1)+0.114*input[:, 2, :, :].unsqueeze(1),
                          0.493*(input[:, 2, :, :].unsqueeze(1)-(0.299*input[:, 0, :, :].unsqueeze(
                              1)+0.587*input[:, 1, :, :].unsqueeze(1)+0.114*input[:, 2, :, :].unsqueeze(1))),
                          0.877*(input[:, 0, :, :].unsqueeze(1)-(0.299*input[:, 0, :, :].unsqueeze(1)+0.587*input[:, 1, :, :].unsqueeze(1)+0.114*input[:, 2, :, :].unsqueeze(1)))), dim=1)

    def __call__(self, input, reference):
        # comment these lines when you inputs and outputs are in [0,1] range already
        input = (input+1)/2
        reference = (reference+1)/2
        total_loss = 0
        if self.rgb:
            total_loss += self.trace(input, reference)
        if self.yuv:
            input_yuv = self.to_YUV(input)
            reference_yuv = self.to_YUV(reference)
            total_loss += self.trace(input_yuv, reference_yuv)
        if self.yuvgrad:
            input_v, input_h = self.get_image_gradients(input_yuv)
            ref_v, ref_h = self.get_image_gradients(reference_yuv)

            total_loss += self.trace(input_v, ref_v)
            total_loss += self.trace(input_h, ref_h)

        return total_loss


class SPL_ComputeWithTrace(torch.nn.Module):
    """
    Slow implementation of the trace loss using the same formula as stated in the paper.
    """

    def __init__(self, weight=[1., 1., 1.]):
        super(SPL_ComputeWithTrace, self).__init__()
        self.weight = weight

    def __call__(self, input, reference):
        a = 0
        b = 0
        for i in range(input.shape[0]):
            for j in range(input.shape[1]):
                a += torch.trace(torch.matmul(F.normalize(input[i, j, :, :], p=2, dim=1), torch.t(
                    F.normalize(reference[i, j, :, :], p=2, dim=1))))/input.shape[2]*self.weight[j]
                b += torch.trace(torch.matmul(torch.t(F.normalize(input[i, j, :, :], p=2, dim=0)), F.normalize(
                    reference[i, j, :, :], p=2, dim=0)))/input.shape[3]*self.weight[j]
        a = -torch.sum(a)/input.shape[0]
        b = -torch.sum(b)/input.shape[0]
        return a+b


class SPLoss(torch.nn.Module):
    def __init__(self):
        super(SPLoss, self).__init__()

    def __call__(self, input, reference):
        a = torch.sum(torch.sum(F.normalize(input, p=2, dim=2) *
                      F.normalize(reference, p=2, dim=2), dim=2, keepdim=True))
        b = torch.sum(torch.sum(F.normalize(input, p=2, dim=3) *
                      F.normalize(reference, p=2, dim=3), dim=3, keepdim=True))
        return -(a + b) / input.size(2)


def get_gen_adversarial_loss(real_X, disc_Y, gen_XY, adv_criterion):
    '''
    Return the adversarial loss of the generator given inputs
    (and the generated images for testing purposes).
    Parameters:
      real_X: the real images from pile X
      disc_Y: the discriminator for class Y; takes images and returns real/fake class Y
          prediction matrices
      gen_XY: the generator for class X to Y; takes images and returns the images 
          transformed to class Y
      adv_criterion: the adversarial loss function; takes the discriminator 
                predictions and the target labels and returns a adversarial 
                loss (which you aim to minimize)
    '''
    fake_Y = gen_XY(real_X)
    disc_fake_Y_hat = disc_Y(fake_Y)
    adversarial_loss = adv_criterion(
        disc_fake_Y_hat, torch.ones_like(disc_fake_Y_hat))

    return adversarial_loss, fake_Y


def get_identity_loss(real_X, gen_YX, identity_criterion):
    '''
    Return the identity loss of the generator given inputs
    (and the generated images for testing purposes).
    Parameters:
      real_X: the real images from pile X
      gen_YX: the generator for class Y to X; takes images and returns the images 
          transformed to class X
      identity_criterion: the identity loss function; takes the real images from X and
                      those images put through a Y->X generator and returns the identity 
                      loss (which you aim to minimize)
    '''
    identity_X = gen_YX(real_X)
    identity_loss = identity_criterion(identity_X, real_X)

    return identity_loss, identity_X


def get_cycle_consistency_loss(real_X, fake_Y, gen_YX, cycle_criterion):
    '''
    Return the cycle consistency loss of the generator given inputs
    (and the generated images for testing purposes).
    Parameters:
      real_X: the real images from pile X
      fake_Y: the generated images of class Y
      gen_YX: the generator for class Y to X; takes images and returns the images 
              transformed to class X
      cycle_criterion: the cycle consistency loss function; takes the real images from X and
                      those images put through a X->Y generator and then Y->X generator
                      and returns the cycle consistency loss (which you aim to minimize)
    '''

    cycle_X = gen_YX(fake_Y)
    cycle_loss = cycle_criterion(cycle_X, real_X)

    return cycle_loss, cycle_X


def SPL_Loss(target, generated):
    # Gradient Profile Loss
    GPL = GPLoss()

    # Color Profile Loss
    # You can define the desired color spaces in the initialization
    # default is True for all
    CPL = CPLoss(rgb=True, yuv=True, yuvgrad=True)

    gpl_value = GPL(generated, target)
    cpl_value = CPL(generated, target)

    spl_value = gpl_value + cpl_value
    return spl_value


def get_gen_loss(real_A, real_B, gen_AB, gen_BA, disc_A, disc_B, adv_criterion, identity_criterion, cycle_criterion, lambda_identity=0.1, lambda_cycle=10):
    '''
    Return the loss of the generator given inputs.
    Parameters:
      real_A: the real images from pile A
      real_B: the real images from pile B
      gen_AB: the generator for class A to B; takes images and returns the images 
          transformed to class B
      gen_BA: the generator for class B to A; takes images and returns the images 
          transformed to class A
      disc_A: the discriminator for class A; takes images and returns real/fake class A
          prediction matrices
      disc_B: the discriminator for class B; takes images and returns real/fake class B
          prediction matrices
      adv_criterion: the adversarial loss function; takes the discriminator 
          predictions and the true labels and returns a adversarial 
          loss (which you aim to minimize)
      identity_criterion: the reconstruction loss function used for identity loss
          and cycle consistency loss; takes two sets of images and returns
          their pixel differences (which you aim to minimize)
      cycle_criterion: the cycle consistency loss function; takes the real images from X and
          those images put through a X->Y generator and then Y->X generator
          and returns the cycle consistency loss (which you aim to minimize).
          Note that in practice, cycle_criterion == identity_criterion == L1 loss
      lambda_identity: the weight of the identity loss
      lambda_cycle: the weight of the cycle-consistency loss
  '''
    fake_A = gen_BA(real_B)
    fake_B = gen_AB(real_A)
    # Adversarial Loss -- get_gen_adversarial_loss(real_X, disc_Y, gen_XY, adv_criterion)
    adv_loss_BA = SPL_Loss(fake_B, real_A)
    adv_loss_AB = SPL_Loss(fake_A, real_B)
    gen_adversarial_loss = adv_loss_BA + adv_loss_AB

    # Identity Loss -- get_identity_loss(real_X, gen_YX, identity_criterion)
    identity_loss_A, identity_A = get_identity_loss(
        real_A, gen_BA, identity_criterion)
    identity_loss_B, identity_B = get_identity_loss(
        real_B, gen_AB, identity_criterion)
    gen_identity_loss = identity_loss_A + identity_loss_B

    # Cycle-consistency Loss -- get_cycle_consistency_loss(real_X, fake_Y, gen_YX, cycle_criterion)
    cycle_loss_BA, cycle_A = get_cycle_consistency_loss(
        real_A, fake_B, gen_BA, cycle_criterion)
    cycle_loss_AB, cycle_B = get_cycle_consistency_loss(
        real_B, fake_A, gen_AB, cycle_criterion)
    gen_cycle_loss = cycle_loss_BA + cycle_loss_AB

    gen_loss = lambda_identity * gen_identity_loss + \
        lambda_cycle * gen_cycle_loss + gen_adversarial_loss

    return gen_loss, fake_A, fake_B


def get_disc_loss(real_X, fake_X, disc_X, adv_criterion):
    '''
    Return the loss of the discriminator given inputs.
    Parameters:
      real_X: the real images from pile X
      fake_X: the generated images of class X
      disc_X: the discriminator for class X; takes images and returns real/fake class X
          prediction matrices
      adv_criterion: the adversarial loss function; takes the discriminator 
          predictions and the target labels and returns a adversarial 
          loss (which you aim to minimize)
    '''
    disc_fake_X_hat = disc_X(fake_X.detach())  # Detach generator
    disc_fake_X_loss = adv_criterion(
        disc_fake_X_hat, torch.zeros_like(disc_fake_X_hat))
    disc_real_X_hat = disc_X(real_X)
    disc_real_X_loss = adv_criterion(
        disc_real_X_hat, torch.ones_like(disc_real_X_hat))
    disc_loss = (disc_fake_X_loss + disc_real_X_loss) / 2

    return disc_loss
