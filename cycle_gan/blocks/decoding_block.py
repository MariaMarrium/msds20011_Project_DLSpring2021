from torch import nn


class DecodingBlock(nn.Module):
    '''
    DecodingBlock Class:
    Performs a convolutional transpose operation in order to upsample, 
    with an optional instance norm
    Values:
      input_channels: the number of channels to expect from a given input
    '''

    def __init__(self, input_channels, kernel_size=3, use_bn=True, activation='relu'):
        super(DecodingBlock, self).__init__()
        self.conv1 = nn.ConvTranspose2d(
            input_channels, input_channels // 2, kernel_size=3, stride=2, padding=1, output_padding=1)
        if use_bn:
            self.instancenorm = nn.InstanceNorm2d(input_channels // 2)
        self.use_bn = use_bn
        if activation == 'relu':
            self.activation = nn.ReLU()
        else:
            self.activation = nn.LeakyReLU(negative_slope=0.2)

    def forward(self, x):
        '''
        Function for completing a forward pass of DecodingBlock: 
        Given an image tensor, completes an Decoding Block and returns the transformed tensor.
        Parameters:
          x: image tensor of shape (batch size, channels, height, width)
          skip_con_x: the image tensor from the contracting path (from the opposing block of x)
          for the skip connection
        '''
        x = self.conv1(x)
        if self.use_bn:
            x = self.instancenorm(x)
        x = self.activation(x)
        return x
