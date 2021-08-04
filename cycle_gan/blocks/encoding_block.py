from torch import nn


class EncodingBlock(nn.Module):
    '''
    EncodingBlock Class
    Performs a convolution followed by a max pool operation and an optional instance norm.
    Values:
      input_channels: the number of channels to expect from a given input
    '''

    def __init__(self, input_channels, use_bn=True, kernel_size=3, activation='relu'):
        super(EncodingBlock, self).__init__()
        self.conv1 = nn.Conv2d(input_channels, input_channels * 2,
                               kernel_size=kernel_size, padding=1, stride=2, padding_mode='reflect')
        self.activation = nn.ReLU() if activation == 'relu' else nn.LeakyReLU(0.2)
        if use_bn:
            self.instancenorm = nn.InstanceNorm2d(input_channels * 2)
        self.use_bn = use_bn

    def forward(self, x):
        '''
        Function for completing a forward pass of ContractingBlock: 
        Given an image tensor, completes a contracting block and returns the transformed tensor.
        Parameters:
          x: image tensor of shape (batch size, channels, height, width)
        '''
        x = self.conv1(x)
        if self.use_bn:
            x = self.instancenorm(x)
        x = self.activation(x)
        return x
