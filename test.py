from PIL import Image
from torchvision.utils import make_grid, save_image
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
from torch.utils.data import Dataset
from torchvision import transforms
from torch.utils.data import DataLoader
from cycle_gan import Discriminator, Generator, get_gen_loss, get_disc_loss
