from utils import *
import torch
import sys
from torchvision import transforms
import torchvision.datasets as dset
from gan_build import Model
from torchvision import transforms
import argparse
from torch.utils.data import TensorDataset, DataLoader
from custom_dataloader import *

dataset_directory = "fuit-gan/dataset/fruits-360_dataset/fruits-360/Training/"
train = []

FLAGS = None



def main():
    device = torch.device("cuda:0" if FLAGS.cuda else "cpu")
    # device = "cpu"
    if FLAGS.train:
        print("Loading data...\n")

        # list_photos = read_dataset_photos(FLAGS.data_dir)
        # dataset = CustomDataset(transform= transforms.Compose([transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))]))
        dataset = CustomDataset()
     
        # dataset = dset.MNIST(root=FLAGS.data_dir, download=True,
        # transform=transforms.Compose([
        # transforms.Resize(FLAGS.img_size),
        # transforms.ToTensor(),
        # transforms.Normalize((0.5,), (0.5,))
        # ]))
        assert dataset

        dataloader = torch.utils.data.DataLoader(dataset,
            batch_size=FLAGS.batch_size,
            shuffle=True,
            num_workers=4, pin_memory=True)

        print('Creating model...\n')
        model = Model(FLAGS.model, device, dataloader, FLAGS.classes,FLAGS.channels, FLAGS.img_size, FLAGS.latent_dim)
        model.create_optim(FLAGS.lr)
        
        # Train
        print('Training model...\n')
        model.train(FLAGS.epochs, FLAGS.log_interval, FLAGS.out_dir, True)
        # model.save_to('')
    else:
        model = Model(FLAGS.model, device, None, FLAGS.classes,FLAGS.channels, FLAGS.img_size, FLAGS.latent_dim)
        model.load_from(FLAGS.out_dir)
        model.eval(mode=0, batch_size=FLAGS.batch_size)



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Fruit Gan')
    parser.add_argument('--cuda', type=bool, default=False,help='enable CUDA.')
    parser.add_argument('--train', type=bool, default=True,help='train mode or eval mode.')
    parser.add_argument('--data_dir', type=str, default='fuit-gan/dataset/fruits-360_dataset/fruits-360/Test/',help='Directory for dataset.')
    parser.add_argument('--out_dir', type=str, default='fuit-gan\\output',help='Directory for output.')
    parser.add_argument('--epochs', type=int, default=10, help='number of epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='sizeof batches')
    parser.add_argument('--lr', type=float, default=0.0002, help='learning rate')
    parser.add_argument('--latent_dim', type=int, default=100, help='latent space dimension')
    parser.add_argument('--classes', type=int, default=10, help='number of classes')
    parser.add_argument('--channels', type=int, default=1, help='number of image channel')
    parser.add_argument('--log_interval', type=int, default=10, help='interval between logging and image sampling')
    parser.add_argument('--seed', type=int, default=1, help='random seed')
    

    FLAGS = parser.parse_args()
    FLAGS.img_size = 45
    FLAGS.model = "cgan"
    FLAGS.cuda = True
    # FLAGS.device = "cpu"

    if FLAGS.seed is not None:
        torch.manual_seed(FLAGS.seed)
    if FLAGS.cuda:
        torch.cuda.manual_seed(FLAGS.seed)
        np.random.seed(FLAGS.seed)
        torch.backends.cudnn.benchmark = False
    # if FLAGS.train:
    #     utils.clear_folder(FLAGS.out_dir)
    # log_file = os.path.join(FLAGS.out_dir, 'log.txt')
    # print("Logging to {}\n".format(log_file))
    # sys.stdout = utils.StdOut(log_file)


    print("PyTorch version: {}".format(torch.__version__))
    print("CUDA version: {}\n".format(torch.version.cuda))
    print(" " * 9 + "Args" + " " * 9 + "| " + "Type" + " | " + "Value")
    print("-" * 50)
    for arg in vars(FLAGS):
        arg_str = str(arg)
        var_str = str(getattr(FLAGS, arg))
        type_str = str(type(getattr(FLAGS, arg)).__name__)
        print(" " + arg_str + " " * (20-len(arg_str)) + "|" + \
        " " + type_str + " " * (10-len(type_str)) + "|" + \
        " " + var_str)
    
    main()


