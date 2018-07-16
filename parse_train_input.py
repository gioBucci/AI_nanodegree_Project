import argparse
import os

def parse_train_input():

    parser = argparse.ArgumentParser(description="Use a pretrained neural network to classify images." +
                                         " Basic usage: python train.py data_directory." +
                                         " Output: training loss, validation loss, and validation accuracy as the network trains. ")

    parser.add_argument("data_dir",
                            help="Directory where the image-data is stored")

    parser.add_argument("--save_dir",
                            help='Set directory to save checkpoints',
                            default = 'checkpoints')
    
    parser.add_argument("--arch",
                            help='Choose architecture of pretrained network',
                            default = 'vgg16', choices=['vgg11', 'vgg13', 'vgg16', 'vgg19'])
    
    parser.add_argument("--learning_rate",
                            help='Learning rate',
                            default = '0.001', type=float)
    
    parser.add_argument("--hidden_units_1",
                            help='Units of the first hidden layer',
                            default = '15000', type=int)

    parser.add_argument("--hidden_units_2",
                            help='Units of the second hidden layer',
                            default = '10000', type=int)
    
    parser.add_argument("--epochs",
                            help='Epochs',
                            default = '1', type=int)
    
    parser.add_argument("--gpu", help="use GPU for training",
                            action="store_true")
    
    args = parser.parse_args()

    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
    
    print("The algorithm will use a pretrained VGG network to classify the images located in the directory {}".format(args.data_dir))
    print("Checkpoints of the trained model will be saved in the directory {}".format(args.save_dir))
    print("The model is based on a pretrained {} net".format(args.arch))
    print("Hyperparameters:")
    print("learning rate: {}, hidden units in first layer: {}, hidden units in second layer: {}, epochs: {}"
              .format(args.learning_rate, args.hidden_units_1, args.hidden_units_2, args.epochs))
    if args.gpu:
        print("If available GPU will be used for training and validation")

    return args
    
