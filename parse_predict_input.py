import argparse
import os

def parse_predict_input():

    parser = argparse.ArgumentParser(description="Predict flower name and probability from an image.")

    parser.add_argument("path_to_image",
                            help="Path to the flower image you want to classify")
    
    parser.add_argument("path_to_checkpoint",
                            help="Path to the checkpoint of the model used for classification")

    parser.add_argument("--top_k",
                            help='Number of most likely classes',
                            default = '3', type=int)
    
    parser.add_argument("--category_names",
                            help='json file containing a dictionary of cathegory names')
    
    parser.add_argument("--gpu",
                            help="use GPU for inference",
                            action="store_true")
    
    args = parser.parse_args()
    
    print("The algorithm will load a trained model to classify the image {}".format(args.path_to_image))
    print("The checkpoint of the trained model is saved as {}".format(args.path_to_checkpoint))
    print("The top {} cathegories will be listed along with their probabilities".format(args.top_k))
    if args.gpu:
        print("If available GPU will be used for training")

    return args
    
