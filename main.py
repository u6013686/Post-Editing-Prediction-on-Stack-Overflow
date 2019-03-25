import argparse
from model import train_model
from prediction import predict

parser = argparse.ArgumentParser()
parser.add_argument('--pattern', type=str, default='content',
	                    help='edit type. Including image, content, link, format.  Default is content.')
parser.add_argument('--batchsize', type=int, default=64,
	                    help='batch size. Default is 64.')
parser.add_argument('--epoch', type=int, default=10,
	                    help='epoch of training. Default is 10.')
args = parser.parse_args()


train_model(args.pattern, args.batchsize, args.epoch)
predict(args.pattern)

