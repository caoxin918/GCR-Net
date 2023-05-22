import torch.nn as nn
from model import DeepGCN
from data import *
import argparse


def predict():
    parser = argparse.ArgumentParser(description='CLT based on 3D Graph Convolution')
    parser.add_argument('--use_sgd', type=bool, default=False,
                        help='Use SGD')
    parser.add_argument('--lr', type=float, default=0.001, metavar='LR',
                        help='learning rate (default: 0.001, 0.1 if using sgd)')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='SGD momentum (default: 0.9)')
    parser.add_argument('--no_cuda', type=bool, default=False,
                        help='enables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--eval', type=bool, default=False,
                        help='evaluate the model')
    parser.add_argument('--dropout', type=float, default=0.5,
                        help='dropout rate')
    args = parser.parse_args()

    args.cuda = not args.no_cuda and torch.cuda.is_available()

    torch.manual_seed(args.seed)
    device = torch.device("cuda" if args.cuda else "cpu")
    print(device)

    model = DeepGCN(args).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)

    model_CKPT = torch.load(r"the path of your saved model")
    model_refine = {}
    for item in model_CKPT['model'].keys():
        newkey = str(item)[7:]
        model_refine[newkey] = model_CKPT['model'][item]

    model.load_state_dict(model_refine)

    model.eval()

    x = np.load(r"the path of your testing data")
    y = np.load(r"the path of your testing data")

    num_nodes = 4626
    x_data = torch.from_numpy(x[0:4,:,:]).to(device).float().reshape(4,num_nodes, 4)
    y = torch.from_numpy(y[0:4,:]).to(device).to(torch.int64).reshape(4, num_nodes)

    preds = model(x_data).reshape(4, num_nodes)

    criteria = nn.MSELoss()
    loss = criteria(preds, y)

    preds = preds.detach().cpu().numpy()
    y = y.cpu().numpy()

    outstr = 'loss: %.6f' % (loss)
    print(outstr)

    np.save(r"the path to save testing result", preds)
    np.save(r"the path to save real result", y)


predict()
