import torch.nn as nn
from torch.utils.data import DataLoader
from model import DeepGCN
from data import *
from tqdm import tqdm
import argparse
import torch.optim as optim


def main():
    parser = argparse.ArgumentParser(description='CLT based on 3D Graph Convolution')
    parser.add_argument('--epochs', type=int, default=300, metavar='N',
                        help='number of episode to train ')
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

    # Try to load models
    model = DeepGCN(args).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    criteria = nn.MSELoss()


    scheduler = optim.lr_scheduler.StepLR(opt, step_size=500, gamma=0.92)
    print(str(model))

    model = nn.DataParallel(model)


    train_loader = DataLoader(dataset=CltDataset(),
                              batch_size=4,
                              shuffle=True)
    validate_loader = DataLoader(dataset=CltValidateDataset(),
                             batch_size=4,
                             shuffle=True)
    num_node = 4626

    train_tqdm = tqdm(train_loader)
    validate_tqdm = tqdm(validate_loader)
    loss_train = []
    loss_validate = []

    for epoch in range(0, args.epochs):
        print("epoch %s/n" % str(epoch+1))

        scheduler.step()
        model.train()

        train_loss = 0.0
        count = 0.0
        for data, label in train_tqdm:
            data,label = data.to(device),label.to(device)
            num_per_batch = list(data.size())[0]
            data = data.reshape(num_per_batch,num_node,4)
            opt.zero_grad()
            train_pred = model(data).reshape(num_per_batch,num_node)

            loss = criteria(train_pred, label)
            loss.backward()
            opt.step()
            count += num_per_batch
            train_loss += loss.item() * num_per_batch
        outstr = 'Train %d, loss: %.6f' % (epoch,train_loss * 1.0 / count)
        # loss_record
        loss_train.append(train_loss * 1.0 / count)
        print(outstr)

        validate_loss = 0.0
        count = 0.0
        for data, label in validate_tqdm:
            data,label = data.to(device),label.to(device)
            num_per_batch = list(data.size())[0]
            data = data.reshape(num_per_batch, num_node,4)
            model.eval()
            validate_pred = model(data).reshape(num_per_batch,num_node)

            loss = criteria(validate_pred, label)
            count += num_per_batch
            validate_loss += loss.item() * num_per_batch
        outstr = 'Validate %d, loss: %.6f' % (epoch, validate_loss * 1.0 / count)
        loss_validate.append(validate_loss * 1.0 / count)
        print(outstr)

        if (epoch + 1) % 10 == 0:
            state = { 'model': model.state_dict(), 'optimizer':opt.state_dict(), 'epoch': epoch }
            torch.save(state, 'the path to save your model')
            np.save("the path to save the training loss", loss_train)
            np.save("the path to save the validating loss", loss_validate)
            print("model already saved")


if __name__ == "__main__":
    main()
