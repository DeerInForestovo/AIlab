from tools import Net
import tools
import os  # check if the file exists
import torch
import numpy as np


# train a model and store in net.pt
def train(train_data, train_label, test_data, test_label, train_data_len, test_data_len):
    # codes below refers to:
    # https://github.com/MorvanZhou/PyTorch-Tutorial/blob/master/tutorial-contents/302_classification.py
    # and
    # https://machinelearningmastery.com/develop-your-first-neural-network-with-pytorch-step-by-step

    net = Net(n_feature=13, n_hidden=78, n_output=1)  # define the network

    net_file_path = 'AdultCensusIncomeNet.pt'
    if os.path.isfile(net_file_path):
        print("begin from last result")
        state_dict = torch.load(net_file_path)
        net.load_state_dict(state_dict)
    else:
        print("train a new one")

    loss_func = torch.nn.BCELoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=0.00001)

    EPOCH_SIZE = 500
    EPOCH_TOT = 1000
    for epoch in range(EPOCH_TOT):
        for _ in range(EPOCH_SIZE):
            # train
            out = net(train_data)  # input x and predict based on x
            out = out.squeeze(1)
            loss = loss_func(out, train_label)  # must be (1. nn output, 2. target), the target label is NOT one-hotted
            optimizer.zero_grad()  # clear gradients for next train
            loss.backward()  # backpropagation, compute gradients
            optimizer.step()  # apply gradients

        # test
        test_out = torch.round(net(test_data)).squeeze(1).detach().numpy()
        test_prediction = test_out - test_label
        point = len(np.where(test_prediction == 0)[0])
        print("epoch %d: accuracy %.4f" % (epoch, point / test_data_len))

        # save
        torch.save(net.state_dict(), net_file_path)


if __name__ == "__main__":
    # read from files
    census_data = tools.read_csv("./data/traindata.csv")
    train_label = tools.read_label("./data/trainlabel.txt")

    # make dict
    census_data['label'] = train_label
    string_argument_dict = tools.make_dict(census_data)

    # process columns
    census_data = tools.process_columns(census_data=census_data, string_argument_dict=string_argument_dict)

    # get train data
    train_data = census_data.drop(columns='label', axis=1).values
    train_label = census_data['label'].values

    # break train dataset into train dataset and validity test dataset
    train_data_len = len(train_data)
    assert train_data_len == len(train_label)
    train_data_len = int(train_data_len * 0.8)  # train : test = 8 : 2
    train_data, test_data = train_data[:train_data_len], train_data[train_data_len:]
    train_label, test_label = train_label[:train_data_len], train_label[train_data_len:]

    # re-calculate the length
    test_data_len = len(test_data)
    assert test_data_len == len(test_label)
    train_data_len = len(train_data)
    assert train_data_len == len(train_label)

    # change into torch tensor
    train_data = torch.tensor(train_data).type(torch.FloatTensor)
    test_data = torch.tensor(test_data).type(torch.FloatTensor)
    train_label = torch.tensor(train_label).type(torch.FloatTensor)
    # test_label = torch.tensor(test_label).type(torch.FloatTensor)

    train(train_data, train_label, test_data, test_label, train_data_len, test_data_len)

