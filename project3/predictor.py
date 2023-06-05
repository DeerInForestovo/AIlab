from tools import Net
import tools
import os  # check if the file exists
import torch
import numpy as np

if __name__ == "__main__":
    csv_file_path = './data/testdata.csv'
    dict_file_path = 'string_argument_dict.npy'
    net_file_path = 'AdultCensusIncomeNet.pt'
    if os.path.isfile(csv_file_path) and os.path.isfile(dict_file_path) and os.path.isfile(net_file_path):
        # read from files
        census_data = tools.read_csv(csv_file_path)

        # load dict
        string_argument_dict = np.load(dict_file_path, allow_pickle=True).item()

        # process columns
        census_data = tools.process_columns(census_data=census_data, string_argument_dict=string_argument_dict)

        # get test data
        test_data = census_data.values

        # change into torch tensor
        test_data = torch.tensor(test_data).type(torch.FloatTensor)

        # load net
        net = Net(n_feature=13, n_hidden=78, n_output=1)  # define the network
        state_dict = torch.load(net_file_path)
        net.load_state_dict(state_dict)

        # predict
        test_out = torch.round(net(test_data)).squeeze(1).detach().numpy()

        # save
        prediction_file_path = 'testlabel.txt'
        with open(prediction_file_path, 'w') as f:
            for prediction in test_out:
                f.write(str(int(prediction)) + '\n')

    else:
        if os.path.isfile(csv_file_path):
            print("please train the module first")
        else:
            print("please put the csv file in the right place")

