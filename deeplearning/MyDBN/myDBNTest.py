# --coding:utf-8--
import torch
from torch import nn
from torch.utils.data import TensorDataset,  DataLoader, Dataset
import numpy as np
from RBM import RBM
import warnings

class DBN(nn.Module):

    def __init__(self, hidden_units,
                 input_size, k=2,
                 output_size=1,
                 learning_rate=0.01,
                 learning_rate_decay=False,
                 increase_to_cd_k=False,
                 device='cpu'
                 ):

        super(DBN, self).__init__()

        self.hidden_units = hidden_units
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.rbm_layers = []
        self.n_layers = len(hidden_units)
        # 创建多层玻尔兹曼机
        for i in range(self.n_layers):
            if i == 0:
                visible_units = input_size
            else:
                visible_units = hidden_units[i - 1]
            rbm = RBM(n_visible=visible_units, n_hidden=hidden_units[i],
                      k=k, learning_rate=learning_rate,
                      learning_rate_decay=learning_rate_decay,
                      increase_to_cd_k=increase_to_cd_k, device=device)
            self.rbm_layers.append(rbm)

        # self.W_rec = [self.rbm_layers[i].W for i in range(self.n_layers)]
        # self.bias_rec = [self.rbm_layers[i].hidden_bias for i in range(self.n_layers)]
        # for i in range(self.n_layers):
        #     self.register_parameter('W_rec%i' % i, self.W_rec[i])
        #     self.register_parameter('bias_rec%i' % i, self.bias_rec[i])

        self.bpnn = nn.Sequential(  # 用作回归和反向微调参数
            torch.nn.Linear(32, 16),
            torch.nn.ReLU(),
            # torch.nn.Dropout(0.5),
            torch.nn.Linear(16, output_size),
        ).to(self.device)

    def forward(self, x):
        x = x.to(self.device)
        n = len(self.rbm_layers)
        hid_out = x.clone
        for i in range(n):
            hidout, _ = self.rbm_layers[i].sample_hidden(hid_out)
        output = self.bpnn(hidout)

        return output

    def pretrain(self, x, y, epoch=50, batch_size=10):
        """
        Train the DBN model layer by layer and fine-tuning with regression
        layer.
        Args:
            x: DBN model input data. Shape: [batch_size, input_length]
            epoch: Train epoch for each RBM.
            batch_size: DBN train batch size.
        Returns:
        """
        # hid_output_i = torch.tensor(x, dtype=torch.float, device=self.device)
        hid_output_i = x

        for i in range(len(self.rbm_layers)):
            print("Training rbm layer {}.".format(i + 1))

            dataset_i = MyDataset(hid_output_i, y)
            dataloader_i = DataLoader(dataset_i, batch_size=1, drop_last=False)

            # dataset = indefDataSet(hid_output_i, y)
            # dataloader_i = DataLoader(dataset=dataset, batch_size=1, shuffle=True)

            self.rbm_layers[i].train_rbm(dataloader_i, epoch)

            hid_output_i, _ = self.rbm_layers[i].forward(hid_output_i)

        # Set pretrain finish flag.
        self.is_pretrained = True


        return

    def finetune(self, x, y, epoch, batch_size, loss_function, optimizer, lr_steps, shuffle=False):
        """
        Fine-tune the train dataset.
        Args:
            x: Input data
            y: Target data
            epoch: Fine-tuning epoch
            batch_size: Train batch size
            loss_function: Train loss function
            optimizer: Finetune optimizer
            shuffle: True if shuffle train data
        Returns:
        """

        if not self.is_pretrained:
            warnings.warn("Hasn't pretrained DBN model yet. Recommend "
                          "run self.pretrain() first.", RuntimeWarning)

        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, lr_steps, gamma=0.9)  # 经过 lr_steps 个 step, 学习率变为原先的0.5

        dataset = MyDataset(x, y)
        dataloader = DataLoader(dataset, 1, shuffle=shuffle)

        print('Begin fine-tuning.')
        for epoch_i in range(1, epoch + 1):
            total_loss = 0
            t = 0
            for input_data, ground_truth in dataloader:
                print(input_data.shape)
                input_data = input_data.view(-1, input_data.shape[1])
                print(input_data.shape)
                input_data = input_data.to(self.device)
                ground_truth = ground_truth.to(self.device)
                output = self.forward(input_data)
                ground_truth = ground_truth.reshape(-1, 1)
                loss = loss_function(ground_truth, output)
                # print(list(self.parameters()))
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
                t += 1
            total_loss = total_loss / t

            # Display train information
            if total_loss >= 1e-4:
                disp = '{2:.4f}'
            else:
                disp = '{2:.3e}'

            print(('Epoch:{0}/{1} -rbm_train_loss: ' + disp).format(epoch_i, epoch, total_loss))

            scheduler.step()  # 学习率达到一定的时候，自动更改
        self.is_finetune = True

        return

    def predict(self, x, y, batch_size, shuffle=False):
        """
        Predict
        Args:
            x: DBN input data. Type: ndarray. Shape: (batch_size, visible_units)
            batch_size: Batch size for DBN model.
            shuffle: True if shuffle predict input data.
        Returns: Prediction result. Type: torch.tensor(). Device is 'cpu' so
            it can transferred to ndarray.
            Shape: (batch_size, output_units)
        """
        if not self.is_pretrained:
            warnings.warn("Hasn't pretrained DBN model yet. Recommend "
                          "run self.pretrain() first.", RuntimeWarning)

        if not self.is_pretrained:
            warnings.warn("Hasn't finetuned DBN model yet. Recommend "
                          "run self.finetune() first.", RuntimeWarning)
        y_predict = []

        dataset = FineTuningDataset(x, y)
        dataloader = DataLoader(dataset, 1, shuffle=shuffle)

        with torch.no_grad():
            for batch in dataloader:
                y = self.forward(batch[0].view(batch[0].shape[1], -1))
                y = y.view(-1, 1)
                y_predict.append(y.cpu().detach().numpy())

        return y_predict


class FineTuningDataset(Dataset):
    """
    Dataset class for whole dataset. x: input data. y: output data
    """

    def __init__(self, x, y):
        pass

        self.x = x
        self.y = y

    def __getitem__(self, index):
        return self.x[index], self.y[index]

    def __len__(self):
        return len(self.x)


class MyDataset(Dataset):

    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __len__(self):
        return len(self.x)

    def __getitem__(self, item):
        return self.x[item], self.y[item]