# --coding:utf-8--
# RBM.py
import torch
import torch.nn as nn
import math
from torch import sigmoid, bernoulli, matmul
import numpy as np
from torch.utils.data import DataLoader
from GetDataLoader import MyDataset
from torch import optim


class RBM(nn.Module):

    def __init__(self, n_visible, n_hidden, learning_rate=1e-5,
                 learning_rate_decay=False, k=1, batch_size=16,
                 increase_to_cd_k=False, device='cpu'):
        super(RBM, self).__init__()

        self.batch_size = batch_size

        self.k = k
        self.learning_rate = learning_rate
        self.learning_rate_decay = learning_rate_decay
        self.increase_to_cd_k = increase_to_cd_k
        self.device = device

        self.input_size = n_visible
        self.output_size = n_hidden

        W = nn.Parameter(torch.rand(self.input_size, self.output_size, dtype=torch.float32, device=self.device))
        hidden_bias = nn.Parameter(torch.rand(self.output_size, dtype=torch.float32, device=self.device))
        visible_bias = nn.Parameter(torch.rand(self.input_size, dtype=torch.float32, device=self.device))

        self.W = W
        self.hidden_bias = hidden_bias
        self.visible_bias = visible_bias

        self.reset_parameters()


    def sample_hidden(self, visible_probs):

        """
        bernoulli()为概率分布函数，研究表明任意概率分布函数都可以转变为基于能量的模型
        :param visible_probs:可见层数据
        :return: 隐藏层数据
        pj = sigmod(bj + Σ(Wijvi), i=1->n)
        """
        # print(type(visible_probs), print(type(self.W)))
        hidden_probs = torch.matmul(visible_probs, self.W)
        hidden_probs = torch.add(hidden_probs, self.hidden_bias)
        hidden_probs = torch.sigmoid(hidden_probs)
        hidden_sample = torch.bernoulli(hidden_probs)

        return hidden_probs, hidden_sample

    def sample_hidden_2(self, vis_prob):
        """
        Converts the data in visible layer to hidden layer and do sampling.
        Args:
            vis_prob: Visible layer probability. It's also RBM input. Size is
                (n_samples , n_features).
        Returns:
            hid_prob: Hidden layer probability. Size is (n_samples,
                hidden_units)
            hid_sample: None
        """

        # Calculate hid_prob
        hid_prob_res = []
        for i in range(vis_prob.shape[0]):
            # d = torch.from_numpy(vis_prob[i])
            d = vis_prob[i]
            d = torch.tensor(d, dtype=torch.float32, device=self.device)
            hid_prob = torch.matmul(d, self.W)
            hid_prob = torch.add(hid_prob, self.hidden_bias)
            hid_prob = torch.sigmoid(hid_prob)  # 计算sigmod值
            hid_prob_res.append(hid_prob.cpu().detach().numpy())
        print(np.array(hid_prob_res))
        return np.array(hid_prob_res), None

    def sample_visible(self, hidden_probs):

        visible_probs = torch.sigmoid(torch.matmul(hidden_probs, self.W.t()) + self.visible_bias)
        visible_sample = torch.bernoulli(visible_probs)
        return visible_probs, visible_sample

    # def contrastive_divergence(self, visible):
    #
    #     v0 = visible
    #     h0 = self.sample_hidden(v0)
    #     v_k = v0.clone()
    #     for _ in range(self.k):
    #
    #         h_k = self.sample_hidden(v_k)
    #         v_k = self.sample_hidden(h_k)
    #     return v0, h0, v_k

    def reconstruct(self, vis_prob, n_gibbs):
        """
        Reconstruct the sample with k steps of gibbs sampling.
        Args:
            vis_prob: Visible layer probability.
            n_gibbs: Gibbs sampling time, also k.
        Returns:
            Visible probability and sampling after sampling.
        """

        vis_sample = torch.rand(vis_prob.size(), device=self.device)
        for i in range(n_gibbs):
            hid_prob, hid_sample = self.sample_hidden(vis_prob)
            vis_prob, vis_sample = self.sample_visible(hid_prob)
        return vis_prob, vis_sample

    def contrastive_divergence(self, input_data, training=True,
                               n_gibbs_sampling_steps=1, lr=0.001):

        # Positive phase
        positive_hid_prob, positive_hid_dis = self.sample_hidden(input_data)

        positive_associations = torch.matmul(input_data.t(), positive_hid_dis)

        hidden_activations = positive_hid_dis
        vis_prob = torch.rand(input_data.size(), device=self.device)
        hid_prob = torch.rand(positive_hid_prob.size(), device=self.device)
        for i in range(n_gibbs_sampling_steps):
            vis_prob, _ = self.sample_visible(hidden_activations)
            hid_prob, hidden_activations = self.sample_hidden(vis_prob)

        negative_vis_prob = vis_prob
        negative_hid_prob = hid_prob

        negative_associations = torch.matmul(
            negative_vis_prob.t(), negative_hid_prob)

        grad_update = 0
        if training:
            batch_size = self.batch_size
            g = positive_associations - negative_associations
            grad_update = g / batch_size
            v_bias_update = (torch.sum(input_data - negative_vis_prob, dim=0) /
                             batch_size)
            h_bias_update = torch.sum(positive_hid_prob - negative_hid_prob,
                                      dim=0) / batch_size


            self.W.data += lr * grad_update
            self.visible_bias.data += lr * v_bias_update
            self.hidden_bias.data += lr * h_bias_update


        error = torch.mean(torch.sum(
            (input_data - negative_vis_prob) ** 2, dim=0)).item()
        # print('this is error', error)
        # print('this is grade', torch.sum(torch.abs(grad_update)).item())
        return error, torch.sum(torch.abs(grad_update)).item()

    def forward(self, input_data):

        return self.sample_hidden_2(input_data)

    def step(self, input_data, epoch_i, epoch):
        """
        Includes the forward prop and the gradient descent. Used for training.
        Args:
            input_data: RBM visible layer input data.
            epoch_i: Current training epoch.
            epoch: Total training epoch.
        Returns:
        """
        # Gibbs_sampling step gradually increases to k as the train processes.
        if self.increase_to_cd_k:
            n_gibbs_sampling_steps = int(math.ceil((epoch_i / epoch) * self.k))
        else:
            n_gibbs_sampling_steps = self.k

        if self.learning_rate_decay:
            lr = self.learning_rate / epoch_i
        else:
            lr = self.learning_rate
        return self.contrastive_divergence(input_data, True, n_gibbs_sampling_steps, lr)

    def train_rbm(self, train_dataloader, epoch=50):
        """
        Training epoch for a RBM.
        Args:
            train_dataloader: Train dataloader.
            epoch: Train process epoch.
        Returns:
        """

        if isinstance(train_dataloader, DataLoader):
            train_loader = train_dataloader
        else:
            raise TypeError('train_dataloader is not a dataloader instance.')

        for epoch_i in range(1, epoch + 1):
            n_batches = int(len(train_loader))

            cost_ = torch.FloatTensor(n_batches, 1)
            grad_ = torch.FloatTensor(n_batches, 1)

            # Train_loader contains input and output data. However, training
            # of RBM doesn't require output data.
            for i, batch in enumerate(train_loader):
                # print(len(batch))
                batch = batch[0]  # 不取标签
                # print(batch.shape)
                batch = torch.tensor(batch, dtype=torch.float32, device=self.device)
                d = batch[0].view(-1, self.input_size)

                cost_[i - 1], grad_[i - 1] = self.step(d, epoch_i, epoch)
            print('Epoch:{0}/{1} -rbm_train_loss: {2:.3f}'.format(
                epoch_i, epoch, torch.mean(cost_)))

        return

    def reset_parameters(self):
        nn.init.xavier_uniform_(
            self.W, nn.init.calculate_gain(torch.nn.Sigmoid().__class__.__name__.lower()))
        nn.init.zeros_(self.visible_bias)
        nn.init.zeros_(self.hidden_bias)