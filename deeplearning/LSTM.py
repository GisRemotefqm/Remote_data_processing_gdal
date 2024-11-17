# --coding:utf-8--
import torch
class RNN(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.rnn = torch.nn.LSTM(
            input_size=1,
            hidden_size=64,
            num_layers=1,
            batch_first=True
        )
        self.out = torch.nn.Linear(in_features=64, out_features=1)

    def forward(self, x):
        # 一下关于shape的注释只针对单项
        # output: [batch_size, time_step, hidden_size]
        # h_n: [num_layers,batch_size, hidden_size] # 虽然LSTM的batch_first为True,但是h_n/c_n的第一维还是num_layers
        # c_n: 同h_n
        output, (h_n, c_n) = self.rnn(x)
        # output_in_last_timestep=output[:,-1,:] # 也是可以的
        output_in_last_timestep = h_n[-1, :, :]
        # print(output_in_last_timestep.equal(output[:,-1,:])) #ture
        x = self.out(output_in_last_timestep)
        return x


# 我们对x = 2 进行数据填充
x = torch.tensor([[1, 3, 4, 5, 6]]).float().unsqueeze(0)
y = torch.tensor([5, 7, 8, 9, 10]).float()



net = RNN()
# 3. 训练
# 3. 网络的训练（和之前CNN训练的代码基本一样）
optimizer = torch.optim.Adam(net.parameters(), lr=0.001)
loss_F = torch.nn.MSELoss()
for epoch in range(500):  # 数据集只迭代一次
    print(x.view(-1, 1, 1))
    pred = net(x.view(-1, 1, 1))
    print(pred)

    loss = loss_F(pred, y)  # 计算loss
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    print('Epoch: ', epoch, '| train loss: %.4f' % loss.data.numpy())


with torch.no_grad():
        test_pred = net(torch.tensor([[2]]).unsqueeze(0).float().view(-1, 1, 1))
        print(test_pred)
