import torch
import dataset
import yaml
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
from network import Classifier

with open('../conf/config.yml') as fd:
    conf = yaml.load(fd.read(), Loader=yaml.SafeLoader)
    trainconf = conf['train']

dataset = dataset.MyData('../data/Traindata/')
dataloader = DataLoader(dataset,
                        batch_size=20,
                        shuffle=True,
                        num_workers=0,
                        drop_last=True
                        )
network = Classifier()
print('Network built:')
print(network)
lossfunction = nn.CrossEntropyLoss()
optimizer = optim.SGD(network.parameters(),lr=trainconf['learning_rate'])
gpu_ids = [int(i) for i in trainconf['gpu_ids'].split(',')]
network.cuda(device=0)
# network = nn.DataParallel(network,gpu_ids)

for epoch in range(trainconf['epoch']):
    for index,(img,label) in enumerate(dataloader):
        img = img.cuda()
        # 梯度归零
        optimizer.zero_grad()
        # forward一次
        result = network(img)
        # 计算损失函数
        loss = lossfunction(result,torch.Tensor(label[0]))
        # 反向传播
        loss.backward()
        optimizer.step()
        print('Epoch {} finished!'.format(epoch))

torch.save(network.cpu().state_dict(),'../Data/Model/model.pth')
