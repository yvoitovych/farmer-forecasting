import yaml
import torch
import numpy as np
import yaml
import datetime
import os
from torch import nn
from dataset import TradeDataset
from models.rnn import Net
# from metrics import get_metrics

def MAPE(output, target):
    return ((target - output).abs() / target.abs()).mean()
def VWAP(output):
    price_indices = list(range(0, 10)) + list(range(20, 30))
    amount_indices = list(range(10, 20)) + list(range(30, 40))
    return (output[:, price_indices] * output[:, amount_indices]).sum(1) / output[:, amount_indices].sum(1)

class Trainer:
    def __init__(self, config, train_dl, val_dl):
        self.model_name = f'{config["project"]}_{config["task"]}_{datetime.datetime.now()}'
        self.trainloader = train_dl
        self.valloader = val_dl
        self.net = Net().cuda()
        self.price_indices = list(range(0, 10)) + list(range(20, 30))
        self.amount_indices = list(range(10, 20)) + list(range(30, 40))
        self.val_logs_mape = []
        self.val_loss = []
        self.train_loss = []
        self.val_preds = []
        self.val_true = []

        # self.train_metrics = get_metrics(config['metrics'])
        # self.val_metrics = get_metrics(config['metrics'])


        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=config['model']['lr'])
        # self.criterion = nn.SmoothL1Loss(beta=5)
        self.criterion = nn.MSELoss()


    def train(self):
        indices = list(range(0, 10)) + list(range(20, 30))


        min_val_mape = np.inf
        for epoch in range(1500):  # loop over the dataset multiple times


            train_metric_vals = []

            # self.train_metrics[0].clear()
            print(epoch)
            running_loss = 0.0
            self.net.train()
            for i, data in enumerate(self.trainloader):



                input, target = data[0].cuda(), data[1].squeeze(1).cuda()


                self.optimizer.zero_grad()

                output = self.net(input.float())
                output = VWAP(output)
                # print(output.shape, target.shape)
                # print(outputs, labels)
                loss = self.criterion(output, target.float())

                # loss = self.criterion(output[:, indices], target[:, indices].float())




                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.net.parameters(), config['model']['clip'])
                self.optimizer.step()

                # output = output.cpu()
                train_mape = MAPE(output, target)
                # print(train_mape)
                train_metric_vals.append(train_mape)

                # map(lambda x: x.add(outputs, labels.float()), self.train_metrics)

                running_loss += loss.item()
                self.train_loss.append(loss.item())
                if i % 10 == 9:  # print every 2000 mini-batches
                    print('[%d, %5d] TRAIN loss: %.3f' %
                          (epoch + 1, i + 1, running_loss / 10))
                    # train_metrics = self.train_metrics[0].return_value()
                    print(torch.mean(torch.Tensor(train_metric_vals)))

                    running_loss = 0.0




            self.net.eval()
            running_loss = 0.0
            val_metric_vals = []

            for i, data in enumerate(self.valloader):
                input, target = data[0].cuda(), data[1].squeeze(1).cuda()



                output = self.net(input.float())
                output = VWAP(output)


                loss = self.criterion(output, target.float())
                # loss = self.criterion(output[:, indices], target[:, indices].float())


                val_metric_vals.append(MAPE(output, target).item())
                # self.val_metric_vals.append(torch.mean(torch.abs((target[:, indices] - output[:, indices]) / target[:, indices])))


                # self.val_metrics[0].add(outputs.cpu().detach(), labels.cpu())


                running_loss += loss.item()
                self.val_loss.append(loss.item())
                self.val_true.append(target.cpu().detach().numpy().flatten())
                self.val_preds.append(output.cpu().detach().numpy().flatten())


            print('[%d] VAL loss: %.3f' %
                          (epoch + 1, running_loss / i))
            val_metrics = torch.mean(torch.Tensor(val_metric_vals))
            print(val_metrics)

            if val_metrics < min_val_mape:
                print(val_metrics)
                min_val_mape = val_metrics
                torch.save(self.net.state_dict(), f'/common/danylokolinko/alcor/models/{self.model_name}.pth')
            running_loss = 0.0
            self.val_logs_mape += val_metric_vals

        # print('Finished Training')

    def dump(self):
        np.save('/common/danylokolinko/alcor/val_logs_mape.npy', np.array(self.val_logs_mape) )
        np.save('/common/danylokolinko/alcor/val_loss.npy', np.array(self.val_loss) )
        np.save('/common/danylokolinko/alcor/train_loss.npy', np.array(self.train_loss) )
        np.save('/common/danylokolinko/alcor/val_preds.npy', np.concatenate(tuple(self.val_preds))  )
        np.save('/common/danylokolinko/alcor/val_true.npy', np.concatenate(tuple(self.val_true) ))
        # print( np.concatenate(tuple(self.val_true) ).shape,  (self.val_true[0]).shape)




if __name__ == '__main__':
    with open('config/config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    train_ld = torch.utils.data.DataLoader(TradeDataset(config['train'], True),
                                        batch_size=config['batch_size'],
                                        shuffle=True, num_workers=20)
    val_ld = torch.utils.data.DataLoader(TradeDataset(config['val'], False),
                                        batch_size=10,
                                        shuffle=False, num_workers=20)


    # test = iter(train_ld).next()
    # print(test[0].shape, test[1].shape)





    trainer = Trainer(config, train_ld, val_ld)
    trainer.train()
    trainer.dump()

