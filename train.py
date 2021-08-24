from torch.utils.data import DataLoader
from data_manager import EtiDataset
from eti_dataset import ETI
from utils import AverageMeter
import time
import torch
from sklearn.metrics import r2_score


class BaseTrainer:
    def train(self, epoch, model, criterion, optimizer, train_loader):
        batch_loss = AverageMeter()
        batch_corrects = AverageMeter()
        batch_time = AverageMeter()
        data_time = AverageMeter()

        model.train()

        end = time.time()
        for batch_idx, (imgs, scores) in enumerate(train_loader):
            imgs, scores = imgs.cuda(), scores.cuda()
            scores = scores.float()
            imgs = imgs.float()
            # measure data loading time
            data_time.update(time.time() - end)
            # zero the parameter gradients
            optimizer.zero_grad()
            # forward
            outputs, features = model(imgs)
            loss = criterion(outputs, scores.view(-1, 1))
            # backward + optimize
            loss.backward()
            optimizer.step()

            # statistics
            #_, preds = torch.max(outputs.data, 1)
            #batch_corrects.update(torch.sum(preds == scores.data).float() / scores.size(0), scores.size(0))
            batch_corrects.update(r2_score(outputs.view(-1).cpu().data,
                                           scores.cpu().data), scores.size(0))
            batch_loss.update(loss.item(), scores.size(0))
            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

        print('Epoch{0} '
              'Time:{batch_time.sum:.1f}s '
              'Data:{data_time.sum:.1f}s '
              'Loss:{loss.avg:.4f} '
              'R2:{acc.avg:.2%} '.format(
            epoch + 1, batch_time=batch_time,
            data_time=data_time, loss=batch_loss,
            acc=batch_corrects))

        return loss
