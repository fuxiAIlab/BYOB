r"""
Pytorch Pipeline.
https://keras.io/api/models/
https://github.com/PyTorchLightning/pytorch-lightning
https://github.com/skorch-dev/skorch
https://pytorch.org/tutorials/beginner/saving_loading_models.html
https://discuss.pytorch.org/t/when-does-pytorch-dataset-or-dataloader-class-convert-numpy-array-to-tensor/63980
"""

import os
import time
from copy import deepcopy

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, Subset
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from byob.utils import to_array, to_tensor, as_tensor


class Pipeline(object):

    def __init__(self, conf, model):
        super(Pipeline, self).__init__()

        self.ml_task = conf['ml_task']
        assert self.ml_task in ('BIN', 'MUL', 'CLS', 'REG', )
        self.dataset = conf['dataset']
        self.model = model
        self.model_name = type(model).__name__
        # self.model_name = model.__class__.__name__.lower()

        self.verbose = conf.get('verbose', False)
        self.log_steps = conf.get('log_steps', 100)
        self.ckpt_freq = conf.get('ckpt_freq', 1)
        self.output_dir = conf.get('output_dir', './output')

        self.model_dir = os.path.join(self.output_dir, 'models')
        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir, exist_ok=True)
        self.ckpt_dir = os.path.join(self.output_dir, 'checkpoints')
        if not os.path.exists(self.ckpt_dir):
            os.makedirs(self.ckpt_dir, exist_ok=True)
        self.hist_dir = os.path.join(self.output_dir, 'historys')
        if not os.path.exists(self.hist_dir):
            os.makedirs(self.hist_dir, exist_ok=True)

        self.optimizer = torch.optim.Adam(model.parameters(), lr=conf['lr'], weight_decay=conf['weight_decay'])
        self.loss_fn = None
        self.metrics = {}

        self.history = {}
        self.best_epoch = None
        self.total_step = 0
        self.cur_epoch = 0  # current epoch
        self.cur_step = 0  # current step within the epoch
        self.loss = 0.0  # current epoch

        self.device = conf['device']
        # device = torch.device("cpu")
        # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # self.device = torch.device(conf['device'])
        # if conf['device'] == 'cpu':
        #     device = torch.device('cpu')
        # else:
        #     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # self.device = device

        if self.verbose:
            print('-' * 80)
            for idx, m in enumerate(model.modules()):
                print(idx, '->', m)
            print('-' * 80)
            for param in model.parameters():
                print(type(param.data), param.size())
            print('-' * 80)
            print(model.state_dict().keys())
            print('-' * 80)

    def compile(self, optimizer=None, loss_fn=None, metrics=None):
        r"""
        https://keras.io/api/models/model_training_apis/#compile-method
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        """
        if optimizer is not None:
            self.optimizer = optimizer
        if loss_fn is not None:
            self.loss_fn = loss_fn
        self.metrics = {} if metrics is None else metrics

    def fit(self, train_ds, valid_ds, batch_size=128, num_epochs=1):
        r"""
        https://keras.io/api/models/model_training_apis/#fit-method
        """
        self.history.clear()
        best_valid_loss = float("inf")
        for epoch in range(1, num_epochs + 1):
            self.cur_epoch += 1
            self.history.setdefault('epoch', []).append(self.cur_epoch)
            print('-' * 80)
            start_time = time.time()
            hist = self.train(train_ds, batch_size)
            elapsed = time.time() - start_time
            # mins, secs = elapsed // 60, elapsed % 60
            loss, acc = hist['loss'], hist['accuracy']
            # print(type(loss), type(acc), loss.shape, acc.shape, loss, acc)
            self.history.setdefault('train loss', []).append(float(loss))
            self.history.setdefault('train acc', []).append(float(acc))
            self.loss = loss
            print('-' * 80)
            if valid_ds is None:
                # print(f'\ttrain loss: {train_loss:.4f}\t|\ttrain acc: {train_acc * 100:.2f}%')
                print('| epoch {:3d}/{:3d} | time: {:5.2f}s | loss {:5.4f} | acc {:5.4f}'.format(
                    epoch, num_epochs, elapsed, hist['loss'], hist['accuracy'] * 100))
            else:
                start_time = time.time()
                valid_hist = self.evaluate(valid_ds, 2 * batch_size)
                valid_elapsed = int(time.time() - start_time)
                valid_loss, valid_acc = valid_hist['loss'], valid_hist['accuracy']
                self.history.setdefault('valid loss', []).append(float(valid_loss))
                self.history.setdefault('valid acc', []).append(float(valid_acc))
                if valid_hist['loss'] < best_valid_loss:
                    best_valid_loss = valid_hist['loss']
                    self.best_epoch = epoch
                # print(f'\tvalid loss: {valid_loss:.4f}\t|\tvalid acc: {valid_acc * 100:.2f}%')
                print('| epoch {:3d}/{:3d} | train time {:5.2f}s | train loss {:5.4f} | train acc {:5.4f} | '
                      'valid time {:5.2f}s | valid loss {:5.4f} | valid acc {:5.4f}'.format(
                       epoch, num_epochs, elapsed, hist['loss'], hist['accuracy'] * 100,
                       valid_elapsed, valid_hist['loss'], valid_hist['accuracy'] * 100))
            # print('-' * 80)
            # self.optimizer.scheduler.step()
            if epoch % self.ckpt_freq == 0:
                self.save_checkpoint()
        self.save_model()
        return deepcopy(self.history)  # NOT use self.history.copy()

    def predict(self, test_ds, batch_size=128):
        r"""Use model predict method for inference (predict).
        https://keras.io/api/models/model_training_apis/#predict-method
        """
        self.model.eval()  # Turn on the evaluation mode
        pred = []
        data_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=0)
        for step, batch in enumerate(data_loader):
            # a = batch; print(step, type(a), a.shape, a.ndim, a.dtype, a.size())
            y_hat = self.predict_batch(batch)
            pred.append(y_hat)
        pred = np.concatenate(pred, axis=0)
        # a = pred; print(type(a), a.shape, a.ndim, a.dtype, a.size, a.itemsize)
        return pred

    def predict_batch(self, batch):
        r"""Use model forward method for inference (predict).
        https://keras.io/api/models/model_training_apis/#predict_on_batch-method
        """
        self.model.eval()  # Turn on the evaluation mode
        x = batch
        x = x.to(self.device)
        with torch.no_grad():
            y_hat = self.model.predict(x)
        # y_hat.cpu() / y_hat.detach() / y_hat.cpu().detach()
        return y_hat

    def forward(self, test_ds, batch_size=128):
        r"""Use model forward method for inference (predict).
        https://keras.io/api/models/model_training_apis/#predict-method
        """
        self.model.eval()  # Turn on the evaluation mode
        preds = []
        data_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=0)
        for step, batch in enumerate(data_loader):
            y_hat = self.forward_batch(batch)
            preds.append(y_hat)
        preds = torch.cat(preds, dim=0)
        # preds.cpu().numpy() / preds.detach().numpy() / preds.cpu().detach().numpy()
        return preds.cpu().numpy()

    def forward_batch(self, batch):
        r"""Use model forward method for inference (predict).
        https://keras.io/api/models/model_training_apis/#predict_on_batch-method
        """
        self.model.eval()  # Turn on the evaluation mode
        x = batch
        x = x.to(self.device)
        with torch.no_grad():
            logits = self.model(x)
            y_hat = logits.argmax(1) + 1
            # proba = torch.sigmoid(logits)  # binary-class classification
            # proba = F.softmax(logits, dim=-1)  # multi-class classification
        # y_hat.cpu() / y_hat.detach() / y_hat.cpu().detach()
        return y_hat.detach()

    def train(self, train_ds, batch_size=128):
        r"""
        https://keras.io/api/models/model_training_apis/#fit-method
        """
        self.model.train()  # Turn on the train mode
        history = {}
        total_loss = 0.
        start_time = time.time()
        data_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
        # steps_per_epoch = len(data_loader)
        steps_per_epoch = len(data_loader.dataset) // batch_size
        for step, batch in enumerate(data_loader):
            self.cur_step = step
            self.total_step += 1
            hist = self.train_batch(batch)
            total_loss += hist['loss']
            history.setdefault('loss', []).append(hist['loss'])
            for name, metric_fn in self.metrics.items():
                history.setdefault(name, []).append(hist[name])
            if step > 0 and step % self.log_steps == 0:
                train_loss = total_loss / self.log_steps
                elapsed = time.time() - start_time
                print('| {:5d}/{:d} steps | {:5.2f} ms/step | loss {:5.4f}'.format(
                    step, steps_per_epoch, elapsed * 1000 / self.log_steps, train_loss))
                total_loss = 0.
                start_time = time.time()
        # loss = train_loss / len(data_loader.dataset)
        # history['loss'] = np.mean(np.array(history['loss']))
        for name in history:
            history[name] = np.mean(np.array(history[name]))
        return history

    def train_batch(self, batch):
        r"""
        https://keras.io/api/models/model_training_apis/#train_on_batch-method
        """
        self.model.train()  # Turn on the train mode
        x, y = batch
        x, y = x.to(self.device), y.to(self.device)
        logits = self.model(x)
        loss = self.loss_fn(logits, y)
        self.optimizer.zero_grad()
        loss.backward()
        # torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.5)
        self.optimizer.step()
        # batch_loss = len(x) * loss.item()
        hist = {'loss': loss.item()}
        for name, metric_fn in self.metrics.items():
            hist[name] = metric_fn(y.cpu(), logits.cpu().detach())
            # hist[name] = metric_fn(y.cpu().numpy(), logits.cpu().detach().numpy())
        return hist

    def evaluate(self, valid_ds, batch_size=128):
        r"""
        https://keras.io/api/models/model_training_apis/#evaluate-method
        """
        self.model.eval()  # Turn on the evaluation mode
        history = {}
        valid_loss = 0.
        data_loader = DataLoader(valid_ds, batch_size=batch_size, shuffle=False, num_workers=0)
        for batch in data_loader:
            hist = self.evaluate_batch(batch)
            valid_loss += hist['loss']
            history.setdefault('loss', []).append(hist['loss'])
            for name, metric_fn in self.metrics.items():
                history.setdefault(name, []).append(hist[name])
        # loss = valid_loss / len(data_loader.dataset)
        # history['loss'] = np.mean(np.array(history['loss']))
        for name in history:
            history[name] = np.mean(np.array(history[name]))
        return history

    def evaluate_batch(self, batch):
        r"""
        https://keras.io/api/models/model_training_apis/#test_on_batch-method
        """
        self.model.eval()  # Turn on the evaluation mode
        x, y = batch
        x, y = x.to(self.device), y.to(self.device)
        with torch.no_grad():
            logits = self.model(x)
            loss = self.loss_fn(logits, y)
        hist = {'loss': loss.item()}
        for name, metric_fn in self.metrics.items():
            hist[name] = metric_fn(y.cpu(), logits.cpu().detach())
            # hist[name] = metric_fn(y.cpu().numpy(), logits.cpu().detach().numpy())
        return hist

    def validate(self, valid_ds, batch_size):
        return self.evaluate(valid_ds, batch_size)

    def test(self, test_ds, batch_size):
        return self.evaluate(test_ds, batch_size)

    def loss_collate(self, output, target):
        if self.ml_task == 'BIN':
            assert output.size(-1) == 1
            output = output.view(-1)
            target = target.view(-1).float()
        elif self.ml_task == 'MUL':
            assert output.size(-1) > 1
            target = target.view(-1).float()
        elif self.ml_task == 'REG':
            assert output.size(-1) == 1
            output = output.view(-1)
            target = target.view(-1).float()
        else:
            output = output.view(-1, output.size(-1))
            target = target.view(-1).float()
            pass
        return output, target

    def _initializer(self):
        """
        https://pytorch.org/docs/stable/nn.init
        https://keras.io/api/layers/initializers/
        """
        pass

    def _optimizer(self, optim='adam', lr=1e-3):
        """
        https://pytorch.org/docs/stable/optim.html
        """
        params = self.model.parameters()
        if optim == 'sgd':
            optimizer = torch.optim.SGD(params, lr=lr, momentum=0, weight_decay=0)
        else:
            optimizer = torch.optim.Adam(params, lr=lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=0)
        return optimizer

    def save_model(self, path=None):
        if path is not None:
            if path[-3:] != '.pt' and path[-4:] != '.pth':
                path = path + '.pt'
        else:
            file_name = '%s-%s-%s.pt' % (self.dataset, self.model_name, str(self.cur_epoch))
            path = os.path.join(self.model_dir, file_name)
        torch.save(self.model.state_dict(), path)

    def load_model(self, path=None):
        if path is not None:
            if path[-3:] != '.pt' and path[-4:] != '.pth':
                path = path + '.pt'
        else:
            file_name = '%s-%s-%s.pt' % (self.dataset, self.model_name, str(self.cur_epoch))
            path = os.path.join(self.model_dir, file_name)
        self.model.load_state_dict(torch.load(path))
        self.model.eval()

    def save_checkpoint(self, path=None):
        if path is not None:
            path = path + '.tar' if path[-4:] != '.tar' else path
        else:
            file_name = '%s-%s-%s.tar' % (self.dataset, self.model_name, str(self.cur_epoch))
            path = os.path.join(self.ckpt_dir, file_name)
        state = {
            'epoch': self.cur_epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'loss': self.loss
        }
        torch.save(state, path)

    def load_checkpoint(self, path=None):
        if path is not None:
            path = path + '.tar' if path[-4:] != '.tar' else path
        else:
            file_name = '%s-%s-%s.tar' % (self.dataset, self.model_name, str(self.cur_epoch))
            path = os.path.join(self.ckpt_dir, file_name)
        checkpoint = torch.load(path)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.cur_epoch = checkpoint['epoch']
        self.loss = checkpoint['loss']
        self.model.eval()  # or self.model.train()


class PipelineBPRSeq(Pipeline):

    def __init__(self, conf, model):
        super(PipelineBPRSeq, self).__init__(conf, model)

    def predict_batch(self, batch):
        self.model.eval()  # Turn on the evaluation mode
        u, i, seq = as_tensor(*batch, dtype=torch.int64, device=self.device)
        with torch.no_grad():
            y_hat = self.model.predict((u, i, seq))
        return y_hat

    def train_batch(self, batch):
        self.model.train()  # Turn on the train mode
        u, i, j, seq = as_tensor(*batch, dtype=torch.int64, device=self.device)
        p_ui, p_uj = self.model((u, i, j, seq))
        loss = self.loss_fn(p_ui, p_uj)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        hist = {'loss': loss.item()}
        for name, metric_fn in self.metrics.items():
            hist[name] = metric_fn(p_ui.cpu(), p_uj.cpu().detach())
            # hist[name] = metric_fn(p_ui.cpu().numpy(), p_uj.cpu().detach().numpy())
        return hist

    def evaluate_batch(self, batch):
        self.model.eval()  # Turn on the evaluation mode
        u, i, j, seq = as_tensor(*batch, dtype=torch.int64, device=self.device)
        with torch.no_grad():
            p_ui, p_uj = self.model((u, i, j, seq))
            loss = self.loss_fn(p_ui, p_uj)
        hist = {'loss': loss.item()}
        for name, metric_fn in self.metrics.items():
            hist[name] = metric_fn(p_ui.cpu(), p_uj.cpu().detach())
            # hist[name] = metric_fn(p_ui.cpu().numpy(), p_uj.cpu().detach().numpy())
        return hist


class PipelineUIYSeq(Pipeline):

    def __init__(self, conf, model):
        super(PipelineUIYSeq, self).__init__(conf, model)

    def predict_batch(self, batch):
        self.model.eval()  # Turn on the evaluation mode
        u, i, seq = as_tensor(*batch, dtype=torch.int64, device=self.device)
        with torch.no_grad():
            y_hat = self.model.predict((u, i, seq))
        return y_hat

    def train_batch(self, batch):
        self.model.train()  # Turn on the train mode
        u, i, y, seq = as_tensor(*batch, dtype=torch.int64, device=self.device)
        logits = self.model((u, i, seq))
        logits = logits.reshape(-1) if logits.size(-1) == 1 else logits.reshape(-1, logits.size(-1))
        y = y.view(-1).float()
        loss = self.loss_fn(logits, y)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        hist = {'loss': loss.item()}
        for name, metric_fn in self.metrics.items():
            hist[name] = metric_fn(y.cpu(), logits.cpu().detach())
            # hist[name] = metric_fn(y.cpu().numpy(), logits.cpu().detach().numpy())
        return hist

    def evaluate_batch(self, batch):
        self.model.eval()  # Turn on the evaluation mode
        u, i, y, seq = as_tensor(*batch, dtype=torch.int64, device=self.device)
        with torch.no_grad():
            logits = self.model((u, i, seq))
            logits = logits.reshape(-1) if logits.size(-1) == 1 else logits.reshape(-1, logits.size(-1))
            y = y.view(-1).float()
            loss = self.loss_fn(logits, y)
        hist = {'loss': loss.item()}
        for name, metric_fn in self.metrics.items():
            hist[name] = metric_fn(y.cpu(), logits.cpu().detach())
            # hist[name] = metric_fn(y.cpu().numpy(), logits.cpu().detach().numpy())
        return hist
