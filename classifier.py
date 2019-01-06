# -*- coding: utf-8 -*-
# Copyright 2017 Kakao, Recommendation Team
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import json
import threading

import fire
import h5py
import pandas as pd
import tqdm
import numpy as np
import six
import seaborn as sn
import keras
import keras.backend as K

import network
import metric
from collections import defaultdict
from datetime import datetime
from keras.models import load_model
from keras.callbacks import ModelCheckpoint
from six.moves import zip, cPickle
from sklearn.utils import class_weight
from sklearn.metrics import accuracy_score, confusion_matrix

from misc import get_logger, Option

opt = Option('./config.json')
os.environ["CUDA_VISIBLE_DEVICES"] = opt.visible_gpu
if six.PY2:
    cate1 = json.loads(open('../cate1.json').read())
else:
    cate1 = json.loads(open('../cate1.json', 'rb').read().decode('utf-8'))
DEV_DATA_LIST = ['../data/dev.chunk.01']


class Classifier():
    def __init__(self):
        self.logger = get_logger('Classifier')
        self.cate_size = {'b': 57, 'm': 552, 's': 3190, 'd': 404, 'bmsd': 4215}

    def get_sample_generator(self, ds, batch_size, target=None, raise_stop_event=False, cate_only=False, pred_val=None):
        left, limit = 0, ds['uni'].shape[0]
        while True:
            right = min(left + batch_size, limit)

            if opt.multi_label is True and target is not None:
                if cate_only is False:
                    X = [ds[t][left:right, :] for t in ['uni', 'w_uni', 'char', 'img']]
                    Y = ds['%s_cate' % target][left:right]
                else:
                    X = [ds[t][left:right, :] for t in ['uni', 'w_uni', 'img']]
                    if pred_val is None:
                        _t = ds
                        b = _t['b'][left:right, :]
                        m = _t['m'][left:right, :] + self.cate_size['b']
                        s = _t['s'][left:right, :] + self.cate_size['b'] + self.cate_size['m']
                        d = _t['d'][left:right, :] + self.cate_size['b'] + self.cate_size['m'] + self.cate_size['d']
                    else:
                        _t = pred_val
                        b = _t['b'][left:right].values.reshape(-1, 1)
                        m = _t['m'][left:right].values.reshape(-1, 1) + self.cate_size['b']
                        s = _t['s'][left:right].values.reshape(-1, 1) + self.cate_size['b'] + self.cate_size['m']
                        d = _t['d'][left:right].values.reshape(-1, 1) + self.cate_size['b'] + self.cate_size['m'] + self.cate_size['s']
                    bmsd = np.concatenate([b, m, s, d], axis=1)
                    X += [bmsd]
                    Y = ds['cate'][left:right]
            else:
                X = [ds[t][left:right, :] for t in ['uni', 'w_uni', 'img']]
                Y = ds['cate'][left:right]
            yield X, Y
            left = right
            if right == limit:
                left = 0
                if raise_stop_event:
                    raise StopIteration

    def get_inverted_cate1(self, cate1):
        inv_cate1 = {}
        for d in ['b', 'm', 's', 'd']:
            inv_cate1[d] = {v: k for k, v in six.iteritems(cate1[d])}
        return inv_cate1

    def write_prediction_result(self, data, pred_y, meta, out_path, readable, is_train=False):
        pid_order = []
        if is_train is False:
            for data_path in DEV_DATA_LIST:
                h = h5py.File(data_path, 'r')['dev']
                pid_order.extend(h['pid'][::])
        else:
            pid_order.extend(data['pid'][::])

        inv_cate1 = self.get_inverted_cate1(cate1)
        rets = {}
        for pid, b, m, s, d in zip(data['pid'], pred_y[0], pred_y[1], pred_y[2], pred_y[3]):
            if six.PY3:
                pid = pid.decode('utf-8')
            if b not in inv_cate1['b']:
                self.logger.info('b: {}'.format(b))
            if m not in inv_cate1['m']:
                self.logger.info('m: {}'.format(m))
            if s not in inv_cate1['s']:
                self.logger.info('s: {}'.format(s))
            if d not in inv_cate1['d']:
                self.logger.info('d: {}'.format(d))
            assert b in inv_cate1['b']
            assert m in inv_cate1['m']
            assert s in inv_cate1['s']
            assert d in inv_cate1['d']

            tpl = '{pid}\t{b}\t{m}\t{s}\t{d}'
            if readable:
                b = inv_cate1['b'][b]
                m = inv_cate1['m'][m]
                s = inv_cate1['s'][s]
                d = inv_cate1['d'][d]
            rets[pid] = tpl.format(pid=pid, b=b, m=m, s=s, d=d)
        no_answer = '{pid}\t-1\t-1\t-1\t-1'
        with open(out_path, 'w') as fout:
            for pid in pid_order:
                if six.PY3:
                    pid = pid.decode('utf-8')
                ans = rets.get(pid, no_answer.format(pid=pid))
                fout.write(ans)
                fout.write('\n')

    def write_infer_result(self, data, pred_y, meta, out_path, readable, is_train=False):
        pid_order = []
        if is_train is False:
            for data_path in DEV_DATA_LIST:
                h = h5py.File(data_path, 'r')['dev']
                pid_order.extend(h['pid'][::])
        else:
            pid_order.extend(data['pid'][::])

        y2l = {i: s for s, i in six.iteritems(meta['y_vocab'])}
        y2l = list(map(lambda x: x[1], sorted(y2l.items(), key=lambda x: x[0])))
        inv_cate1 = self.get_inverted_cate1(cate1)
        rets = {}
        for pid, y in zip(data['pid'], pred_y):
            if six.PY3:
                pid = pid.decode('utf-8')
            label = y2l[y]
            tkns = list(map(int, label.split('>')))
            b, m, s, d = tkns
            assert b in inv_cate1['b']
            assert m in inv_cate1['m']
            assert s in inv_cate1['s']
            assert d in inv_cate1['d']
            tpl = '{pid}\t{b}\t{m}\t{s}\t{d}'
            if readable:
                b = inv_cate1['b'][b]
                m = inv_cate1['m'][m]
                s = inv_cate1['s'][s]
                d = inv_cate1['d'][d]
            rets[pid] = tpl.format(pid=pid, b=b, m=m, s=s, d=d)
        no_answer = '{pid}\t-1\t-1\t-1\t-1'
        with open(out_path, 'w') as fout:
            for pid in pid_order:
                if six.PY3:
                    pid = pid.decode('utf-8')
                ans = rets.get(pid, no_answer.format(pid=pid))
                fout.write(ans)
                fout.write('\n')

    def infer(self, data_root, model_root, test_root, test_div, out_path, csv_path,
              readable=False, target='bmsd'):
        meta_path = os.path.join(data_root, 'meta')
        meta = cPickle.loads(open(meta_path, 'rb').read())

        K.set_learning_phase(0)
        model = network.get_model('bmsd', self.cate_size)
        self.logger.info('# of classes(train): %s' % len(meta['y_vocab']))
        model.load_weights(model_root)
        K.set_learning_phase(0)

        test_path = os.path.join(test_root, 'data.h5py')
        test_data = h5py.File(test_path, 'r')

        test = test_data[test_div]
        batch_size = opt.batch_size

        # load pred val
        df = pd.read_csv(csv_path, names=['pid', 'b', 'm', 's', 'd'],
                         delimiter='\t', index_col=False, encoding='utf-8')
        df['b'] = df['b'] - 1
        df['m'] = df['m'] - 1
        df['s'] = df['s'] - 1
        df['d'] = df['d'] - 1
        df['s'][df['s'] < 0] = 0
        df['d'][df['d'] < 0] = 0
        pred_y = []
        test_gen = ThreadsafeIter(self.get_sample_generator(test,
                                                            batch_size=batch_size,
                                                            target='bmsd',
                                                            raise_stop_event=True,
                                                            cate_only=True,
                                                            pred_val=df))
        total_test_samples = test['uni'].shape[0]
        with tqdm.tqdm(total=total_test_samples) as pbar:
            for chunk in test_gen:
                total_test_samples = test['uni'].shape[0]
                X, _ = chunk
                _pred_y = model.predict(X)
                pred_y.extend([np.argmax(y) for y in _pred_y])
                pbar.update(X[0].shape[0])
        self.write_infer_result(test, pred_y, meta, out_path, readable=readable)

    def predict(self, data_root, test_root, test_div, out_path, readable=False,
                mode='h5', target='b', cf_map=False, is_train=False):
        meta_path = os.path.join(data_root, 'meta')
        meta = cPickle.loads(open(meta_path, 'rb').read())

        self.logger.info('# of classes(train): %s' % len(meta['y_vocab']))
        K.set_learning_phase(0)
        models = {}
        for idx, cate in enumerate(['b', 'm', 's', 'd']):
            if mode == 'h5':
                model_fname = opt.model_path_list[idx]
                model = load_model(model_fname,
                                   custom_objects={'top1_acc': top1_acc})
            elif mode == 'weights':
                if target == 'bmsd':
                    t = cate
                else:
                    t = target
                model = network.get_model(t, self.cate_size)
                model.load_weights(opt.weight_path_list[idx])
            else:
                raise Exception('Unknown mode: {}'.format(mode))
            models[idx] = model

        test_path = os.path.join(test_root, 'data.h5py')
        test_data = h5py.File(test_path, 'r')

        test = test_data[test_div]
        batch_size = opt.batch_size
        total_test_samples = test['uni'].shape[0]
        with tqdm.tqdm(total=total_test_samples) as pbar:
            pred_dic = defaultdict()
            if cf_map is True:
               true_dic = defaultdict()
            for idx, cate in enumerate(['b', 'm', 's', 'd']):
                for chunk in ThreadsafeIter(self.get_sample_generator(test,
                                                            batch_size=batch_size,
                                                            target=cate,
                                                            raise_stop_event=True)):
                    X, Y = chunk
                    _pred_y = models[idx].predict(X)
                    if idx not in pred_dic:
                        _p = np.argmax(_pred_y, axis=1) + 1
                        if cate == 's' or cate == 'd':
                            _p[_p == 1] = -1
                        pred_dic[idx] = _p
                        if cf_map is True:
                            _y = np.argmax(Y, axis=1) + 1
                            if cate == 's' or cate == 'd':
                                _y[_y == 1] = -1
                            true_dic[idx] = _y
                    else:
                        _p = np.argmax(_pred_y, axis=1) + 1
                        if cate == 's' or cate == 'd':
                            _p[_p == 1] = -1
                        pred_dic[idx] = np.concatenate([pred_dic[idx], _p])
                        if cf_map is True:
                            _y = np.argmax(Y, axis=1) + 1
                            if cate == 's' or cate == 'd':
                                _y[_y == 1] = -1
                            true_dic[idx] = np.concatenate([true_dic[idx], _y])
                            if len(pred_dic[idx]) != len(true_dic[idx]):
                                raise Exception('{} != {}'.format(len(pred_dic[idx]), len(true_dic[idx])))
                    pbar.update(X[0].shape[0] // 4)
            if cf_map is True:
                for idx, ((cate, size), font_scale, dpi) in enumerate(zip(self.cate_size.items(),
                                                                          [0.1, 0.01, 0.001, 0.01, 0.001],
                                                                          [800, 800, 4000, 800, 4000])):
                    y_test, y_pred = true_dic[idx], pred_dic[idx]
                    top1_acc = accuracy_score(y_test, y_pred)
                    conf_mat = confusion_matrix(y_test, y_pred)
                    classes = [label for label in range(size)]
                    sn.set(font_scale=font_scale)
                    svm = sn.heatmap(conf_mat, annot=True if idx == 0 else False, cmap='Blues')
                    figure = svm.get_figure()
                    figure.savefig('{}_conf_{:.2f}.png'.format(cate, top1_acc), dpi=dpi)
                    figure.clf()
                    if target == cate:
                        break
        self.write_prediction_result(test, pred_dic, meta, out_path, readable=readable, is_train=False)

    def train(self, data_root, out_dir, target='bmsd', weight_path=None, weight_mode=None, model_name=None, cate_only=False):
        assert target in ['b', 'm', 's', 'd', 'bmsd']

        data_path = os.path.join(data_root, 'data.h5py')
        meta_path = os.path.join(data_root, 'meta')
        data = h5py.File(data_path, 'r')
        meta = cPickle.loads(open(meta_path, 'rb').read())
        self.weight_fname = os.path.join(out_dir, 'weights')
        self.model_fname = os.path.join(out_dir, 'model')
        if not os.path.isdir(out_dir):
            os.makedirs(out_dir)

        self.logger.info('# of classes: %s' % self.cate_size[target])

        train = data['train']
        dev = data['dev']

        c_name = target if target != 'bmsd' else 'b'
        self.logger.info('# of train samples: %s' % train['{}_cate'.format(c_name)].shape[0])
        self.logger.info('# of dev samples: %s' % dev['{}_cate'.format(c_name)].shape[0])

        callbacks = []
        for i, t in enumerate(['b', 'm', 's', 'd', 'bmsd']):
            if t == target:
                monitor_name = 'val_loss'
                callbacks += [ModelCheckpoint('{}_0{}'.format(self.weight_fname, i + 1),
                    monitor=monitor_name,
                    save_best_only=True, mode='min', period=opt.num_checkpoint)]

        # generate vocab matrix TODO
        '''
        vocab_mat = np.zeros((len(meta['y_vocab']), 4), dtype=np.int32)
        inv_y_vocab = {v: k for k, v in six.iteritems(meta['y_vocab'])}
        for k, v in inv_y_vocab.items():
            item = list(map(int, v.split('>')))
            vocab_mat[k] = np.array(item).reshape(1, 4)
        '''
        model = network.get_model(target, self.cate_size)
        if weight_path is not None:
            if os.path.exists(weight_path):
                model.load_weights(weight_path)
            else:
                raise Exception('Not exist path: {}'.format(weight_path))

        total_train_samples = train['uni'].shape[0]
        train_gen = self.get_sample_generator(train,
                                              batch_size=opt.batch_size, target=target, cate_only=cate_only)
        self.steps_per_epoch = int(np.ceil(total_train_samples / float(opt.batch_size)))

        total_dev_samples = dev['uni'].shape[0]
        dev_gen = self.get_sample_generator(dev,
                                            batch_size=opt.batch_size, target=target, cate_only=cate_only)
        self.validation_steps = int(np.ceil(total_dev_samples / float(opt.batch_size)))

        if not model_name:
            now = datetime.now()
            model_name = '{}-{}-{} {}:{}:{}'.format(now.year, now.month, now.day,
                    now.hour, now.minute, now.second)
        callbacks += [keras.callbacks.TensorBoard(log_dir='./graph/{0}'.format(model_name),
                      histogram_freq=0, write_graph=True, write_images=True)]

        if weight_mode == 'class_weight':
            label = list(np.argmax(train['{}_cate'.format(target)], axis=1))\
                    + list(np.argmax(dev['{}_cate'.format(target)], axis=1))
            weights = class_weight.compute_class_weight('balanced',
                    np.unique(label), label)
            self.logger.info('class weight length: {}'.format(len(weights)))
            self.logger.info('class weight mean: {}'.format(np.mean(weights)))
            self.logger.info('class weight std: {}'.format(np.std(weights)))
            self.logger.info('class weight min: {}'.format(np.min(weights)))
            self.logger.info('class weight max: {}'.format(np.max(weights)))
            self.logger.info('class weight median: {}'.format(np.median(weights)))
            class_weights = {k : v for k, v in zip(np.unique(label), weights)}
        elif weight_mode == 'score_weight':
            label, class_weights = [], []
            for k, v in inv_y_vocab.items():
                score = sum([0 if d == -1  else s
                            for d, s  in zip(list(map(int, v.split('>'))),
                                             [1.0, 1.2, 1.3, 1.4])]) / 4.0 + 0.75
                label.append(k)
                class_weights.append(score)
            class_weights = list(np.power(class_weights, opt.score_exp))
            self.logger.info('score exp: {}'.format(opt.score_exp))
            self.logger.info('score weight length: {}'.format(len(class_weights)))
            self.logger.info('score weight mean: {}'.format(np.mean(class_weights)))
            self.logger.info('score weight std: {}'.format(np.std(class_weights)))
            self.logger.info('score weight min: {}'.format(np.min(class_weights)))
            self.logger.info('score weight max: {}'.format(np.max(class_weights)))
            self.logger.info('score weight median: {}'.format(np.median(class_weights)))
            class_weights = {k : v for k, v in zip(label, class_weights)}
        else:
            class_weights = None

        model.fit_generator(generator=train_gen,
                            steps_per_epoch=self.steps_per_epoch,
                            epochs=opt.num_epochs,
                            validation_data=dev_gen,
                            validation_steps=self.validation_steps,
                            shuffle=True,
                            class_weight=class_weights,
                            callbacks=callbacks)

        for i, t in enumerate(['b', 'm', 's', 'd', 'bmsd']):
            if t == target:
                model.load_weights('{}_0{}'.format(self.weight_fname, i+1)) # loads from checkout point if exists
                open('{}_0{}.json'.format(self.model_fname, i+1), 'w').write(model.to_json())
                model.save('{}_0{}.h5'.format(self.model_fname, i+1))
        data.close()


class ThreadsafeIter(object):
    def __init__(self, it):
        self._it = it
        self._lock = threading.Lock()

    def __iter__(self):
        return self

    def __next__(self):
        with self._lock:
            return next(self._it)

    def next(self):
        with self._lock:
            return self._it.next()


if __name__ == '__main__':
    clsf = Classifier()
    fire.Fire({'train': clsf.train,
               'predict': clsf.predict,
               'infer': clsf.infer})
