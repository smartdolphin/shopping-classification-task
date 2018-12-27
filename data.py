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
os.environ['OMP_NUM_THREADS'] = '1'
import re
import sys
import traceback
from collections import Counter
from multiprocessing import Pool

import json
import pandas as pd
import tqdm
import fire
import h5py
import numpy as np
import mmh3
import six
from keras.utils.np_utils import to_categorical
from kor_char_parser import decompose_str_as_one_hot
from six.moves import cPickle

from misc import get_logger, Option
opt = Option('./config.json')

re_sc = re.compile('[\!@#$%\^&\*\(\)-=\[\]\{\}\.,/\?~\+\'"|]')


class Reader(object):
    def __init__(self, data_path_list, div, begin_offset, end_offset):
        self.div = div
        self.data_path_list = data_path_list
        self.begin_offset = begin_offset
        self.end_offset = end_offset

    def is_range(self, i):
        if self.begin_offset is not None and i < self.begin_offset:
            return False
        if self.end_offset is not None and self.end_offset <= i:
            return False
        return True

    def get_size(self):
        offset = 0
        count = 0
        for data_path in self.data_path_list:
            h = h5py.File(data_path, 'r')
            sz = h[self.div]['pid'].shape[0]
            if not self.begin_offset and not self.end_offset:
                offset += sz
                count += sz
                continue
            if self.begin_offset and offset + sz < self.begin_offset:
                offset += sz
                continue
            if self.end_offset and self.end_offset < offset:
                break
            for i in range(sz):
                if not self.is_range(offset + i):
                    continue
                count += 1
            offset += sz
        return count

    def get_class(self, h, i):
        b = h['bcateid'][i]
        m = h['mcateid'][i]
        s = h['scateid'][i]
        d = h['dcateid'][i]
        return '%s>%s>%s>%s' % (b, m, s, d)

    def generate(self):
        offset = 0
        for data_path in self.data_path_list:
            h = h5py.File(data_path, 'r')[self.div]
            sz = h['pid'].shape[0]
            if self.begin_offset and offset + sz < self.begin_offset:
                offset += sz
                continue
            if self.end_offset and self.end_offset < offset:
                break
            for i in range(sz):
                if not self.is_range(offset + i):
                    continue
                class_name = self.get_class(h, i)
                yield h['pid'][i], class_name, h, i
            offset += sz

    def get_y_vocab(self, data_path):
        y_vocab = {}
        h = h5py.File(data_path, 'r')[self.div]
        sz = h['pid'].shape[0]
        for i in tqdm.tqdm(range(sz), mininterval=1):
            class_name = self.get_class(h, i)
            if class_name not in y_vocab:
                y_vocab[class_name] = len(y_vocab)
        return y_vocab


def preprocessing(data):
    try:
        cls, data_path_list, div, out_path, begin_offset, end_offset = data
        data = cls()
        data.load_y_vocab()
        data.preprocessing(data_path_list, div, begin_offset, end_offset, out_path)
    except Exception:
        raise Exception("".join(traceback.format_exception(*sys.exc_info())))


def build_y_vocab(data):
    try:
        data_path, div = data
        reader = Reader([], div, None, None)
        y_vocab = reader.get_y_vocab(data_path)
    except Exception:
        raise Exception("".join(traceback.format_exception(*sys.exc_info())))
    return y_vocab


def make_csv_content(data):
    try:
        cls, data_root, our_dir, data_path, target = data
        data = cls()
        data.generate_csv(data_root, our_dir, data_path, target)
    except Exception:
        raise Exception("".join(traceback.format_exception(*sys.exc_info())))


def csv_worker(data):
    data, meta, inv_cate1, classes = data
    try:
        size = len(data['pid'])
        df = pd.DataFrame(columns=['brand', 'maker', 'model', 'pid', 'price',
            'product', 'updttm', 'cate', '1st', '2nd', '3rd', '4th', 'classes'])

        for i in tqdm.trange(size):
            label = classes[i]
            tkns = list(map(int, label.split('>')))
            b, m, s, d = tkns
            if b != -1:
                assert b in inv_cate1['b']
                b = inv_cate1['b'][b]
            if m != -1:
                assert m in inv_cate1['m']
                m = inv_cate1['m'][m]
            assert s in inv_cate1['s']
            assert d in inv_cate1['d']
            s = inv_cate1['s'][s]
            d = inv_cate1['d'][d]
            pid = data['pid'][i].decode('utf-8')

            n_class = ''
            if b != -1 and m != -1:
                assert label in meta['y_vocab']
                n_class = meta['y_vocab'][label]

            df.loc[i] = [data['brand'][i].decode('utf-8'),
                         data['maker'][i].decode('utf-8'),
                         data['model'][i].decode('utf-8'),
                         pid,
                         data['price'][i],
                         data['product'][i].decode('utf-8'),
                         data['updttm'][i].decode('utf-8'),
                         label,
                         b, m, s, d,
                         n_class]
    except Exception:
        raise Exception(''.join(traceback.format_exception(*sys.exc_info())))
    return df


class Data:
    y_vocab_path = './data/y_vocab.cPickle' if six.PY2 else './data/y_vocab.py3.cPickle'
    tmp_chunk_tpl = 'tmp/base.chunk.%s'

    def __init__(self):
        self.logger = get_logger('data')
        self.cate_size = {'b': 57, 'm': 552, 's': 3190, 'd': 404}
        self.filter_words = set()
        if os.path.exists(opt.filter_path):
            with open(opt.filter_path, 'r', encoding='utf-8') as f:
                for line in f.readlines():
                    self.filter_words.add(line.strip())

    def load_y_vocab(self):
        self.y_vocab = cPickle.loads(open(self.y_vocab_path, 'rb').read())

    def build_y_vocab(self):
        pool = Pool(opt.num_workers)
        try:
            rets = pool.map_async(build_y_vocab,
                                  [(data_path, 'train')
                                   for data_path in opt.train_data_list]).get(99999999)
            pool.close()
            pool.join()
            y_vocab = set()
            for _y_vocab in rets:
                for k in six.iterkeys(_y_vocab):
                    y_vocab.add(k)
            self.y_vocab = {y: idx for idx, y in enumerate(y_vocab)}
            self.inv_y_vocab = {v: k for k, v in six.iteritems(self.y_vocab)}
        except KeyboardInterrupt:
            pool.terminate()
            pool.join()
            raise
        self.logger.info('size of y vocab: %s' % len(self.y_vocab))
        cPickle.dump(self.y_vocab, open(self.y_vocab_path, 'wb'), 2)

    def _split_data(self, data_path_list, div, chunk_size):
        total = 0
        for data_path in data_path_list:
            h = h5py.File(data_path, 'r')
            sz = h[div]['pid'].shape[0]
            total += sz
        chunks = [(i, min(i + chunk_size, total))
                  for i in range(0, total, chunk_size)]
        return chunks

    def preprocessing(self, data_path_list, div, begin_offset, end_offset, out_path):
        self.div = div
        reader = Reader(data_path_list, div, begin_offset, end_offset)
        rets = []
        for pid, label, h, i in reader.generate():
            y, y_cate, x = self.parse_data(label, h, i)
            if y is None:
                continue
            rets.append((pid, y, y_cate, x))
        self.logger.info('sz=%s' % (len(rets)))
        open(out_path, 'wb').write(cPickle.dumps(rets, 2))
        self.logger.info('%s ~ %s done. (size: %s)' % (begin_offset, end_offset, end_offset - begin_offset))

    def _preprocessing(self, cls, data_path_list, div, chunk_size):
        chunk_offsets = self._split_data(data_path_list, div, chunk_size)
        num_chunks = len(chunk_offsets)
        self.logger.info('split data into %d chunks, # of classes=%s' % (num_chunks, len(self.y_vocab)))
        pool = Pool(opt.num_workers)
        try:
            pool.map_async(preprocessing, [(cls,
                                            data_path_list,
                                            div,
                                            self.tmp_chunk_tpl % cidx,
                                            begin,
                                            end)
                                           for cidx, (begin, end) in enumerate(chunk_offsets)]).get(9999999)
            pool.close()
            pool.join()
        except KeyboardInterrupt:
            pool.terminate()
            pool.join()
            raise
        return num_chunks

    def parse_data(self, label, h, i):
        y_val = self.y_vocab.get(label)
        if y_val is None and self.div in ['dev', 'test']:
            y_val = 0
        if y_val is None and self.div != 'test':
            return [None] * 3
        Y = to_categorical(y_val, len(self.y_vocab))

        product = h['product'][i]
        if six.PY3:
            product = product.decode('utf-8')
        product = self.filter_func(product)
        product = re_sc.sub(' ', product).strip().split()
        words = [w.strip() for w in product]
        words = [w for w in words
                 if len(w) >= opt.min_word_length and len(w) < opt.max_word_length]
        if not words:
            return [None] * 3

        if opt.data_mode == 'seq':
            hash_func = hash if six.PY2 else lambda x: mmh3.hash(x, seed=17)
            x = [hash_func(w) % opt.unigram_hash_size + 1 for w in words]
            xv = Counter(x).most_common(opt.max_len)
            ch = decompose_str_as_one_hot(''.join(words))[:opt.max_len]
        else:
            hash_func = hash if six.PY2 else lambda x: mmh3.hash(x, seed=17)
            x = [hash_func(w) % opt.unigram_hash_size + 1 for w in words]
            xv = Counter(x).most_common(opt.max_len)

        x = np.zeros(opt.max_len, dtype=np.float32)
        v = np.zeros(opt.max_len, dtype=np.int32)
        c = np.zeros(opt.max_len, dtype=np.float32)
        for i in range(len(xv)):
            if opt.data_mode == 'seq':
                x[i] = xv[i][0]
                v[i] = xv[i][1]
                c[i] = ch[i]
            else:
                x[i] = xv[i][0]
                v[i] = xv[i][1]

        # image feature
        img = h['img_feat'][i]

        # price feature
        price = -1 if opt.mode == 'seq' else h['price'][i]

        # set multi y for each category
        cate_y = {}
        tkns = list(map(int, label.split('>')))
        for cate, c_label in zip(['b', 'm', 's', 'd'], tkns):
            cate_y[cate] = 0 if c_label == -1 else c_label - 1
            cate_y['{}_cate'.format(cate)] = to_categorical(cate_y[cate], self.cate_size[cate])

        return Y, cate_y, (x, v, c, img, price)

    def filter_func(self, sentence):
        for filter_str in self.filter_words:
            if filter_str in sentence:
                sentence = sentence.replace(filter_str, '').strip()
        return sentence

    def create_dataset(self, g, size, num_classes):
        shape = (size, opt.max_len)
        img_shape = (size, opt.img_size)
        g.create_dataset('uni', shape, chunks=True, dtype=np.int32)
        g.create_dataset('w_uni', shape, chunks=True, dtype=np.float32)
        g.create_dataset('char', shape, chunks=True, dtype=np.float32)
        g.create_dataset('img', img_shape, chunks=True, dtype=np.float32)
        g.create_dataset('price', (size, 1), chunks=True, dtype=np.int32)
        g.create_dataset('cate', (size, num_classes), chunks=True, dtype=np.int32)
        g.create_dataset('b', (size, 1), chunks=True, dtype=np.int32)
        g.create_dataset('m', (size, 1), chunks=True, dtype=np.int32)
        g.create_dataset('s', (size, 1), chunks=True, dtype=np.int32)
        g.create_dataset('d', (size, 1), chunks=True, dtype=np.int32)
        g.create_dataset('b_cate', (size, self.cate_size['b']), chunks=True, dtype=np.int32)
        g.create_dataset('m_cate', (size, self.cate_size['m']), chunks=True, dtype=np.int32)
        g.create_dataset('s_cate', (size, self.cate_size['s']), chunks=True, dtype=np.int32)
        g.create_dataset('d_cate', (size, self.cate_size['d']), chunks=True, dtype=np.int32)
        g.create_dataset('pid', (size,), chunks=True, dtype='S12')

    def init_chunk(self, chunk_size, num_classes):
        chunk_shape = (chunk_size, opt.max_len)
        # image feature size is 2048
        img_shape = (chunk_size, opt.img_size)
        chunk = {}
        chunk['uni'] = np.zeros(shape=chunk_shape, dtype=np.int32)
        chunk['w_uni'] = np.zeros(shape=chunk_shape, dtype=np.float32)
        chunk['char'] = np.zeros(shape=chunk_shape, dtype=np.float32)
        chunk['img'] = np.zeros(shape=img_shape, dtype=np.float32)
        chunk['price'] = np.zeros(shape=(chunk_size, 1), dtype=np.int32)
        chunk['cate'] = np.zeros(shape=(chunk_size, num_classes), dtype=np.int32)
        chunk['b'] = np.zeros(shape=(chunk_size, 1), dtype=np.int32)
        chunk['m'] = np.zeros(shape=(chunk_size, 1), dtype=np.int32)
        chunk['s'] = np.zeros(shape=(chunk_size, 1), dtype=np.int32)
        chunk['d'] = np.zeros(shape=(chunk_size, 1), dtype=np.int32)
        chunk['b_cate'] = np.zeros(shape=(chunk_size, self.cate_size['b']), dtype=np.int32)
        chunk['m_cate'] = np.zeros(shape=(chunk_size, self.cate_size['m']), dtype=np.int32)
        chunk['s_cate'] = np.zeros(shape=(chunk_size, self.cate_size['s']), dtype=np.int32)
        chunk['d_cate'] = np.zeros(shape=(chunk_size, self.cate_size['d']), dtype=np.int32)
        chunk['pid'] = []
        chunk['num'] = 0
        return chunk

    def copy_chunk(self, dataset, chunk, offset, with_pid_field=False):
        num = chunk['num']
        dataset['uni'][offset:offset + num, :] = chunk['uni'][:num]
        dataset['w_uni'][offset:offset + num, :] = chunk['w_uni'][:num]
        dataset['char'][offset:offset + num, :] = chunk['char'][:num]
        dataset['img'][offset:offset + num, :] = chunk['img'][:num]
        dataset['price'][offset:offset + num, :] = chunk['price'][:num]
        dataset['cate'][offset:offset + num] = chunk['cate'][:num]
        dataset['b'][offset:offset + num] = chunk['b'][:num]
        dataset['m'][offset:offset + num] = chunk['m'][:num]
        dataset['s'][offset:offset + num] = chunk['s'][:num]
        dataset['d'][offset:offset + num] = chunk['d'][:num]
        dataset['b_cate'][offset:offset + num] = chunk['b_cate'][:num]
        dataset['m_cate'][offset:offset + num] = chunk['m_cate'][:num]
        dataset['s_cate'][offset:offset + num] = chunk['s_cate'][:num]
        dataset['d_cate'][offset:offset + num] = chunk['d_cate'][:num]
        if with_pid_field:
            dataset['pid'][offset:offset + num] = chunk['pid'][:num]

    def copy_bulk(self, A, B, offset, y_offset, with_pid_field=False):
        num = B['cate'].shape[0]
        y_num = B['cate'].shape[1]
        A['uni'][offset:offset + num, :] = B['uni'][:num]
        A['w_uni'][offset:offset + num, :] = B['w_uni'][:num]
        A['char'][offset:offset + num, :] = B['char'][:num]
        A['img'][offset:offset + num, :] = B['img'][:num]
        A['price'][offset:offset + num, :] = B['price'][:num]
        A['cate'][offset:offset + num, y_offset:y_offset + y_num] = B['cate'][:num]
        A['b'][offset:offset + num, y_offset:y_offset + y_num] = B['b'][:num]
        A['m'][offset:offset + num, y_offset:y_offset + y_num] = B['m'][:num]
        A['s'][offset:offset + num, y_offset:y_offset + y_num] = B['s'][:num]
        A['d'][offset:offset + num, y_offset:y_offset + y_num] = B['d'][:num]
        A['b_cate'][offset:offset + num, y_offset:y_offset + y_num] = B['b_cate'][:num]
        A['m_cate'][offset:offset + num, y_offset:y_offset + y_num] = B['m_cate'][:num]
        A['s_cate'][offset:offset + num, y_offset:y_offset + y_num] = B['s_cate'][:num]
        A['d_cate'][offset:offset + num, y_offset:y_offset + y_num] = B['d_cate'][:num]
        if with_pid_field:
            A['pid'][offset:offset + num] = B['pid'][:num]

    def get_train_indices(self, size, train_ratio):
        train_indices = np.random.rand(size) < train_ratio
        train_size = int(np.count_nonzero(train_indices))
        return train_indices, train_size

    def make_db(self, data_name, output_dir='data/train', train_ratio=0.8):
        if data_name == 'train':
            div = 'train'
            data_path_list = opt.train_data_list
        elif data_name == 'dev':
            div = 'dev'
            data_path_list = opt.dev_data_list
        elif data_name == 'test':
            div = 'test'
            data_path_list = opt.test_data_list
        else:
            assert False, '%s is not valid data name' % data_name

        all_train = train_ratio >= 1.0
        all_dev = train_ratio == 0.0

        np.random.seed(17)
        self.logger.info('make database from data(%s) with train_ratio(%s)' % (data_name, train_ratio))

        self.load_y_vocab()
        num_input_chunks = self._preprocessing(Data,
                                               data_path_list,
                                               div,
                                               chunk_size=opt.chunk_size)
        if not os.path.isdir(output_dir):
            os.makedirs(output_dir)

        data_fout = h5py.File(os.path.join(output_dir, 'data.h5py'), 'w')
        meta_fout = open(os.path.join(output_dir, 'meta'), 'wb')

        reader = Reader(data_path_list, div, None, None)
        tmp_size = reader.get_size()
        train_indices, train_size = self.get_train_indices(tmp_size, train_ratio)

        dev_size = tmp_size - train_size
        if all_dev:
            train_size = 1
            dev_size = tmp_size
        if all_train:
            dev_size = 1
            train_size = tmp_size

        train = data_fout.create_group('train')
        dev = data_fout.create_group('dev')
        self.create_dataset(train, train_size, len(self.y_vocab))
        self.create_dataset(dev, dev_size, len(self.y_vocab))
        self.logger.info('train_size ~ %s, dev_size ~ %s' % (train_size, dev_size))

        sample_idx = 0
        dataset = {'train': train, 'dev': dev}
        num_samples = {'train': 0, 'dev': 0}
        chunk_size = opt.db_chunk_size
        chunk = {'train': self.init_chunk(chunk_size, len(self.y_vocab)),
                 'dev': self.init_chunk(chunk_size, len(self.y_vocab))}
        chunk_order = list(range(num_input_chunks))
        np.random.shuffle(chunk_order)
        for input_chunk_idx in chunk_order:
            path = os.path.join(self.tmp_chunk_tpl % input_chunk_idx)
            self.logger.info('processing %s ...' % path)
            data = list(enumerate(cPickle.loads(open(path, 'rb').read())))
            np.random.shuffle(data)
            for data_idx, (pid, y, y_cate, vw) in data:
                if y is None:
                    continue
                v, w, ch, img, price = vw
                is_train = train_indices[sample_idx + data_idx]
                if all_dev:
                    is_train = False
                if all_train:
                    is_train = True
                if v is None:
                    continue
                c = chunk['train'] if is_train else chunk['dev']
                idx = c['num']
                c['uni'][idx] = v
                c['w_uni'][idx] = w
                c['char'][idx] = ch
                c['img'][idx] = img
                c['price'][idx] = price
                c['cate'][idx] = y
                c['b'][idx] = y_cate['b']
                c['m'][idx] = y_cate['m']
                c['s'][idx] = y_cate['s']
                c['d'][idx] = y_cate['d']
                c['b_cate'][idx] = y_cate['b_cate']
                c['m_cate'][idx] = y_cate['m_cate']
                c['s_cate'][idx] = y_cate['s_cate']
                c['d_cate'][idx] = y_cate['d_cate']
                c['num'] += 1
                if not is_train:
                    c['pid'].append(np.string_(pid))
                for t in ['train', 'dev']:
                    if chunk[t]['num'] >= chunk_size:
                        self.copy_chunk(dataset[t], chunk[t], num_samples[t],
                                        with_pid_field=t == 'dev')
                        num_samples[t] += chunk[t]['num']
                        chunk[t] = self.init_chunk(chunk_size, len(self.y_vocab))
            sample_idx += len(data)
        for t in ['train', 'dev']:
            if chunk[t]['num'] > 0:
                self.copy_chunk(dataset[t], chunk[t], num_samples[t],
                                with_pid_field=t == 'dev')
                num_samples[t] += chunk[t]['num']

        for div in ['train', 'dev']:
            ds = dataset[div]
            size = num_samples[div]
            shape = (size, opt.max_len)
            ds['uni'].resize(shape)
            ds['w_uni'].resize(shape)
            ds['char'].resize(shape)
            ds['cate'].resize((size, len(self.y_vocab)))
            ds['b_cate'].resize((size, self.cate_size['b']))
            ds['m_cate'].resize((size, self.cate_size['m']))
            ds['s_cate'].resize((size, self.cate_size['s']))
            ds['d_cate'].resize((size, self.cate_size['d']))

        data_fout.close()
        meta = {'y_vocab': self.y_vocab}
        meta_fout.write(cPickle.dumps(meta, 2))
        meta_fout.close()

        self.logger.info('# of classes: %s' % len(meta['y_vocab']))
        self.logger.info('# of samples on train: %s' % num_samples['train'])
        self.logger.info('# of samples on dev: %s' % num_samples['dev'])
        self.logger.info('data: %s' % os.path.join(output_dir, 'data.h5py'))
        self.logger.info('meta: %s' % os.path.join(output_dir, 'meta'))

    def get_class(self, h, i):
        b = h['bcateid'][i]
        m = h['mcateid'][i]
        s = h['scateid'][i]
        d = h['dcateid'][i]
        return '%s>%s>%s>%s' % (b, m, s, d)

    def get_inverted_cate1(self, cate1):
        inv_cate1 = {}
        for d in ['b', 'm', 's', 'd']:
            inv_cate1[d] = {v: k for k, v in six.iteritems(cate1[d])}
        return inv_cate1

    def generate_csv(self, data_root, our_dir, data_path, target):
        self.logger.info('load from {}'.format(data_path))
        if six.PY2:
            cate1 = json.loads(open('../cate1.json').read())
        else:
            cate1 = json.loads(open('../cate1.json', 'rb').read().decode('utf-8'))

        inv_cate1 = self.get_inverted_cate1(cate1)
        del cate1

        meta_path = os.path.join(data_root, 'meta')
        meta = cPickle.loads(open(meta_path, 'rb').read())

        with h5py.File(data_path) as h:
            data= h[target]
            size = h[target]['pid'].shape[0]
            classes = [self.get_class(data, i) for i in range(size)]
            self.logger.info('{} size is {}'.format(data_path, size))
            ndata = [{}] * opt.num_workers
            work_size = size // opt.num_workers

            offset = 0
            for n in tqdm.trange(opt.num_workers):
                for k, v in data.items():
                    ndata[n][k] = v.value[offset:offset+work_size if n < opt.num_workers - 1 else size]
                offset += work_size

            pool = Pool(opt.num_workers)
            try:
                rets = pool.map_async(csv_worker,
                                      [(ndata[n], meta, inv_cate1, classes)
                                       for n in range(opt.num_workers)]).get(99999999)
                pool.close()
                pool.join()
            except KeyboardInterrupt:
                pool.terminate()
                pool.join()

        del ndata, meta, inv_cate1, classes
        df = pd.concat(rets)
        del rets

        file_name = os.path.basename(data_path)
        if not os.path.exists(our_dir):
            os.makedirs(our_dir, exist_ok=True)
        csv_name = file_name.split('.')[0]
        csv_num = file_name.split('.')[2]
        csv_file_path = os.path.join(our_dir, '{}_{}.csv'.format(csv_name, csv_num))
        df.to_csv(csv_file_path)
        self.logger.info('{} is saved'.format(csv_file_path))

    def make_csv(self, data_root, our_dir, target='train', config_path='./config.json'):
        cfg_opt = Option(config_path)
        if target == 'train':
            data_list = cfg_opt.train_data_list
        elif target == 'dev':
            data_list = cfg_opt.dev_data_list
        elif target == 'test':
            data_list = cfg_opt.test_data_list
        else:
            raise Exception('unknown target :{}'.format(target))

        rets = [make_csv_content((Data, data_root, our_dir, data_path, target)) for data_path in data_list]
        self.logger.info('{} jobs finish'.format(len(cfg_opt.train_data_list)))

    def decode_vocab(self, out_path):
        if six.PY2:
            cate1 = json.loads(open('../cate1.json').read())
        else:
            cate1 = json.loads(open('../cate1.json', 'rb').read().decode('utf-8'))

        self.load_y_vocab()
        inv_y_vocab = {v: k for k, v in six.iteritems(self.y_vocab)}
        inv_cate1 = self.get_inverted_cate1(cate1)

        rets = []
        for y in tqdm.trange(len(self.y_vocab)):
            label = inv_y_vocab[y]
            tkns = list(map(int, label.split('>')))
            b, m, s, d = tkns
            assert b in inv_cate1['b']
            assert m in inv_cate1['m']
            assert s in inv_cate1['s']
            assert d in inv_cate1['d']
            tpl = '{b}\t{m}\t{s}\t{d}'
            b = inv_cate1['b'][b]
            m = inv_cate1['m'][m]
            s = inv_cate1['s'][s]
            d = inv_cate1['d'][d]
            rets.append(tpl.format(b=b, m=m, s=s, d=d))
        with open(out_path, 'w', encoding='utf-8') as fout:
            for i in tqdm.trange(len(self.y_vocab)):
                fout.write(rets[i])
                fout.write('\n')
        self.logger.info('save to {}'.format(out_path))


if __name__ == '__main__':
    data = Data()
    fire.Fire({'make_db': data.make_db,
               'make_csv': data.make_csv,
               'decode_vocab': data.decode_vocab,
               'build_y_vocab': data.build_y_vocab})
