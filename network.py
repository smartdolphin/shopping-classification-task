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
import tensorflow as tf

import keras
from keras import regularizers
from keras.models import Model
from keras.layers.merge import dot
from keras.layers import Dense, Input, concatenate, BatchNormalization, SimpleRNN
from keras.layers import Conv1D, MaxPooling1D, GlobalMaxPooling1D
from keras.layers.core import Reshape

from keras.layers.embeddings import Embedding
from keras.layers.core import Dropout, Activation
from keras.utils.vis_utils import plot_model

from functools import partial, update_wrapper
from metric import fbeta_score_macro, arena_score
from misc import get_logger, Option, ModelMGPU
opt = Option('./config.json')


def get_model(target_model, cate_size):
    if target_model == 'b':
        model = TextB().get_model(cate_size)
    elif target_model == 'm':
        model = TextM().get_model(cate_size)
    elif target_model == 's':
        model = TextS().get_model(cate_size)
    elif target_model == 'd':
        model = TextD().get_model(cate_size)
    elif target_model == 'bmsd':
        model = TextBMSD().get_model(cate_size)
    else:
        raise Exception('Unknown model: {}'.format(target_model))
    return model


def top1_acc(x, y):
    return keras.metrics.top_k_categorical_accuracy(x, y, k=1)


class TextOnly:
    def __init__(self):
        self.logger = get_logger('textonly')

    def get_model(self, num_classes, activation='sigmoid'):
        max_len = opt.max_len
        voca_size = opt.unigram_hash_size + 1

        with tf.device('/gpu:0'):
            embd = Embedding(voca_size,
                             opt.embd_size,
                             name='uni_embd')

            t_uni = Input((max_len,), name="input_1")
            t_uni_embd = embd(t_uni)  # token

            w_uni = Input((max_len,), name="input_2")
            w_uni_mat = Reshape((max_len, 1))(w_uni)  # weight

            uni_embd_mat = dot([t_uni_embd, w_uni_mat], axes=1)
            uni_embd = Reshape((opt.embd_size, ))(uni_embd_mat)

            embd_out = Dropout(rate=0.5)(uni_embd)
            relu = Activation('relu', name='relu1')(embd_out)
            outputs = Dense(num_classes, activation=activation)(relu)
            model = Model(inputs=[t_uni, w_uni], outputs=outputs)
            optm = keras.optimizers.Nadam(opt.lr)
            model.compile(loss='binary_crossentropy',
                          optimizer=optm,
                          metrics=[top1_acc])
            model.summary(print_fn=lambda x: self.logger.info(x))
        return model


class TextImage:
    def __init__(self, vocab_matrix=None):
        self.logger = get_logger('text_img')
        self.vocab = vocab_matrix

    def get_model(self, num_classes, activation='softmax'):
        max_len = opt.max_len
        voca_size = opt.unigram_hash_size + 1

        embd = Embedding(voca_size,
                         opt.embd_size,
                         name='uni_embd')

        t_uni = Input((max_len,), name="input_1")
        t_uni_embd = embd(t_uni)  # token

        w_uni = Input((max_len,), name="input_2")
        w_uni_mat = Reshape((max_len, 1))(w_uni)  # weight

        # image feature
        img = Input((opt.img_size,), name="input_3")

        uni_embd_mat = dot([t_uni_embd, w_uni_mat], axes=1)
        uni_embd = Reshape((opt.embd_size, ))(uni_embd_mat)
        img_feat = Reshape((opt.img_size, ))(img)
        pair = concatenate([uni_embd, img_feat])
        embd_out = BatchNormalization()(pair)
        relu = Activation('relu', name='relu1')(embd_out)
        outputs = Dense(num_classes, activation=activation)(relu)
        model = Model(inputs=[t_uni, w_uni, img], outputs=outputs)
        if opt.num_gpus > 1:
            model = ModelMGPU(model, gpus=opt.num_gpus)
        optm = keras.optimizers.Nadam(opt.lr)
        metrics = [top1_acc, fbeta_score_macro]

        # metric for kakao arena
        arena_score_metric = update_wrapper(partial(arena_score,
                                            vocab_matrix=self.vocab),
                                            arena_score)
        metrics += [arena_score_metric] if self.vocab is not None else []

        model.compile(loss='categorical_crossentropy',
                      optimizer=optm,
                      metrics=metrics)
        model.summary(print_fn=lambda x: self.logger.info(x))
        return model


class TextB:
    def __init__(self, vocab_matrix=None):
        self.logger = get_logger('text_b')
        self.vocab = vocab_matrix

    def get_model(self, num_classes, activation='softmax'):
        max_len = opt.max_len
        voca_size = opt.unigram_hash_size + 1

        # image feature
        img = Input((opt.img_size,), name="input_img")
        img_feat = Reshape((opt.img_size, ))(img)

        embd = Embedding(voca_size,
                         opt.embd_size,
                         name='uni_embd')

        t_uni = Input((max_len,), name="input_1")
        t_uni_embd = embd(t_uni)  # token

        w_uni = Input((max_len,), name="input_2")
        w_uni_mat = Reshape((max_len, 1))(w_uni)  # weight

        uni_embd_mat = dot([t_uni_embd, w_uni_mat], axes=1)
        uni_embd = Reshape((opt.embd_size, ))(uni_embd_mat)

        # cnn
        char_in = Input((max_len,), name="input_3")
        cnn_embd = Embedding(opt.char_vocab_size,
                             opt.embd_size,
                             name='cnn_embd')(char_in)
        x = Conv1D(opt.num_filters, 7,  activation='relu', padding='same')(cnn_embd)
        x = MaxPooling1D(2)(x)
        x = Conv1D(opt.num_filters, 7, activation='relu', padding='same')(x)
        x = GlobalMaxPooling1D()(x)
        x = Dropout(0.5)(x)
        cnn_out = Dense(32, activation='relu', kernel_regularizer=regularizers.l2(opt.weight_decay))(x)
        pair = concatenate([uni_embd, img_feat, cnn_out])
        embd_out = BatchNormalization()(pair)
        relu = Activation('relu', name='relu')(embd_out)
        outputs = Dense(num_classes['b'], activation=activation, name='b_out')(relu)
        model = Model(inputs=[t_uni, w_uni, char_in, img], outputs=outputs)
        if opt.graphviz:
            plot_model(model, to_file='b.png', show_shapes=True, show_layer_names=True)
        if opt.num_gpus > 1:
            model = ModelMGPU(model, gpus=opt.num_gpus)
        optm = keras.optimizers.Nadam(opt.lr)
        metrics = [top1_acc, fbeta_score_macro]

        model.compile(loss='categorical_crossentropy',
                      optimizer=optm,
                      metrics=metrics)
        model.summary(print_fn=lambda x: self.logger.info(x))
        return model


class TextM:
    def __init__(self, vocab_matrix=None):
        self.logger = get_logger('text_m')
        self.vocab = vocab_matrix

    def get_model(self, num_classes, activation='softmax'):
        max_len = opt.max_len
        voca_size = opt.unigram_hash_size + 1

        # image feature
        img = Input((opt.img_size,), name="input_img")
        img_feat = Reshape((opt.img_size, ))(img)

        embd = Embedding(voca_size,
                         opt.embd_size,
                         name='uni_embd')

        t_uni = Input((max_len,), name="input_1")
        t_uni_embd = embd(t_uni)  # token

        w_uni = Input((max_len,), name="input_2")
        w_uni_mat = Reshape((max_len, 1))(w_uni)  # weight

        # b cate
        b_in = Input((1,), name="input_b")
        b_dense = Dense(opt.embd_size // 4)(b_in)
        b_dense = BatchNormalization()(b_dense)
        b_dense = Activation('relu', name='b_relu_1')(b_dense)
        b_dense = Dense(opt.embd_size // 4)(b_dense)
        b_dense = BatchNormalization()(b_dense)
        b_dense = Activation('relu', name='b_relu_2')(b_dense)
        b_dense = Dropout(rate=0.5)(b_dense)

        # cnn
        char_in = Input((max_len,), name="input_c")
        cnn_embd = Embedding(opt.char_vocab_size,
                             opt.embd_size,
                             name='cnn_embd')(char_in)
        x = Conv1D(opt.num_filters, 7,  activation='relu', padding='same')(cnn_embd)
        x = MaxPooling1D(2)(x)
        x = Conv1D(opt.num_filters, 7, activation='relu', padding='same')(x)
        x = GlobalMaxPooling1D()(x)
        x = Dropout(0.5)(x)
        cnn_out = Dense(32, activation='relu', kernel_regularizer=regularizers.l2(opt.weight_decay))(x)

        uni_embd_mat = dot([t_uni_embd, w_uni_mat], axes=1)
        uni_embd = Reshape((opt.embd_size, ))(uni_embd_mat)
        pair = concatenate([b_dense, uni_embd, img_feat, cnn_out])
        embd_out = BatchNormalization()(pair)
        relu = Activation('relu', name='m_relu')(embd_out)
        outputs = Dense(num_classes['m'], activation=activation, name='m_out')(relu)
        model = Model(inputs=[t_uni, w_uni, char_in, img, b_in], outputs=outputs)
        if opt.graphviz:
            plot_model(model, to_file='m.png', show_shapes=True, show_layer_names=True)
        if opt.num_gpus > 1:
            model = ModelMGPU(model, gpus=opt.num_gpus)
        optm = keras.optimizers.Nadam(opt.lr)
        metrics = [top1_acc, fbeta_score_macro]

        model.compile(loss='categorical_crossentropy',
                      optimizer=optm,
                      metrics=metrics)
        model.summary(print_fn=lambda x: self.logger.info(x))
        return model


class TextS:
    def __init__(self, vocab_matrix=None):
        self.logger = get_logger('text_s')
        self.vocab = vocab_matrix

    def get_model(self, num_classes, activation='softmax'):
        max_len = opt.max_len
        voca_size = opt.unigram_hash_size + 1

        # image feature
        img = Input((opt.img_size,), name="input_img")
        img_feat = Reshape((opt.img_size, ))(img)

        embd = Embedding(voca_size,
                         opt.embd_size,
                         name='uni_embd')

        t_uni = Input((max_len,), name="input_1")
        t_uni_embd = embd(t_uni)  # token

        w_uni = Input((max_len,), name="input_2")
        w_uni_mat = Reshape((max_len, 1))(w_uni)  # weight

        uni_embd_mat = dot([t_uni_embd, w_uni_mat], axes=1)
        uni_embd = Reshape((opt.embd_size, ))(uni_embd_mat)

        # sequence model
        bm_in = Input((2,), name="input_bm")
        embd_bm_seq = Embedding(num_classes['b'] + num_classes['m'],
                                opt.embd_size,
                                name='bm_embd_seq')(bm_in)
        bm_seq = SimpleRNN(opt.embd_size // 2)(embd_bm_seq)

        # cnn
        char_in = Input((max_len,), name="input_c")
        cnn_embd = Embedding(opt.char_vocab_size,
                             opt.embd_size,
                             name='cnn_embd')(char_in)
        x = Conv1D(opt.num_filters, 7,  activation='relu', padding='same')(cnn_embd)
        x = MaxPooling1D(2)(x)
        x = Conv1D(opt.num_filters, 7, activation='relu', padding='same')(x)
        x = GlobalMaxPooling1D()(x)
        x = Dropout(0.5)(x)
        cnn_out = Dense(32, activation='relu', kernel_regularizer=regularizers.l2(opt.weight_decay))(x)

        s_pair = concatenate([bm_seq, uni_embd, img_feat, cnn_out])
        s_embd_out = Dropout(rate=0.5)(s_pair)
        s_relu = Activation('relu', name='relu')(s_embd_out)
        outputs = Dense(num_classes['s'], activation=activation, name='s_out')(s_relu)
        model = Model(inputs=[t_uni, w_uni, char_in, img, bm_in], outputs=outputs)
        if opt.graphviz:
            plot_model(model, to_file='s.png', show_shapes=True, show_layer_names=True)
        if opt.num_gpus > 1:
            model = ModelMGPU(model, gpus=opt.num_gpus)
        optm = keras.optimizers.Nadam(opt.lr)
        metrics = [top1_acc, fbeta_score_macro]

        model.compile(loss='categorical_crossentropy',
                      optimizer=optm,
                      metrics=metrics)
        model.summary(print_fn=lambda x: self.logger.info(x))
        return model


class TextD:
    def __init__(self, vocab_matrix=None):
        self.logger = get_logger('text_d')
        self.vocab = vocab_matrix

    def get_model(self, num_classes, activation='softmax'):
        max_len = opt.max_len
        voca_size = opt.unigram_hash_size + 1

        # image feature
        img = Input((opt.img_size,), name="input_img")
        img_feat = Reshape((opt.img_size, ))(img)

        embd = Embedding(voca_size,
                         opt.embd_size,
                         name='uni_embd')

        t_uni = Input((max_len,), name="input_1")
        t_uni_embd = embd(t_uni)  # token

        w_uni = Input((max_len,), name="input_2")
        w_uni_mat = Reshape((max_len, 1))(w_uni)  # weight

        uni_embd_mat = dot([t_uni_embd, w_uni_mat], axes=1)
        uni_embd = Reshape((opt.embd_size, ))(uni_embd_mat)

        # d cate
        bms_in = Input((3,), name="input_bms")
        embd_bms_seq = Embedding(num_classes['b'] + num_classes['m'] + num_classes['s'],
                                 opt.embd_size,
                                 name='bms_embd_seq')(bms_in)
        bms_seq = SimpleRNN(opt.embd_size // 2)(embd_bms_seq)

        # cnn
        char_in = Input((max_len,), name="input_c")
        cnn_embd = Embedding(opt.char_vocab_size,
                             opt.embd_size,
                             name='cnn_embd')(char_in)
        x = Conv1D(opt.num_filters, 7,  activation='relu', padding='same')(cnn_embd)
        x = MaxPooling1D(2)(x)
        x = Conv1D(opt.num_filters, 7, activation='relu', padding='same')(x)
        x = GlobalMaxPooling1D()(x)
        x = Dropout(0.5)(x)
        cnn_out = Dense(32, activation='relu', kernel_regularizer=regularizers.l2(opt.weight_decay))(x)

        d_pair = concatenate([bms_seq, uni_embd, img_feat, cnn_out])
        d_embd_out = BatchNormalization()(d_pair)
        d_relu = Activation('relu', name='relu')(d_embd_out)
        outputs = Dense(num_classes['d'], activation=activation, name='d_out')(d_relu)
        model = Model(inputs=[t_uni, w_uni, char_in, img, bms_in], outputs=outputs)
        if opt.graphviz:
            plot_model(model, to_file='d.png', show_shapes=True, show_layer_names=True)
        if opt.num_gpus > 1:
            model = ModelMGPU(model, gpus=opt.num_gpus)
        optm = keras.optimizers.Nadam(opt.lr)
        metrics = [top1_acc, fbeta_score_macro]

        model.compile(loss='categorical_crossentropy',
                      optimizer=optm,
                      metrics=metrics)
        model.summary(print_fn=lambda x: self.logger.info(x))
        return model


class TextBMSD:
    def __init__(self, vocab_matrix=None):
        self.logger = get_logger('text_bmsd')
        self.vocab = vocab_matrix

    def get_model(self, num_classes, activation='softmax'):
        max_len = opt.max_len
        voca_size = opt.unigram_hash_size + 1
        # image feature
        img = Input((opt.img_size,), name="input_img")
        img_feat = Reshape((opt.img_size, ))(img)

        # char input
        char_in = Input((max_len,), name="input_c")

        # b cate
        embd_b = Embedding(voca_size,
                         opt.embd_size,
                         name='b_embd')

        t_uni = Input((max_len,), name="input_1")
        b_embd = embd_b(t_uni)  # token

        w_uni = Input((max_len,), name="input_2")
        w_uni_mat = Reshape((max_len, 1))(w_uni)  # weight

        b_uni_embd_mat = dot([b_embd, w_uni_mat], axes=1)
        b_uni_embd = Reshape((opt.embd_size, ))(b_uni_embd_mat)

        # b cnn
        b_cnn_embd = Embedding(opt.char_vocab_size,
                               opt.embd_size,
                               name='b_cnn_embd')(char_in)
        b_x = Conv1D(opt.num_filters, 7,  activation='relu', padding='same')(b_cnn_embd)
        b_x = MaxPooling1D(2)(b_x)
        b_x = Conv1D(opt.num_filters, 7, activation='relu', padding='same')(b_x)
        b_x = GlobalMaxPooling1D()(b_x)
        b_x = Dropout(0.5)(b_x)
        b_cnn_out = Dense(32, activation='relu', kernel_regularizer=regularizers.l2(opt.weight_decay))(b_x)

        b_pair = concatenate([b_uni_embd, img_feat, b_cnn_out])
        b_embd_out = BatchNormalization()(b_pair)
        b_relu = Activation('relu', name='b_relu')(b_embd_out)
        b_out = Dense(num_classes['b'], activation=activation, name='b_out')(b_relu)

        # m cate
        b_in = Input((1,), name="input_b")
        b_dense = Dense(opt.embd_size // 4)(b_in)
        b_dense = BatchNormalization()(b_dense)
        b_dense = Activation('relu', name='b_relu_1')(b_dense)
        b_dense = Dense(opt.embd_size // 4)(b_dense)
        b_dense = BatchNormalization()(b_dense)
        b_dense = Activation('relu', name='b_relu_2')(b_dense)
        b_dense = Dropout(rate=0.5)(b_dense)

        embd_m = Embedding(voca_size,
                           opt.embd_size,
                           name='m_embd')
        m_embd = embd_m(t_uni)  # token
        m_uni_embd_mat = dot([m_embd, w_uni_mat], axes=1)
        m_uni_embd = Reshape((opt.embd_size, ))(m_uni_embd_mat)

        # m cnn
        m_cnn_embd = Embedding(opt.char_vocab_size,
                               opt.embd_size,
                               name='m_cnn_embd')(char_in)
        m_x = Conv1D(opt.num_filters, 7,  activation='relu', padding='same')(m_cnn_embd)
        m_x = MaxPooling1D(2)(m_x)
        m_x = Conv1D(opt.num_filters, 7, activation='relu', padding='same')(m_x)
        m_x = GlobalMaxPooling1D()(m_x)
        m_x = Dropout(0.5)(m_x)
        m_cnn_out = Dense(32, activation='relu', kernel_regularizer=regularizers.l2(opt.weight_decay))(m_x)

        m_pair = concatenate([b_dense, m_uni_embd, img_feat, m_cnn_out])
        m_embd_out = BatchNormalization()(m_pair)
        m_relu = Activation('relu', name='m_relu')(m_embd_out)
        m_out = Dense(num_classes['m'], activation=activation, name='m_out')(m_relu)

        # s cate
        embd_s = Embedding(voca_size,
                           opt.embd_size,
                           name='s_embd')
        s_embd = embd_s(t_uni)  # token
        s_uni_embd_mat = dot([s_embd, w_uni_mat], axes=1)
        s_uni_embd = Reshape((opt.embd_size, ))(s_uni_embd_mat)

        bm_in = Input((2,), name="input_bm")
        embd_bm_seq = Embedding(num_classes['b'] + num_classes['m'],
                                opt.embd_size,
                                name='bm_embd_seq')(bm_in)
        bm_seq = SimpleRNN(opt.embd_size // 2)(embd_bm_seq)

        # s cnn
        s_cnn_embd = Embedding(opt.char_vocab_size,
                               opt.embd_size,
                               name='s_cnn_embd')(char_in)
        s_x = Conv1D(opt.num_filters, 7,  activation='relu', padding='same')(s_cnn_embd)
        s_x = MaxPooling1D(2)(s_x)
        s_x = Conv1D(opt.num_filters, 7, activation='relu', padding='same')(s_x)
        s_x = GlobalMaxPooling1D()(s_x)
        s_x = Dropout(0.5)(s_x)
        s_cnn_out = Dense(32, activation='relu', kernel_regularizer=regularizers.l2(opt.weight_decay))(s_x)

        s_pair = concatenate([bm_seq, s_uni_embd, img_feat, s_cnn_out])
        s_embd_out = BatchNormalization()(s_pair)
        s_relu = Activation('relu', name='s_relu')(s_embd_out)
        s_out = Dense(num_classes['s'], activation=activation, name='s_out')(s_relu)

        # d cate
        embd_d = Embedding(voca_size,
                           opt.embd_size,
                           name='d_embd')
        d_embd = embd_d(t_uni)  # token
        d_uni_embd_mat = dot([d_embd, w_uni_mat], axes=1)
        d_uni_embd = Reshape((opt.embd_size, ))(d_uni_embd_mat)

        bms_in = Input((3,), name="input_bms")
        embd_bms_seq = Embedding(num_classes['b'] + num_classes['m'] + num_classes['s'],
                                 opt.embd_size,
                                 name='bms_embd_seq')(bms_in)
        bms_seq = SimpleRNN(opt.embd_size // 2)(embd_bms_seq)

        # d cnn
        d_cnn_embd = Embedding(opt.char_vocab_size,
                               opt.embd_size,
                               name='d_cnn_embd')(char_in)
        d_x = Conv1D(opt.num_filters, 7,  activation='relu', padding='same')(d_cnn_embd)
        d_x = MaxPooling1D(2)(d_x)
        d_x = Conv1D(opt.num_filters, 7, activation='relu', padding='same')(d_x)
        d_x = GlobalMaxPooling1D()(d_x)
        d_x = Dropout(0.5)(d_x)
        d_cnn_out = Dense(32, activation='relu', kernel_regularizer=regularizers.l2(opt.weight_decay))(d_x)

        d_pair = concatenate([bms_seq, d_uni_embd, img_feat, d_cnn_out])
        d_embd_out = BatchNormalization()(d_pair)
        d_relu = Activation('relu', name='d_relu')(d_embd_out)
        d_out = Dense(num_classes['d'], activation=activation, name='d_out')(d_relu)

        # bmsd cate
        bmsd_pair = concatenate([b_out, m_out, s_out, d_out])
        bmsd_pair = BatchNormalization()(bmsd_pair)
        bmsd_relu = Activation('relu', name='bmsd_relu')(bmsd_pair)
        bmsd_out = Dense(num_classes['bmsd'], activation=activation, name='bmsd_out')(bmsd_relu)

        model = Model(inputs=[t_uni, w_uni, char_in, img, b_in, bm_in, bms_in],
                      outputs=[b_out, m_out, s_out, d_out, bmsd_out])
        if opt.graphviz:
            plot_model(model, to_file='bmsd.png', show_shapes=True, show_layer_names=True)
        if opt.num_gpus > 1:
            model = ModelMGPU(model, gpus=opt.num_gpus)
        optm = keras.optimizers.Nadam(opt.lr)
        metrics = [top1_acc, fbeta_score_macro]

        model.compile(loss='categorical_crossentropy',
                      optimizer=optm,
                      metrics=metrics)
        model.summary(print_fn=lambda x: self.logger.info(x))
        return model


class TextImageNN:
    def __init__(self):
        self.logger = get_logger('text_img_nn')

    def get_model(self, num_classes, activation='softmax'):
        max_len = opt.max_len
        voca_size = opt.unigram_hash_size + 1

        embd = Embedding(voca_size,
                         opt.embd_size,
                         name='uni_embd')

        t_uni = Input((max_len,), name="input_1")
        t_uni_embd = embd(t_uni)  # token

        w_uni = Input((max_len,), name="input_2")
        w_uni_mat = Reshape((max_len, 1))(w_uni)  # weight

        # image feature
        img = Input((opt.img_size,), name="input_3")

        uni_embd_mat = dot([t_uni_embd, w_uni_mat], axes=1)
        uni_embd = Reshape((opt.embd_size, ))(uni_embd_mat)
        img_feat = Reshape((opt.img_size, ))(img)
        pair = concatenate([uni_embd, img_feat])
        x = Dropout(rate=0.5)(pair)
        x = Dense(opt.hidden_size, activation='relu')(x)
        x = Dropout(rate=0.5)(x)
        x = Dense(opt.hidden_size // 2, activation='relu')(x)
        x = Dropout(rate=0.5)(x)
        outputs = Dense(num_classes, activation=activation)(x)
        model = Model(inputs=[t_uni, w_uni, img], outputs=outputs)
        if opt.num_gpus > 1:
            model = ModelMGPU(model, gpus=opt.num_gpus)
        optm = keras.optimizers.Nadam(opt.lr)
        model.compile(loss='categorical_crossentropy',
                    optimizer=optm,
                    metrics=[top1_acc])
        model.summary(print_fn=lambda x: self.logger.info(x))
        return model


class TextImagePrice:
    def __init__(self):
        self.logger = get_logger('text_img_price')

    def get_model(self, num_classes, activation='softmax'):
        max_len = opt.max_len
        voca_size = opt.unigram_hash_size + 1

        embd = Embedding(voca_size,
                         opt.embd_size,
                         name='uni_embd')

        t_uni = Input((max_len,), name="input_1")
        t_uni_embd = embd(t_uni)  # token

        w_uni = Input((max_len,), name="input_2")
        w_uni_mat = Reshape((max_len, 1))(w_uni)  # weight

        # image feature
        img = Input((opt.img_size,), name="input_3")

        # price feature
        price = Input((1,), name="input_4")

        uni_embd_mat = dot([t_uni_embd, w_uni_mat], axes=1)
        uni_embd = Reshape((opt.embd_size, ))(uni_embd_mat)
        img_feat = Reshape((opt.img_size, ))(img)
        price_feat = Reshape((1, ))(price)
        pair = concatenate([uni_embd, img_feat, price_feat])
        embd_out = Dropout(rate=0.5)(pair)
        relu = Activation('relu', name='relu1')(embd_out)
        outputs = Dense(num_classes, activation=activation)(relu)
        model = Model(inputs=[t_uni, w_uni, img, price], outputs=outputs)
        optm = keras.optimizers.Nadam(opt.lr)
        model.compile(loss='categorical_crossentropy',
                    optimizer=optm,
                    metrics=[top1_acc])
        model.summary(print_fn=lambda x: self.logger.info(x))
        return model


class TextImagePriceNN:
    def __init__(self):
        self.logger = get_logger('text_img_price_nn')

    def get_model(self, num_classes, activation='softmax'):
        max_len = opt.max_len
        hidden_size = opt.hidden_size
        voca_size = opt.unigram_hash_size + 1

        embd = Embedding(voca_size,
                         opt.embd_size,
                         name='uni_embd')

        t_uni = Input((max_len,), name="input_1")
        t_uni_embd = embd(t_uni)  # token

        w_uni = Input((max_len,), name="input_2")
        w_uni_mat = Reshape((max_len, 1))(w_uni)  # weight

        # image feature
        img = Input((opt.img_size,), name="input_3")

        # price feature
        price = Input((1,), name="input_4")

        text_embd = dot([t_uni_embd, w_uni_mat], axes=1)
        text_embd = Reshape((opt.embd_size, ))(text_embd)
        img_feat = Reshape((opt.img_size, ))(img)
        price_feat = Reshape((1, ))(price)
        x = concatenate([text_embd, img_feat, price_feat])
        x = Dropout(rate=0.5)(x)
        x = Dense(hidden_size, activation='relu')(x)
        x = Dropout(rate=0.5)(x)
        x = Dense(hidden_size // 2, activation='relu')(x)
        x = Dropout(rate=0.5)(x)
        outputs = Dense(num_classes, activation=activation)(x)
        model = Model(inputs=[t_uni, w_uni, img, price], outputs=outputs)
        optm = keras.optimizers.Nadam(opt.lr)
        model.compile(loss='categorical_crossentropy',
                    optimizer=optm,
                    metrics=[top1_acc])
        model.summary(print_fn=lambda x: self.logger.info(x))
        return model
