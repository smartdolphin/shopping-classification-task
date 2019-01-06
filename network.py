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
        model = TextBMSD2().get_model(cate_size)
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

        uni_embd_mat = dot([t_uni_embd, w_uni_mat], normalize=True, axes=1)
        uni_embd = Reshape((opt.embd_size, ))(uni_embd_mat)

        # cnn
        char_in = Input((max_len,), name="input_3")
        cnn_embd = Embedding(opt.char_vocab_size,
                             opt.embd_size,
                             name='cnn_embd')(char_in)
        x = Conv1D(opt.num_filters, 7,  activation='relu', padding='same',
                   kernel_initializer='he_normal')(cnn_embd)
        x = MaxPooling1D(2)(x)
        x = Conv1D(opt.num_filters, 7, activation='relu', padding='same',
                   kernel_initializer='he_normal')(x)
        x = GlobalMaxPooling1D()(x)
        x = Dropout(rate=0.5)(x)
        cnn_out = Dense(32, activation='relu', kernel_initializer='he_normal',
                      kernel_regularizer=regularizers.l2(opt.weight_decay))(x)
        #        x = BatchNormalization()(x)
        #cnn_out = Activation('relu', name='cnn_relu')(x)

        pair = concatenate([uni_embd, img_feat, cnn_out])
        out = BatchNormalization()(pair)
        out = Activation('relu', name='relu1')(out)
        out = Dropout(rate=0.5)(out)
        outputs = Dense(num_classes['b'], kernel_initializer='he_normal',
                        activation=activation, name='b_out')(out)
        model = Model(inputs=[t_uni, w_uni, char_in, img], outputs=outputs)
        if opt.graphviz:
            plot_model(model, to_file='b.png', show_shapes=True, show_layer_names=True)
        if opt.num_gpus > 1:
            model = ModelMGPU(model, gpus=opt.num_gpus)
        optm = keras.optimizers.Nadam(opt.lr)
        metrics = [top1_acc]

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
        b_model = TextB().get_model(num_classes)
        b_model.load_weights(opt.weight_path_list[0])
        embedding_weight = b_model.get_layer('uni_embd').get_weights()[0]
        b_embd = Embedding(voca_size,
                           opt.embd_size,
                           weights=[embedding_weight],
                           trainable=False,
                           name='b_embd')(t_uni)
        t_uni_embd2 = Embedding(voca_size,
                                opt.embd_size,
                                name='uni_embd2')(t_uni)
        bm_embd_seq = concatenate([b_embd, t_uni_embd2], axis=-1)
        bm_seq = SimpleRNN(opt.embd_size // 2, recurrent_dropout=0.4)(bm_embd_seq)

        b_embd_mat = dot([b_embd, w_uni_mat], normalize=True, axes=1)
        b_embd_out = Reshape((opt.embd_size,))(b_embd_mat)

        # cnn
        char_in = Input((max_len,), name="input_3")
        cnn_embd = Embedding(opt.char_vocab_size,
                             opt.embd_size,
                             name='cnn_embd')(char_in)
        x = Conv1D(opt.num_filters, 7,  activation='relu', padding='same',
                   kernel_initializer='he_normal')(cnn_embd)
        x = MaxPooling1D(2)(x)
        x = Conv1D(opt.num_filters, 7, activation='relu', padding='same',
                   kernel_initializer='he_normal')(x)
        x = GlobalMaxPooling1D()(x)
        x = Dense(32, kernel_initializer='he_normal',
                      kernel_regularizer=regularizers.l2(opt.weight_decay))(x)
        x = BatchNormalization()(x)
        cnn_out = Activation('relu', name='cnn_relu')(x)

        # cos similarity
        uni_embd_mat = dot([t_uni_embd, w_uni_mat], normalize=True, axes=1)
        uni_embd = Reshape((opt.embd_size, ))(uni_embd_mat)
        pair = concatenate([b_embd_out, uni_embd, bm_seq, img_feat, cnn_out])
        out = BatchNormalization()(pair)
        out = Activation('relu', name='m_relu')(out)
        out = Dropout(rate=0.5)(out)
        outputs = Dense(num_classes['m'], kernel_initializer='he_normal',
                        activation=activation, name='m_out')(out)
        model = Model(inputs=[t_uni, w_uni, char_in, img], outputs=outputs)
        if opt.graphviz:
            plot_model(model, to_file='m.png', show_shapes=True, show_layer_names=True)
        if opt.num_gpus > 1:
            model = ModelMGPU(model, gpus=opt.num_gpus)
        optm = keras.optimizers.Nadam(opt.lr)
        metrics = [top1_acc]

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

        # b cate
        b_model = TextB().get_model(num_classes)
        b_model.load_weights(opt.weight_path_list[0])
        b_embedding_weight = b_model.get_layer('uni_embd').get_weights()[0]
        b_embd = Embedding(voca_size,
                           opt.embd_size,
                           weights=[b_embedding_weight],
                           trainable=False,
                           name='b_embd')(t_uni)
        b_embd_mat = dot([b_embd, w_uni_mat], normalize=True, axes=1)
        b_embd_out = Reshape((opt.embd_size,))(b_embd_mat)

        # m cate
        m_model = TextM().get_model(num_classes)
        m_model.load_weights(opt.weight_path_list[1])
        m_embedding_weight = m_model.get_layer('uni_embd').get_weights()[0]
        m_embd = Embedding(voca_size,
                           opt.embd_size,
                           weights=[m_embedding_weight],
                           trainable=False,
                           name='m_embd')(t_uni)
        t_uni_embd2 = Embedding(voca_size,
                                opt.embd_size,
                                name='uni_embd2')(t_uni)
        bms_embd_seq = concatenate([b_embd, m_embd, t_uni_embd2], axis=-1)
        bms_seq = SimpleRNN(opt.embd_size // 2, recurrent_dropout=0.4)(bms_embd_seq)

        m_embd_mat = dot([m_embd, w_uni_mat], normalize=True, axes=1)
        m_embd_out = Reshape((opt.embd_size,))(m_embd_mat)

        # cnn
        char_in = Input((max_len,), name="input_3")
        cnn_embd = Embedding(opt.char_vocab_size,
                             opt.embd_size,
                             name='cnn_embd')(char_in)
        x = Conv1D(opt.num_filters, 7,  activation='relu', padding='same',
                   kernel_initializer='he_normal')(cnn_embd)
        x = MaxPooling1D(2)(x)
        x = Conv1D(opt.num_filters, 7, activation='relu', padding='same',
                   kernel_initializer='he_normal')(x)
        x = GlobalMaxPooling1D()(x)
        x = Dense(32, kernel_initializer='he_normal',
                      kernel_regularizer=regularizers.l2(opt.weight_decay))(x)
        x = BatchNormalization()(x)
        cnn_out = Activation('relu', name='cnn_relu')(x)

        # cos similarity
        uni_embd_mat = dot([t_uni_embd, w_uni_mat], normalize=True, axes=1)
        s_embd_out = Reshape((opt.embd_size, ))(uni_embd_mat)
        pair = concatenate([b_embd_out, m_embd_out, s_embd_out, bms_seq, img_feat, cnn_out])
        out = BatchNormalization()(pair)
        out = Activation('relu', name='relu')(out)
        out = Dropout(rate=0.5)(out)
        outputs = Dense(num_classes['s'], activation=activation, name='s_out')(out)
        model = Model(inputs=[t_uni, w_uni, char_in, img], outputs=outputs)
        if opt.graphviz:
            plot_model(model, to_file='s.png', show_shapes=True, show_layer_names=True)
        if opt.num_gpus > 1:
            model = ModelMGPU(model, gpus=opt.num_gpus)
        optm = keras.optimizers.Nadam(opt.lr)
        metrics = [top1_acc]

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

        # b cate
        b_model = TextB().get_model(num_classes)
        b_model.load_weights(opt.weight_path_list[0])
        b_embedding_weight = b_model.get_layer('uni_embd').get_weights()[0]
        b_embd = Embedding(voca_size,
                           opt.embd_size,
                           weights=[b_embedding_weight],
                           trainable=False,
                           name='b_embd')(t_uni)
        b_embd_mat = dot([b_embd, w_uni_mat], normalize=True, axes=1)
        b_embd_out = Reshape((opt.embd_size,))(b_embd_mat)

        # m cate
        m_model = TextM().get_model(num_classes)
        m_model.load_weights(opt.weight_path_list[1])
        m_embedding_weight = m_model.get_layer('uni_embd').get_weights()[0]
        m_embd = Embedding(voca_size,
                           opt.embd_size,
                           weights=[m_embedding_weight],
                           trainable=False,
                           name='m_embd')(t_uni)
        t_uni_embd2 = Embedding(voca_size,
                                opt.embd_size,
                                name='uni_embd2')(t_uni)
        m_embd_mat = dot([m_embd, w_uni_mat], normalize=True, axes=1)
        m_embd_out = Reshape((opt.embd_size,))(m_embd_mat)

        # s cate
        s_model = TextS().get_model(num_classes)
        s_model.load_weights(opt.weight_path_list[2])
        s_embedding_weight = s_model.get_layer('uni_embd').get_weights()[0]
        s_embd = Embedding(voca_size,
                           opt.embd_size,
                           weights=[s_embedding_weight],
                           trainable=False,
                           name='s_embd')(t_uni)
        t_uni_embd3 = Embedding(voca_size,
                                opt.embd_size,
                                name='uni_embd2')(t_uni)
        bmsd_embd_seq = concatenate([b_embd, m_embd, s_embd, t_uni_embd3], axis=-1)
        bmsd_seq = SimpleRNN(opt.embd_size // 2, recurrent_dropout=0.4)(bmsd_embd_seq)

        s_embd_mat = dot([s_embd, w_uni_mat], normalize=True, axes=1)
        s_embd_out = Reshape((opt.embd_size,))(s_embd_mat)

        # cnn
        char_in = Input((max_len,), name="input_3")
        cnn_embd = Embedding(opt.char_vocab_size,
                             opt.embd_size,
                             name='cnn_embd')(char_in)
        x = Conv1D(opt.num_filters, 7,  activation='relu', padding='same',
                   kernel_initializer='he_normal')(cnn_embd)
        x = MaxPooling1D(2)(x)
        x = Conv1D(opt.num_filters, 7, activation='relu', padding='same',
                   kernel_initializer='he_normal')(x)
        x = GlobalMaxPooling1D()(x)
        x = Dense(32, kernel_initializer='he_normal',
                      kernel_regularizer=regularizers.l2(opt.weight_decay))(x)
        x = BatchNormalization()(x)
        cnn_out = Activation('relu', name='cnn_relu')(x)

        # cos similarity
        uni_embd_mat = dot([t_uni_embd, w_uni_mat], normalize=True, axes=1)
        d_embd_out = Reshape((opt.embd_size, ))(uni_embd_mat)

        d_pair = concatenate([b_embd_out, m_embd_out, d_embd_out, s_embd_out, bmsd_seq, img_feat, cnn_out])
        d_out = BatchNormalization()(d_pair)
        d_out = Activation('relu', name='relu')(d_out)
        d_out = Dropout(rate=0.5)(d_out)
        outputs = Dense(num_classes['d'], activation=activation, name='d_out')(d_out)
        model = Model(inputs=[t_uni, w_uni, char_in, img], outputs=outputs)
        if opt.graphviz:
            plot_model(model, to_file='d.png', show_shapes=True, show_layer_names=True)
        if opt.num_gpus > 1:
            model = ModelMGPU(model, gpus=opt.num_gpus)
        optm = keras.optimizers.Nadam(opt.lr)
        metrics = [top1_acc]

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
        b_embd_out = Dropout(rate=0.5)(b_pair)
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

        embd_m = Embedding(voca_size,
                           opt.embd_size,
                           name='m_embd')
        m_embd = embd_m(t_uni)  # token
        m_uni_embd_mat = dot([m_embd, w_uni_mat], axes=1)
        m_uni_embd = Reshape((opt.embd_size, ))(m_uni_embd_mat)

        m_pair = concatenate([b_dense, m_uni_embd, img_feat])
        m_embd_out = Dropout(rate=0.5)(m_pair)
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

        s_pair = concatenate([bm_seq, s_uni_embd, img_feat])
        s_embd_out = Dropout(rate=0.5)(s_pair)
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

        d_pair = concatenate([bms_seq, d_uni_embd, img_feat])
        d_embd_out = Dropout(rate=0.5)(d_pair)
        d_relu = Activation('relu', name='d_relu')(d_embd_out)
        d_out = Dense(num_classes['d'], activation=activation, name='d_out')(d_relu)

        # bmsd cate
        bmsd_pair = concatenate([b_relu, m_relu, s_relu, d_relu])
        bmsd_pair = Dropout(rate=0.5)(bmsd_pair)
        bmsd_relu = Activation('relu', name='bmsd_relu')(bmsd_pair)
        bmsd_out = Dense(num_classes['bmsd'], activation=activation, name='bmsd_out')(bmsd_relu)

        model = Model(inputs=[t_uni, w_uni, char_in, img, b_in, bm_in, bms_in],
                      outputs=[b_out, m_out, s_out, d_out, bmsd_out])
        if opt.graphviz:
            plot_model(model, to_file='bmsd.png', show_shapes=True, show_layer_names=True)
        if opt.num_gpus > 1:
            model = ModelMGPU(model, gpus=opt.num_gpus)
        optm = keras.optimizers.Nadam(opt.lr)
        metrics = [top1_acc]

        model.compile(loss='categorical_crossentropy',
                      optimizer=optm,
                      metrics=metrics)
        model.summary(print_fn=lambda x: self.logger.info(x))
        return model


class TextBMSD2:
    def __init__(self, vocab_matrix=None):
        self.logger = get_logger('text_bmsd2')
        self.vocab = vocab_matrix

    def get_model(self, num_classes, activation='softmax'):
        max_len = opt.max_len
        voca_size = opt.unigram_hash_size + 1
        # image feature
        img = Input((opt.img_size,), name="input_img")
        img_feat = Reshape((opt.img_size, ))(img)

        # text
        embd = Embedding(voca_size,
                         opt.embd_size,
                         name='embd')

        t_uni = Input((max_len,), name="input_1")
        uni_embd = embd(t_uni)  # token

        w_uni = Input((max_len,), name="input_2")
        w_uni_mat = Reshape((max_len, 1))(w_uni)  # weight

        uni_embd_mat = dot([uni_embd, w_uni_mat], normalize=True, axes=1)
        uni_embd = Reshape((opt.embd_size, ))(uni_embd_mat)

        # cate only
        bmsd_in = Input((4,), name="input_bmsd")
        embd_bmsd_seq = Embedding(num_classes['b'] + num_classes['m'] + num_classes['s'] + num_classes['d'],
                                 opt.embd_size,
                                 name='bmsd_embd_seq')(bmsd_in)
        bmsd_seq = SimpleRNN(opt.embd_size, recurrent_dropout=0.4)(embd_bmsd_seq)

        pair = concatenate([bmsd_seq, uni_embd, img_feat])
        out = BatchNormalization()(pair)
        out = Activation('relu', name='d_relu')(out)
        out = Dropout(rate=0.5)(out)
        out = Dense(num_classes['bmsd'], activation=activation, name='out')(out)

        model = Model(inputs=[t_uni, w_uni, img, bmsd_in],
                      outputs=[out])
        if opt.graphviz:
            plot_model(model, to_file='cate_bmsd.png', show_shapes=True, show_layer_names=True)
        if opt.num_gpus > 1:
            model = ModelMGPU(model, gpus=opt.num_gpus)
        optm = keras.optimizers.Nadam(opt.lr)
        metrics = [top1_acc]

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
