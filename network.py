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
from keras.models import Model
from keras.layers.merge import dot
from keras.layers import Dense, Input, concatenate, BatchNormalization
from keras.layers.core import Reshape

from keras.layers.embeddings import Embedding
from keras.layers.core import Dropout, Activation

from functools import partial, update_wrapper
from metric import fbeta_score_macro, arena_score
from misc import get_logger, Option, ModelMGPU
opt = Option('./config.json')


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
    def __init__(self, vocab_matrix):
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

        # metric for kakao arena
        arena_score_metric = update_wrapper(partial(arena_score,
                                            vocab_matrix=self.vocab),
                                            arena_score)

        model.compile(loss='categorical_crossentropy',
                    optimizer=optm,
                    metrics=[top1_acc, fbeta_score_macro, arena_score_metric])
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
