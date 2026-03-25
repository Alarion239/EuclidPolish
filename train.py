import os

from scipy import signal
import numpy as np
import time
import tensorflow as tf

from model import evaluate

from tf_keras.applications.vgg19 import preprocess_input
from tf_keras.losses import BinaryCrossentropy
from tf_keras.losses import MeanAbsoluteError
from tf_keras.losses import MeanSquaredError
from tf_keras.metrics import Mean
from tf_keras.optimizers import Adam
from tf_keras.optimizers.schedules import PiecewiseConstantDecay

class Trainer:
    def __init__(self,
                 model,
                 loss=MeanAbsoluteError(),
                 learning_rate=PiecewiseConstantDecay(boundaries=[200000], values=[1e-3, 5e-4]),
                 checkpoint_dir='./ckpt/edsr',
                 nbit=16,
                 fn_kernel=None):

        self.now = None
        self.loss = loss
        self.checkpoint = tf.train.Checkpoint(step=tf.Variable(0),
                                              psnr=tf.Variable(-1.0),
                                              optimizer=Adam(learning_rate),
                                              model=model)
        self.checkpoint_manager = tf.train.CheckpointManager(checkpoint=self.checkpoint,
                                                             directory=checkpoint_dir,
                                                             max_to_keep=3)

        self.restore()
        if fn_kernel is not None:
            self.kernel = np.load(fn_kernel)
        else:
            self.kernel = None

    @property
    def model(self):
        return self.checkpoint.model

    def train(self, train_dataset, valid_dataset, steps=300000,
              evaluate_every=1000, save_best_only=True, nbit=16):
        loss_mean = Mean()

        ckpt_mgr = self.checkpoint_manager
        ckpt = self.checkpoint

        self.now = time.perf_counter()

        for lr, hr in train_dataset.take(steps - ckpt.step.numpy()):
            ckpt.step.assign_add(1)
            step = ckpt.step.numpy()
            loss = self.train_step(lr, hr)
            loss_mean(loss)


            if step % evaluate_every == 0:
                loss_value = loss_mean.result()
                loss_mean.reset_state()

                # Compute PSNR on validation dataset
                psnr_value = self.evaluate(valid_dataset, nbit=nbit)

                duration = time.perf_counter() - self.now
                print(f'{step}/{steps}: loss = {loss_value.numpy():.3f}, PSNR = {psnr_value.numpy():3f} ({duration:.2f}s)')

                if save_best_only and psnr_value <= ckpt.psnr:
                    self.now = time.perf_counter()
                    continue

                ckpt.psnr = psnr_value
                ckpt_mgr.save()

                self.now = time.perf_counter()

    def kernel_loss(self, sr, lr):
        lr_estimate = signal.fftconvolve(sr.numpy(), self.kernel, mode='same')

        print(lr.shape, lr_estimate[2::4, 2::4].shape)
        exit()

    @tf.function
    def train_step(self, lr, hr, gg=1.):
        with tf.GradientTape() as tape:
            lr = tf.cast(lr, tf.float32)
            hr = tf.cast(hr, tf.float32)
            sr = self.checkpoint.model(lr, training=True)
            loss_value = self.loss(sr, hr)

        gradients = tape.gradient(loss_value, self.checkpoint.model.trainable_variables)
        self.checkpoint.optimizer.apply_gradients(zip(gradients, self.checkpoint.model.trainable_variables))

        return loss_value


    def evaluate(self, dataset, nbit=16):
        return evaluate(self.checkpoint.model, dataset, nbit=nbit)

    def restore(self):
        if self.checkpoint_manager.latest_checkpoint:
            self.checkpoint.restore(self.checkpoint_manager.latest_checkpoint)
            print(f'Model restored from checkpoint at step {self.checkpoint.step.numpy()}.')
