"""Variational-autoencoder architecture for pair-CSRO prediction.

The model maps a transformed binary-feature stack of shape ``(14, 14, 9)`` to a
``(14, 14, 1)`` matrix of Warren-Cowley CSRO parameters for all element pairs.
The architecture follows Kingma & Welling (2014) and is unchanged from the
version used to train the released ensemble weights (``data/models``).
"""
from __future__ import annotations

import os

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
import numpy as np
import tensorflow as tf

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

# Element vocabulary (row/column order of every 14x14 array in this project).
ELEMENTS = ["Al", "Ti", "V", "Cr", "Mn", "Fe", "Co", "Ni",
            "Zr", "Nb", "Mo", "Hf", "Ta", "W"]

DEFAULT_LATENT_DIM = 20
DEFAULT_LEARNING_RATE = 5e-4
RECONSTRUCTION_LOSS_FACTOR = 1000


def build_vae(input_dim, latent_space_dim=DEFAULT_LATENT_DIM):
    """Build encoder and decoder sub-models.

    Parameters
    ----------
    input_dim : tuple
        Shape of a single input sample, e.g. ``(14, 14, 9)``.
    latent_space_dim : int
        Dimension of the latent space (20 in the published model).

    Returns
    -------
    (encoder, decoder, encoder_mu, encoder_log_variance)
        ``encoder``/``decoder`` are ``tf.keras`` models; ``encoder_mu`` and
        ``encoder_log_variance`` are the latent tensors used by the loss.
    """

    def sampling(mu_log_variance):
        mu, log_variance = mu_log_variance
        epsilon = tf.keras.backend.random_normal(
            shape=tf.keras.backend.shape(mu), mean=0.0, stddev=1.0, seed=1)
        return mu + tf.keras.backend.exp(log_variance / 2) * epsilon

    x = tf.keras.layers.Input(shape=input_dim, name="encoder_input")

    h = tf.keras.layers.Conv2D(1, (3, 3), padding="same", strides=1, name="encoder_conv_1")(x)
    h = tf.keras.layers.BatchNormalization(name="encoder_norm_1")(h)
    h = tf.keras.activations.tanh(h)

    h = tf.keras.layers.Conv2D(32, (3, 3), padding="same", strides=1, name="encoder_conv_2")(h)
    h = tf.keras.layers.BatchNormalization(name="encoder_norm_2")(h)
    h = tf.keras.activations.tanh(h)

    h = tf.keras.layers.Conv2D(64, (3, 3), padding="same", strides=2, name="encoder_conv_3")(h)
    h = tf.keras.layers.BatchNormalization(name="encoder_norm_3")(h)
    h = tf.keras.activations.tanh(h)

    shape_before_flatten = tf.keras.backend.int_shape(h)[1:]
    h = tf.keras.layers.Flatten()(h)

    encoder_mu = tf.keras.layers.Dense(latent_space_dim, name="encoder_mu")(h)
    encoder_log_variance = tf.keras.layers.Dense(latent_space_dim, name="encoder_log_variance")(h)

    encoder_output = tf.keras.layers.Lambda(sampling, name="encoder_output")(
        [encoder_mu, encoder_log_variance])
    encoder = tf.keras.models.Model(x, encoder_output, name="encoder_model")

    decoder_input = tf.keras.layers.Input(shape=(latent_space_dim,), name="decoder_input")
    d = tf.keras.layers.Dense(np.prod(shape_before_flatten), name="decoder_dense_1")(decoder_input)
    d = tf.keras.layers.Reshape(shape_before_flatten)(d)

    d = tf.keras.layers.Conv2DTranspose(64, (3, 3), padding="same", strides=1, name="decoder_conv_tran_1")(d)
    d = tf.keras.layers.BatchNormalization(name="decoder_norm_1")(d)
    d = tf.keras.activations.tanh(d)

    d = tf.keras.layers.Conv2DTranspose(64, (3, 3), padding="same", strides=2, name="decoder_conv_tran_2")(d)
    d = tf.keras.layers.BatchNormalization(name="decoder_norm_2")(d)
    d = tf.keras.activations.tanh(d)

    d = tf.keras.layers.Conv2DTranspose(1, (3, 3), padding="same", strides=1, name="decoder_conv_tran_4")(d)
    decoder_output = tf.keras.activations.tanh(d)
    decoder = tf.keras.models.Model(decoder_input, decoder_output, name="decoder_model")

    return encoder, decoder, encoder_mu, encoder_log_variance


def loss_func(encoder_mu, encoder_log_variance):
    """VAE loss = scaled reconstruction MSE + KL divergence."""

    def vae_reconstruction_loss(y_true, y_predict):
        rec = tf.keras.backend.mean(tf.keras.backend.square(y_true - y_predict), axis=[1, 2, 3])
        return RECONSTRUCTION_LOSS_FACTOR * rec

    def vae_kl_loss(y_true, y_predict):
        return -0.5 * tf.keras.backend.sum(
            1.0 + encoder_log_variance - tf.keras.backend.square(encoder_mu)
            - tf.keras.backend.exp(encoder_log_variance), axis=1)

    def vae_loss(y_true, y_predict):
        return vae_reconstruction_loss(y_true, y_predict) + vae_kl_loss(y_true, y_predict)

    return vae_loss


def compile_vae(input_dim, latent_space_dim=DEFAULT_LATENT_DIM,
                learning_rate=DEFAULT_LEARNING_RATE):
    """Assemble and compile the full encoder->decoder VAE (matches the released
    weight files; use for **loading weights and inference**)."""
    encoder, decoder, mu, log_var = build_vae(input_dim, latent_space_dim)
    vae_input = tf.keras.layers.Input(shape=input_dim, name="VAE_input")
    vae = tf.keras.models.Model(vae_input, decoder(encoder(vae_input)), name="VAE")
    vae.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
                loss=loss_func(mu, log_var))
    return vae


def compile_trainable_vae(input_dim, latent_space_dim=DEFAULT_LATENT_DIM,
                          learning_rate=DEFAULT_LEARNING_RATE):
    """TF>=2.15-safe trainable VAE.

    Identical objective (scaled reconstruction MSE + KL) but the KL term is
    attached with ``add_loss`` on the in-graph latent tensors instead of a loss
    closure over external symbolic tensors (which raises under TF>=2.15 ``fit``).
    Use this for (re)training; use :func:`compile_vae` to load released weights.
    """
    encoder, decoder, _, _ = build_vae(input_dim, latent_space_dim)
    # expose mu/log_var from the encoder submodel by name
    enc_in = encoder.input
    mu = encoder.get_layer("encoder_mu").output
    log_var = encoder.get_layer("encoder_log_variance").output
    z = encoder.output
    mu_model = tf.keras.models.Model(enc_in, [z, mu, log_var], name="encoder_with_stats")

    vae_input = tf.keras.layers.Input(shape=input_dim, name="VAE_input")
    z_s, mu_s, lv_s = mu_model(vae_input)
    out = decoder(z_s)
    vae = tf.keras.models.Model(vae_input, out, name="VAE")
    kl = -0.5 * tf.reduce_mean(tf.reduce_sum(
        1.0 + lv_s - tf.square(mu_s) - tf.exp(lv_s), axis=1))
    vae.add_loss(kl)
    vae.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
                loss=lambda yt, yp: RECONSTRUCTION_LOSS_FACTOR *
                tf.reduce_mean(tf.square(yt - yp), axis=[1, 2, 3]))
    return vae
