# -*- coding:utf-8 -*-
import tensorflow as tf

from pgn_tf2.loss import calc_loss
import time
import os


def train_model(model, dataset, params, checkpoint_manager):
    epochs = params['epochs']

    optimizer = tf.keras.optimizers.Adagrad(params['learning_rate'],
                                            initial_accumulator_value=params['adagrad_init_acc'],
                                            clipnorm=params['max_grad_norm'],
                                            epsilon=params['eps'])

    # @tf.function(input_signature=(tf.TensorSpec(shape=[params["batch_size"], params["dec_units"]], dtype=tf.int32),
    #                               tf.TensorSpec(shape=[params["batch_size"], params["max_enc_len"]], dtype=tf.int32),
    #                               tf.TensorSpec(shape=[params["batch_size"], params["max_enc_len"]], dtype=tf.int32),
    #                               tf.TensorSpec(shape=[None], dtype=tf.int32),
    #                               tf.TensorSpec(shape=[params["batch_size"], params["max_dec_len"]], dtype=tf.int32),
    #                               tf.TensorSpec(shape=[params["batch_size"], params["max_dec_len"]], dtype=tf.int32),
    #                               tf.TensorSpec(shape=[params["batch_size"], params["max_enc_len"]], dtype=tf.int32),
    #                               tf.TensorSpec(shape=[params["batch_size"], params["max_dec_len"]], dtype=tf.int32)))
    def train_step(enc_hidden, enc_input, extended_enc_input, max_oov_len, \
                   dec_input, dec_target, enc_pad_mask, dec_pad_mask):
        with tf.GradientTape() as tape:
            # 逐个预测序列
            # encoder
            enc_output, enc_hidden = model.encoder(enc_input, enc_hidden)

            # decoder
            dec_hidden = enc_hidden
            final_dists, attentions, coverages = model(dec_input,
                                                       dec_hidden,
                                                       enc_output,
                                                       extended_enc_input,
                                                       max_oov_len,
                                                       enc_pad_mask=enc_pad_mask,
                                                       use_coverage=params['use_coverage'],
                                                       coverage=None)

            batch_loss, log_loss, cov_loss = calc_loss(dec_target,
                                                       final_dists,
                                                       dec_pad_mask,
                                                       attentions,
                                                       params['cov_loss_wt'],
                                                       params['eps'])

        variables = model.encoder.trainable_variables + model.decoder.trainable_variables + \
                    model.attention.trainable_variables + model.pointer.trainable_variables

        gradients = tape.gradient(batch_loss, variables)

        optimizer.apply_gradients(zip(gradients, variables))

        return batch_loss, log_loss, cov_loss

    best_loss = 10
    for epoch in range(epochs):
        start = time.time()
        enc_hidden = model.encoder.initialize_hidden_state()

        total_loss = 0
        total_log_loss = 0
        total_cov_loss = 0
        step = 0
        for encoder_batch_data, decoder_batch_data in dataset:
            batch_loss, log_loss, cov_loss = train_step(enc_hidden,
                                                        encoder_batch_data["enc_input"],
                                                        encoder_batch_data["extended_enc_input"],
                                                        encoder_batch_data["max_oov_len"],
                                                        decoder_batch_data["dec_input"],
                                                        decoder_batch_data["dec_target"],
                                                        enc_pad_mask=encoder_batch_data["encoder_pad_mask"],
                                                        dec_pad_mask=decoder_batch_data["decoder_pad_mask"])

            total_loss += batch_loss
            total_log_loss += log_loss
            total_cov_loss += cov_loss
            if step % 1 == 0:
                if params['use_coverage']:

                    print('Epoch {} Batch {} avg_loss {:.4f} log_loss {:.4f} cov_loss {:.4f}'.format(epoch + 1,
                                                                                                     step,
                                                                                                     total_loss / step,
                                                                                                     total_log_loss / step,
                                                                                                     total_cov_loss / step))
                else:
                    print('Epoch {} Batch {} avg_loss {:.4f}'.format(epoch + 1,
                                                                     step,
                                                                     total_loss / step))
            step += 1

        # saving (checkpoint) the model every 2 epochs
        if (epoch + 1) % 1 == 0:
            if total_loss / step < best_loss:
                best_loss = total_loss / step
                ckpt_save_path = checkpoint_manager.save()
                print('Saving checkpoint for epoch {} at {} ,best loss {}'.format(epoch + 1, ckpt_save_path, best_loss))
                print('Epoch {} Loss {:.4f}'.format(epoch + 1, total_loss / step))
                print('Time taken for 1 epoch {} sec\n'.format(time.time() - start))

