import numpy as np
from tqdm import tqdm
import time, os
import tensorflow as tf
from tensorflow.contrib.rnn import LSTMCell
from tensorflow.contrib.layers import xavier_initializer
from tensorflow.contrib.crf import crf_log_likelihood
from tensorflow.contrib.crf import viterbi_decode
from data_loader import load_pretrain_embedding, batch_yield, pad_sequences


class BiLSTM_CRF(object):
    def __init__(self, args, tag2label, vocab, log_path, logger, config, pretrain_embedding=None):
        self.batch_size = args.batch_size
        self.epoch_num = args.epoch
        self.hidden_size = args.hidden_units
        self.CRF = args.CRF
        self.embedding_size = args.embedding_size
        self.update_embedding = args.update_embedding
        self.pretrain_embedding = pretrain_embedding
        self.dropout_keep_prob = args.dropout
        self.optimizer = args.optimizer
        self.lr_pl = args.lr
        self.clip_grad = args.clip
        self.tag2label = tag2label
        self.label2tag = {self.tag2label[tag]: tag for tag in self.tag2label}
        self.num_tags = len(tag2label)
        self.vocab = vocab
        self.shuffle = args.shuffle
        self.model_path = log_path['model_path']
        self.summary_path = log_path['summary_path']
        self.logger = logger
        self.result_path = log_path['result_path']
        self.config = config

    def build_graph(self):
        # placeholder
        self.inputs = tf.placeholder(tf.int32, shape=[None, None], name='word_tokens')
        self.labels = tf.placeholder(tf.int32, shape=[None, None], name='labels')
        self.sequence_lengths = tf.placeholder(tf.int32, shape=[None], name='sequence_lengths')
        self.dropout_pl = tf.placeholder(tf.float32, name='dropout')

        # embedding
        with tf.variable_scope('word_embedding'):
            if self.pretrain_embedding is None:
                self.embeddings = tf.Variable(self.init_matrix([len(self.vocab), self.embedding_size]),
                                              dtype=tf.float32,
                                              trainable=self.update_embedding,
                                              name='embedding')
            else:
                self.embeddings = load_pretrain_embedding(self.pretrain_embedding)

            self.word_tokens = tf.nn.embedding_lookup(self.embeddings, self.inputs, name='word_tokens')
            self.word_tokens = tf.nn.dropout(self.word_tokens, self.dropout_pl)

        # Model_layers
        with tf.variable_scope('Bi-LSTM'):
            cell_forward = LSTMCell(self.hidden_size)
            cell_backward = LSTMCell(self.hidden_size)
            (output_fw_seq, output_bw_seq), _ = tf.nn.bidirectional_dynamic_rnn(
                cell_fw=cell_forward,
                cell_bw=cell_backward,
                inputs=self.word_tokens,
                sequence_length=self.sequence_lengths,
                dtype=tf.float32
            )
            output = tf.concat([output_fw_seq, output_bw_seq], axis=-1)
            output = tf.nn.dropout(output, self.dropout_pl)

        with tf.variable_scope('MLP'):
            W = tf.get_variable(name='W',
                                shape=[2 * self.hidden_size, self.num_tags],
                                initializer=xavier_initializer(),
                                dtype=tf.float32)

            b = tf.get_variable(name='b',
                                shape=[self.num_tags],
                                initializer=tf.zeros_initializer(),
                                dtype=tf.float32)

            output_shape = tf.shape(output)
            output = tf.reshape(output, [-1, 2 * self.hidden_size]) # shape = [batch_size * sequence_length, 2 * hidden_size]
            output = tf.nn.xw_plus_b(output, W, b)

            self.logits = tf.reshape(output, [-1, output_shape[1], self.num_tags])
            self.pred = tf.nn.softmax(self.logits)

        if not self.CRF:
            self.label_pred = tf.cast(tf.argmax(self.logits, axis=-1), tf.int32)

        # Loss define
            losses = tf.nn.softmax_cross_entropy_with_logits(logits=self.logits,
                                                                  labels=self.labels)
            mask = tf.sequence_mask(self.sequence_lengths)
            losses = tf.boolean_mask(losses, mask)
            self.loss = tf.reduce_mean(losses)
        else:
            log_likelihood, self.transition_params = crf_log_likelihood(inputs=self.logits,
                                                                        tag_indices=self.labels,
                                                                        sequence_lengths=self.sequence_lengths)
            self.loss = -tf.reduce_mean(log_likelihood)

        tf.summary.scalar('loss', self.loss)

        # optimizer_define
        with tf.variable_scope('optimizer'):
            self.global_step = tf.Variable(tf.constant(0), name='global_step', trainable=False)
            if self.optimizer == 'Adam':
                optim = tf.train.AdamOptimizer(learning_rate=self.lr_pl)
            elif self.optimizer == 'Adadelta':
                optim = tf.train.AdadeltaOptimizer(learning_rate=self.lr_pl)
            elif self.optimizer == 'Adagrad':
                optim = tf.train.AdagradDAOptimizer(learning_rate=self.lr_pl)
            elif self.optimizer == 'RMSProp':
                optim = tf.train.RMSPropOptimizer(learning_rate=self.lr_pl)
            elif self.optimizer == 'Momentum':
                optim = tf.train.MomentumOptimizer(learning_rate=self.lr_pl)
            else:
                optim = tf.train.GradientDescentOptimizer(learning_rate=self.lr_pl)

            grads_and_vars = optim.compute_gradients(self.loss)
            grads_and_vars_with_clip = [[tf.clip_by_value(g, -self.clip_grad, self.clip_grad), v]
                                        for g, v in grads_and_vars]
            self.train_op = optim.apply_gradients(grads_and_vars_with_clip, global_step=self.global_step)

    def add_summary(self, sess):
        self.merged = tf.summary.merge_all()
        self.file_writer = tf.summary.FileWriter(self.summary_path, sess.graph)

    def train(self, sess, train, dev, saver):
        self.add_summary(sess)
        for epoch in range(self.epoch_num):
            self.run_one_epoch(sess, train, dev, self.tag2label, epoch, saver)

    def run_one_epoch(self, sess, train, dev, tag2label, epoch, saver):
        num_batches = (len(train) + self.batch_size - 1) // self.batch_size
        start_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        batches = batch_yield(train, self.batch_size, self.vocab, self.tag2label, shuffle=self.shuffle)
        for step, (seqs, labels) in tqdm(enumerate(batches)):
            # print('Processing: {} batch / {} batches.'.format(step + 1, num_batches))
            step_num = epoch * num_batches + step + 1
            feed_dict, _ = self.get_feed_dict(seqs, labels, self.dropout_keep_prob)
            _, loss, summary, _ = sess.run([self.train_op, self.loss, self.merged, self.global_step],
                                                   feed_dict=feed_dict)
            if step + 1 == 1 or (step + 1) % 300 == 0 or (step + 1) == num_batches:
                self.logger.info(
                    '{}\tepoch: {}\tstep: {}\tloss: {}\tglobal_step: {}'.format(
                        start_time, epoch + 1, step + 1, loss, step_num
                    )
                )
            self.file_writer.add_summary(summary, step_num)

            # Write down the checkpoint
            if step + 1 == num_batches:
                saver.save(sess, self.model_path, global_step=step_num)

        self.logger.info('========== Validation ==========')
        label_list_dev, seq_len_list_dev = self.dev_one_epoch(sess, dev)
        self.evaluate(label_list_dev, dev, epoch)

    def get_feed_dict(self, seqs, labels=None, dropout=None):
        word_tokens, seq_len_list = pad_sequences(seqs, pad_mark=0)
        feed_dict = {self.inputs: word_tokens,
                     self.sequence_lengths: seq_len_list}
        if labels is not None:
            labels_, _ = pad_sequences(labels, pad_mark=0)
            feed_dict[self.labels] = labels_
        if dropout is not None:
            feed_dict[self.dropout_pl] = dropout

        return feed_dict, seq_len_list

    def dev_one_epoch(self, sess, dev):
        label_list, seq_len_list = [], []
        for seqs, labels in batch_yield(dev, self.batch_size, self.vocab, self.tag2label, shuffle=False):
            label_list_, seq_len_list_ = self.predict_one_batch(sess, seqs)
            label_list.extend(label_list_)
            seq_len_list.extend(seq_len_list_)

        return label_list, seq_len_list

    def predict_one_batch(self, sess, seqs):
        feed_dict, seq_len_list = self.get_feed_dict(seqs, dropout=1.0)

        if self.CRF:
            logits, transition_params = sess.run([self.logits, self.transition_params],
                                                 feed_dict=feed_dict)
            label_list = []
            for logit, seq_len in zip(logits, seq_len_list):
                viterbi_seq, _ = viterbi_decode(logit[: seq_len], transition_params)
                label_list.append(viterbi_seq)
            return label_list, seq_len_list
        else:
            label_list = sess.run(self.label_pred, feed_dict=feed_dict)
            return label_list, seq_len_list

    def evaluate(self, label_list, data, epoch=None):
        model_predicts = []
        for preds_, (sent, tag) in zip(label_list, data):
            tag_ = [self.label2tag[pred_] for pred_ in preds_]
            sent_res = []
            if len(preds_) != len(sent):
                print('Warning for sequence lenght miss match between inputs and outputs...')
                print("input_length: {}\t output_length: {}".format(len(sent), len(preds_)))
            for i in range(len(sent)):
                sent_res.append([sent[i], tag[i], tag_[i]])
            model_predicts.append(sent_res)
        epoch_num = str(epoch + 1) if epoch != None else 'None epoch'
        label_path = self.result_path + '_label_' + epoch_num
        metric_path = self.result_path + '_metric_' + epoch_num
        for metric in self.conlleval(model_predicts, label_path, metric_path):
            self.logger.info(metric)

    def conlleval(self, label_predict, label_path, metric_path):
        eval_perl = './output/conlleval_rev.pl'
        if not os.path.exists(eval_perl):
            os.makedirs(eval_perl)
        with open(label_path, 'w', encoding='utf-8') as f:
            for sent_result in label_predict:
                for char, tag, tag_ in sent_result:
                    f.write("{}\t{}\t{}\n".format(char, tag, tag_))
                    f.flush()
                f.write('\n')
        os.system("perl\t{} < {} > {}".format(eval_perl, label_path, metric_path))
        with open(metric_path, 'r', encoding='utf-8') as fr:
            metrics = [line.strip() for line in fr]

        return metrics

    def demo_one(self, sess, sent):
        label_list = []
        for seqs, _ in batch_yield(sent, self.batch_size, self.vocab, self.tag2label, shuffle=False):
            label_list_, _ = self.predict_one_batch(sess, seqs)
            label_list.extend(label_list_)
        tag = [self.label2tag[label] for label in label_list[0]]

        return tag

    def init_matrix(self, shape):
        return tf.random_normal(shape, stddev=0.1)


