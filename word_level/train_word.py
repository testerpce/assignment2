
from lstm_model import LSTModel
from collections import Counter
import tensorflow as tf
import numpy as np
import pickle
import nltk
import re
import os

from traitlets import Bunch


def save_pickle(obj, file_name):
    with open(file_name, 'wb') as f:
        pickle.dump(obj, f)


def load_pkl(file_name):
    with open(file_name, 'rb') as f:
        return pickle.load(f)


def actual_text():
    text = nltk.corpus.gutenberg.raw('austen-sense.txt')
    text = text.lower()
    text = re.sub("([@]+)|([-]{2,})|(\\n)|([\s]{2,})|([\"*:\[\]()]+)", ' ', text)
    sentens = np.array(nltk.tokenize.sent_tokenize(text))
    tot = len(sentens)
    train_data = sentens[0:int(0.8*tot)]
    dev = sentens[int(0.8*tot):int((0.8+0.2)*tot)]
    test = sentens[int(0.8*tot):]
    train_data = nltk.tokenize.word_tokenize(' '.join(train_data))
    dev = nltk.tokenize.word_tokenize(' '.join(dev))
    test = nltk.tokenize.word_tokenize(' '.join(test))
    save_pickle(train_data, 'train_data.pkl')
    save_pickle(dev, 'dev.pkl')
    save_pickle(test, 'test.pkl')


def data_chars():
    train_data = load_pkl('train_data.pkl')
    word_counter = Counter(train_data)
    words = ['<unk>' if word_counter[word] < 2 else word for word in train_data]
    all_words = sorted(set(words))
    vocab_size = len(all_words)
    print('Vocab Size', vocab_size)
    word_value = dict((x, i) for i, x in enumerate(all_words))
    word_train = np.array([word_value[x] for x in words])
    print('Total number of words =', len(word_train))
    save_pickle(all_words, 'all_words.pkl')
    save_pickle(word_train, 'word_train.pkl')
    save_pickle(vocab_size, 'vocab_size.pkl')


def make_batch(bsize=30, slen=30):
    word_train = load_pkl('word_train.pkl')
    num_batches = int((len(word_train) - 1) / (bsize * slen))
    find = num_batches * bsize * slen
    x = word_train[:find]
    y = word_train[1:(find + 1)]
    batch_x = np.split(np.reshape(x, [bsize, -1]), num_batches, 1)
    batch_y = np.split(np.reshape(y, [bsize, -1]), num_batches, 1)
    save_pickle(batch_x, 'batch_x.pkl')
    save_pickle(batch_y, 'batch_y.pkl')


def make_args(bsize=30, slen=30):
    vocab_size = load_pkl('vocab_size.pkl')
    args = {
        'mtype': 'lstm',
        'num_layers': 2,
        'rnn_size': 128,
        'lr': 0.002,
        'decay': 0.97,
        'batch_size': bsize,
        'seq_length': slen,
        'vocab_size': vocab_size
    }
    save_pickle(Bunch(args), 'args.pkl')


def train(epochs=1):
    args = load_pkl('args.pkl')
    model = LSTModel(args)
    init = tf.global_variables_initializer()
    tf_saver = tf.train.Saver(tf.global_variables())
    batch_x = load_pkl('batch_x.pkl')
    batch_y = load_pkl('batch_y.pkl')
    num_batches = len(batch_x)
    with tf.Session() as sess:
        sess.run(init)
        for e in range(epochs):
            sess.run(tf.assign(model.lr, args.lr * (args.decay ** e)))
            state = sess.run(model.initial_state)
            for b in range(num_batches):
                x = batch_x[b]
                y = batch_y[b]
                feed = {
                    model.input_data: x,
                    model.targets: y
                }
                for i, (c, h) in enumerate(model.initial_state):
                    feed[c] = state[i].c
                    feed[h] = state[i].h
                _, train_loss, state, predicted_output = sess.run([model.optimizer, model.cost, model.final_state, model.predicted_output], feed)
                accuracy = np.sum(np.equal(y, predicted_output)) / float(y.size)
                print("{}/{} - epoch {} , loss = {:.3f}, accuracy = {}".format(e * num_batches + b, epochs * num_batches, e, train_loss, accuracy))
                if ((e * num_batches + b) % 1000 == 0) or (e == epochs - 1 and b == num_batches - 1):
                    checkpoint_path = os.path.join('./word_model', 'model.ckpt')
                    tf_saver.save(sess, checkpoint_path, global_step=e * num_batches + b)
                    print("Model saved to {}".format(checkpoint_path))


def generate_sentence(start="the ", n_predictions=100):
    args = load_pkl('args.pkl')
    model = LSTModel(args, training=False)
    all_words = load_pkl('all_words.pkl')
    word_value = dict((x, i) for i, x in enumerate(all_words))
    value_word = dict((i, x) for i, x in enumerate(all_words))
    sentence = start
    init = tf.global_variables_initializer()
    tf_saver = tf.train.Saver(tf.global_variables())
    tf.reset_default_graph()
    with tf.Session() as sess:
        sess.run(init)
        checkpoint = tf.train.get_checkpoint_state('./word_model')
        if checkpoint and checkpoint.model_checkpoint_path:
            tf_saver.restore(sess, checkpoint.model_checkpoint_path)
        state = sess.run(model.cell.zero_state(1, tf.float32))
        for c in start[:-1]:
            x = np.reshape(word_value[c], [1, 1])
            feed = {model.input_data: x, model.initial_state: state}
            state = sess.run(model.final_state, feed)

        c = start[-1]
        for i in range(n_predictions):
            x = np.reshape(word_value[c], [1, 1])
            feed = {model.input_data: x, model.initial_state: state}
            prob, state = sess.run([model.probs, model.final_state], feed)
            val = int(np.searchsorted(np.cumsum(prob[0]), np.random.rand(1)))
            c = value_word[val]
            sentence += [c]

        return sentence


def perplexity():
    tf.reset_default_graph()
    args = load_pkl('args.pkl')
    args.batch_size = 1
    model = LSTModel(args, training=True)
    # loading dev set
    text = load_pkl('dev.pkl')
    all_words = load_pkl('all_words.pkl')
    word_value = dict((x, i) for i, x in enumerate(all_words))
    init = tf.global_variables_initializer()
    text = list(text)
    n_batches = int(len(text) / args.seq_length)
    for i in range(len(text)):
        if text[i] not in all_words:
            text[i] = '<unk>'
    test_value_set = np.array([word_value[c] for c in text])
    limit = n_batches * args.seq_length
    data_x = np.reshape(test_value_set[:limit], [n_batches, args.seq_length])
    data_y = np.reshape(test_value_set[1:limit + 1], [n_batches, args.seq_length])

    logprob = 0
    with tf.Session() as sess:
        sess.run(init)
        tf_saver = tf.train.Saver(tf.global_variables())
        checkpoint = tf.train.get_checkpoint_state('./word_model')
        if checkpoint and checkpoint.model_checkpoint_path:
            tf_saver.restore(sess, checkpoint.model_checkpoint_path)
        state = sess.run(model.cell.zero_state(1, tf.float32))
        for i in range(n_batches):
            seq = np.reshape(data_x[i, :], [args.batch_size, args.seq_length])
            feed = {model.input_data: seq, model.initial_state: state}
            prob, state = sess.run([model.probs, model.final_state], feed)
            prob = np.log(prob[np.arange(len(prob)), data_y[i, :]])
            logprob += np.sum(prob)

            # print(prob[0][char_to_value[c1]],c1)
        perplex = np.exp(-logprob / (args.seq_length * n_batches))
        return perplex


actual_text()
data_chars()
make_batch()
make_args()
train(epochs=5)
print(perplexity())
