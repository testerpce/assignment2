import nltk
from lstm_model import LSTModel
import tensorflow as tf
import numpy as np
import pickle


def load_pkl(file_name):
    with open(file_name, 'rb') as f:
        return pickle.load(f)


start = 'the '
n_predictions = 200
start = nltk.tokenize.word_tokenize(start)
args = load_pkl('args.pkl')
model = LSTModel(args, training=False)
all_words = load_pkl('all_words.pkl')
word_value = dict((x, i) for i, x in enumerate(all_words))
value_word = dict((i, x) for i, x in enumerate(all_words))
sentence = start
init = tf.global_variables_initializer()
tf_saver = tf.train.Saver(tf.global_variables())
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

print(' '.join(sentence))
