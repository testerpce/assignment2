from lstm_model import LSTModel
import tensorflow as tf
import numpy as np
import pickle


def load_pkl(file_name):
    with open(file_name, 'rb') as f:
        return pickle.load(f)


start = 'the '
n_predictions = 200
args = load_pkl('args.pkl')
model = LSTModel(args, training=False)
all_chars = load_pkl('all_chars.pkl')
char_value = dict((x, i) for i, x in enumerate(all_chars))
value_char = dict((i, x) for i, x in enumerate(all_chars))
sentence = start
init = tf.global_variables_initializer()
tf_saver = tf.train.Saver(tf.global_variables())
with tf.Session() as sess:
    sess.run(init)
    checkpoint = tf.train.get_checkpoint_state('./char_model')
    if checkpoint and checkpoint.model_checkpoint_path:
        tf_saver.restore(sess, checkpoint.model_checkpoint_path)
    state = sess.run(model.cell.zero_state(1, tf.float32))
    for c in start[:-1]:
        x = np.reshape(char_value[c], [1, 1])
        feed = {model.input_data: x, model.initial_state: state}
        state = sess.run(model.final_state, feed)

    c = start[-1]
    for i in range(n_predictions):
        x = np.reshape(char_value[c], [1, 1])
        feed = {model.input_data: x, model.initial_state: state}
        prob, state = sess.run([model.probs, model.final_state], feed)
        if c == ' ':
            val = int(np.searchsorted(np.cumsum(prob[0]), np.random.rand(1)))
        else:
            val = int(np.argmax(prob[0]))
        c = value_char[val]
        if c=='.' or c=='?' or c=='!':
        	break
        sentence += c

print(sentence)
