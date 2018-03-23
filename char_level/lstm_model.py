import tensorflow as tf
from tensorflow.contrib import rnn
from tensorflow.contrib import legacy_seq2seq
from tensorflow.python.ops.nn_ops import sparse_softmax_cross_entropy_with_logits


class LSTModel():
    def __init__(self, args, training=True):
        self.args = args
        # When we don't train then we will take in one character at a time and try to predict
        if not training:
            args.batch_size = 1
            args.seq_length = 1
        # Assign the basic type of RNN unit
        if args.mtype == 'rnn':
            cell_fn = rnn.BasicRNNCell
        elif args.mtype == 'gru':
            cell_fn = rnn.GRUCell
        elif args.mtype == 'lstm':
            cell_fn = rnn.BasicLSTMCell
        elif args.mtype == 'nas':
            cell_fn = rnn.NASCell
        else:
            raise Exception("model type not supported: {}".format(args.model))

        cells = []
        for _ in range(args.num_layers):
            cell = cell_fn(args.rnn_size)
            cells.append(cell)

        self.cell = cell = rnn.MultiRNNCell(cells, state_is_tuple=True)

        self.input_data = tf.placeholder(
            tf.int32, [args.batch_size, args.seq_length])
        self.targets = tf.placeholder(
            tf.int32, [args.batch_size, args.seq_length])
        self.initial_state = cell.zero_state(args.batch_size, tf.float32)

        softmax_w = tf.get_variable("softmax_w", [args.rnn_size, args.vocab_size])
        softmax_b = tf.get_variable("softmax_b", [args.vocab_size])

        embedding = tf.get_variable("embedding", [args.vocab_size, args.rnn_size])
        inputs = tf.nn.embedding_lookup(embedding, self.input_data)
        inputs = tf.split(inputs, args.seq_length, 1)
        inputs = [tf.squeeze(input_, [1]) for input_ in inputs]

        outputs, last_state = legacy_seq2seq.rnn_decoder(inputs, self.initial_state, cell)
        output = tf.reshape(tf.concat(outputs, 1), [-1, args.rnn_size])

        self.logits = tf.matmul(output, softmax_w) + softmax_b
        self.probs = tf.nn.softmax(self.logits)
        self.predicted_output = tf.reshape(tf.argmax(self.probs, 1), [args.batch_size, args.seq_length])

        loss = sparse_softmax_cross_entropy_with_logits(logits=[self.logits], labels=[tf.reshape(self.targets, [-1])])

        self.cost = tf.reduce_sum(loss) / args.batch_size / args.seq_length
        self.final_state = last_state
        self.lr = tf.Variable(0.0, trainable=False)
        self.optimizer = tf.train.AdamOptimizer(self.lr).minimize(self.cost)
