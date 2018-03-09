import tensorflow as tf
from tensorflow.python.ops.rnn_cell import DropoutWrapper
from tensorflow.python.ops import variable_scope as vs
from tensorflow.python.ops import rnn_cell

class char_CNN_layer1(object):
    def __init__(self, hidden_size):
        self.hidden_size = hidden_size

    def build_graph(self, inputs, scope_val):
        with vs.variable_scope(scope_val):
            wcnn1=tf.get_variable("wcnn1",[5, 97, self.hidden_size], initializer=tf.contrib.layers.xavier_initializer())
            bcnn1=tf.get_variable("bcnn1",[self.hidden_size], initializer=tf.contrib.layers.xavier_initializer())
            out = tf.nn.conv1d(tf.one_hot(inputs, 97), wcnn1, stride=1, padding='SAME')
            out = tf.nn.relu(out + bcnn1) - 0.01 * tf.nn.relu(-out - bcnn1)
            return out

class BiLSTM_layer3(object):
    def __init__(self, hidden_size, keep_prob):
        self.hidden_size = hidden_size
        self.keep_prob = keep_prob
        self.rnn_cell_fw = tf.contrib.rnn.BasicLSTMCell(self.hidden_size, forget_bias=1.0)
        self.rnn_cell_fw = DropoutWrapper(self.rnn_cell_fw, input_keep_prob=self.keep_prob)
        self.rnn_cell_bw = tf.contrib.rnn.BasicLSTMCell(self.hidden_size, forget_bias=1.0)
        self.rnn_cell_bw = DropoutWrapper(self.rnn_cell_bw, input_keep_prob=self.keep_prob)

    def build_graph(self, inputs, masks):
        with vs.variable_scope("BiLSTM3"):
            input_lens = tf.reduce_sum(masks, reduction_indices=1) # shape (batch_size)

            # Note: fw_out and bw_out are the hidden states for every timestep.
            # Each is shape (batch_size, seq_len, hidden_size).
            (fw_out, bw_out), _ = tf.nn.bidirectional_dynamic_rnn(self.rnn_cell_fw, self.rnn_cell_bw, inputs, input_lens, dtype=tf.float32)

            # Concatenate the forward and backward hidden states
            out = tf.concat([fw_out, bw_out], 2)

            # Apply dropout
            out = tf.nn.dropout(out, self.keep_prob)

            return out

class Attention_layer4(object):
    def __init__(self, keep_prob):
        # self.embed_dim = embed_dim
        self.keep_prob = keep_prob

    def build_graph(self, H, U, U_mask):
        with vs.variable_scope("Attention"):
            H=tf.transpose(H,perm=[0,2,1])
            U=tf.transpose(U,perm=[0,2,1])
            # Calculate attention distribution
            # H shape (batch_size, embedding_size, context_len)
            # U shape (batch_size, embedding_size, question_len)
            H_shap=H.get_shape().as_list()
            U_shap=U.get_shape().as_list()
            embed_dim=H_shap[1]
            batch_dim=H_shap[0]
            w1=tf.get_variable("w1",[embed_dim], initializer=tf.contrib.layers.xavier_initializer())
            w2=tf.get_variable("w2",[embed_dim], initializer=tf.contrib.layers.xavier_initializer())
            w3=tf.get_variable("w3",[embed_dim], initializer=tf.contrib.layers.xavier_initializer())
            p1=tf.tensordot(w1, H, axes=[[0], [1]])   # shape (batch_size, contex_len)
            p2=tf.tensordot(w2, U, axes=[[0], [1]])   # shape (batch_size, question_len)
            p3_1=tf.multiply(tf.expand_dims(H,3),tf.expand_dims(U,2))  # shape (batch_size,embedding_size, contex_len, question_len)
            p3=tf.tensordot(w3, p3_1,axes=[[0],[1]]) # batch_dim contex_len question_len
            S_hat = tf.add(tf.add(tf.expand_dims(p1, 2), tf.expand_dims(p2, 1)), p3) # batch_dim contex_len question_len
            _, S = masked_softmax(S_hat, tf.expand_dims(U_mask, 1), 2)
            a=tf.nn.softmax(S,dim=1) # batch_dim contex_len question_len
            b=tf.nn.softmax(tf.reduce_max(S,reduction_indices=2),dim=1)  # batch_dim contex_len
            U_hat=tf.matmul(U,tf.transpose(a,perm=[0,2,1])) #batch_dim embed_dim context_len
            H_hat=tf.matmul(H, tf.expand_dims(b, 2)) #batch_dim embed_dim 
            output1=tf.concat([H,U_hat],1)  #batch_dim 2*embed_dim context_len
            output2=tf.concat([tf.multiply(H,U_hat),tf.multiply(H,H_hat)],1)  #batch_dim 2*embed_dim context_len
            output=tf.concat([output1,output2],1)  #batch_dim 4*embed_dim context_len
            output = tf.nn.dropout(output, self.keep_prob)
            return output


class BiLSTM_layer5(object):
    def __init__(self, hidden_size, keep_prob):
        self.hidden_size = hidden_size
        self.keep_prob = keep_prob
        self.rnn_cell_fw = tf.contrib.rnn.BasicLSTMCell(self.hidden_size, forget_bias=1.0)
        self.rnn_cell_fw = DropoutWrapper(self.rnn_cell_fw, input_keep_prob=self.keep_prob)
        self.rnn_cell_bw = tf.contrib.rnn.BasicLSTMCell(self.hidden_size, forget_bias=1.0)
        self.rnn_cell_bw = DropoutWrapper(self.rnn_cell_bw, input_keep_prob=self.keep_prob)

    def build_graph(self, inputs, scope_val):
        with vs.variable_scope(scope_val):
            # input_lens = tf.reduce_sum(masks, reduction_indices=1) # shape (batch_size)

            # Note: fw_out and bw_out are the hidden states for every timestep.
            # Each is shape (batch_size, seq_len, hidden_size).
            (fw_out, bw_out), (out_fw_state,out_bw_state) = tf.nn.bidirectional_dynamic_rnn(self.rnn_cell_fw, self.rnn_cell_bw, inputs, dtype=tf.float32)
            #input_new=tf.concat([out_fw_state,out_bw_state],2)
            #(fw_out2, bw_out2), _ = tf.nn.bidirectional_dynamic_rnn(fw_out, bw_out, input_new, dtype=tf.float32)
            # Concatenate the forward and backward hidden states
            out = tf.concat([fw_out, bw_out], 2)

            # Apply dropout
            out = tf.nn.dropout(out, self.keep_prob)

            return out
class OutputLayer_6(object):
    def __init__(self, hidden_size, keep_prob):
        self.hidden_size = hidden_size
        self.keep_prob = keep_prob
        self.rnn_cell_fw = tf.contrib.rnn.BasicLSTMCell(self.hidden_size, forget_bias=1.0)
        self.rnn_cell_fw = DropoutWrapper(self.rnn_cell_fw, input_keep_prob=self.keep_prob)
        self.rnn_cell_bw = tf.contrib.rnn.BasicLSTMCell(self.hidden_size, forget_bias=1.0)
        self.rnn_cell_bw = DropoutWrapper(self.rnn_cell_bw, input_keep_prob=self.keep_prob)

    def build_graph(self, G, M):
        with vs.variable_scope("OutputLayer"):

            # Linear downprojection layer
            (fw_out, bw_out), (out_fw_state,out_bw_state) = tf.nn.bidirectional_dynamic_rnn(self.rnn_cell_fw, self.rnn_cell_bw, tf.transpose(M, perm=[0, 2, 1]), dtype=tf.float32)
            M2 = tf.transpose(tf.concat([fw_out, bw_out], 2), perm = [0, 2, 1])

            G_shap=G.get_shape().as_list()
            M_shap=M.get_shape().as_list()
            M2_shap=M2.get_shape().as_list()
            w1=tf.get_variable("w1",[G_shap[1]+M_shap[1]], initializer=tf.contrib.layers.xavier_initializer())
            w2=tf.get_variable("w2",[G_shap[1]+M2_shap[1]], initializer=tf.contrib.layers.xavier_initializer())

            p1=tf.nn.softmax(tf.tensordot(w1, tf.concat([G,M],1), axes=[[0], [1]]))
            p2=tf.nn.softmax(tf.tensordot(w2, tf.concat([G,M2],1), axes=[[0], [1]]))
            return p1,p2

def masked_softmax(logits, mask, dim):
    exp_mask = (1 - tf.cast(mask, 'float')) * (-1e30) # -large where there's padding, 0 elsewhere
    masked_logits = tf.add(logits, exp_mask) # where there's padding, set logits to -large
    prob_dist = tf.nn.softmax(masked_logits, dim)
    return masked_logits, prob_dist