import tensorflow as tf
from tensorflow.python.ops.rnn_cell import DropoutWrapper
from tensorflow.python.ops import variable_scope as vs
from tensorflow.python.ops import rnn_cell

class BiLSTM_layer3(object):
    def __init__(self, hidden_size, keep_prob):
        self.hidden_size = hidden_size
        self.keep_prob = keep_prob
        self.rnn_cell_fw = tf.contrib.rnn.LSTMcell(self.hidden_size, forget_bias=1.0)
        # self.rnn_cell_fw = DropoutWrapper(self.rnn_cell_fw, input_keep_prob=self.keep_prob)
        self.rnn_cell_bw = tf.contrib.rnn.LSTMcell(self.hidden_size, forget_bias=1.0)
        # self.rnn_cell_bw = DropoutWrapper(self.rnn_cell_bw, input_keep_prob=self.keep_prob)

    def build_graph(self, inputs, masks):
        with vs.variable_scope("BiLSTM"):
            input_lens = tf.reduce_sum(masks, reduction_indices=1) # shape (batch_size)

            # Note: fw_out and bw_out are the hidden states for every timestep.
            # Each is shape (batch_size, seq_len, hidden_size).
            (fw_out, bw_out), _ = tf.nn.bidirectional_dynamic_rnn(self.rnn_cell_fw, self.rnn_cell_bw, inputs, input_lens, dtype=tf.float32)

            # Concatenate the forward and backward hidden states
            out = tf.concat([fw_out, bw_out], 2)

            # Apply dropout
            # out = tf.nn.dropout(out, self.keep_prob)

            return out

class Attention_layer4(object):
    def __init__(self, keep_prob):
        # self.embed_dim = embed_dim
        self.keep_prob = keep_prob

    def build_graph(self, H, U):
        with vs.variable_scope("Attention"):
            H=tf.transpose(H,perm=[0,2,1])
            U=tf.transpose(U,perm=[0,2,1])
            # Calculate attention distribution
            # H shape (batch_size, embedding_size, context_len)
            # U shape (batch_size, embedding_size, question_len)
            embed_dim=tf.shape(H)[1]
            batch_dim=tf.shape(H)[0]
            w1=tf.get_variable("w1",[1,embed_dim], initializer=tf.contrib.layers.xavier_initializer())
            w2=tf.get_variable("w2",[1,embed_dim], initializer=tf.contrib.layers.xavier_initializer())
            w3=tf.get_variable("w3",[1,embed_dim], initializer=tf.contrib.layers.xavier_initializer())
            p1=tf.matmul(tf.expand_dims(tf.tile(w1,[batch_dim,1]),1),H)   # shape (batch_size, 1, contex_len)
            p2=tf.matmul(tf.expand_dims(tf.tile(w2,[batch_dim,1]),1),U)   # shape (batch_size, 1, question_len)
            p3_1=tf.multiply(tf.expand_dims(H,3),tf.expand_dims(U,2))  # shape (batch_size,embedding_size, contex_len, question_len)
            p3=tf.tensordot(tf.tile(w3,[batch_dim,1]),p3_1,axes=[[1],[1]]) # batch_dim contex_len question_len
            S=tf.add(tf.add(tf.tile(tf.transpose(p1, perm=[0,2,1]),[1,1,tf.shape(U)[1]]),tf.tile(p2,[1,tf.shape(H)[1],1])),p3) # batch_dim contex_len question_len
            a=tf.nn.softmax(S,axis=1) # batch_dim contex_len question_len
            b=tf.softmax(tf.reduce_max(S,axis=2),axis=1)  # batch_dim contex_len
            U_hat=tf.matmul(U,tf.transpose(a,perm=[0,2,1])) #batch_dim embed_dim context_len
            H_hat=tf.matmul(H,b) #batch_dim embed_dim
            output1=tf.concat([H,U_hat],1)  #batch_dim 2*embed_dim context_len
            output2=tf.concat([tf.multiply(H,U_hat),tf.multiply(H,tf.expand_dims(H_hat,2))],1)  #batch_dim 2*embed_dim context_len
            output=tf.concat([output1,output2],1)  #batch_dim 4*embed_dim context_len
            # output = tf.nn.dropout(output, self.keep_prob)
            return output


class BiLSTM_layer5(object):
    def __init__(self, hidden_size, keep_prob):
        self.hidden_size = hidden_size
        self.keep_prob = keep_prob
        self.rnn_cell_fw = tf.contrib.rnn.LSTMcell(self.hidden_size, forget_bias=1.0)
        # self.rnn_cell_fw = DropoutWrapper(self.rnn_cell_fw, input_keep_prob=self.keep_prob)
        self.rnn_cell_bw = tf.contrib.rnn.LSTMcell(self.hidden_size, forget_bias=1.0)
        # self.rnn_cell_bw = DropoutWrapper(self.rnn_cell_bw, input_keep_prob=self.keep_prob)

    def build_graph(self, inputs):
        with vs.variable_scope("BiLSTM"):
            # input_lens = tf.reduce_sum(masks, reduction_indices=1) # shape (batch_size)

            # Note: fw_out and bw_out are the hidden states for every timestep.
            # Each is shape (batch_size, seq_len, hidden_size).
            (fw_out, bw_out), (out_fw_state,out_bw_state) = tf.nn.bidirectional_dynamic_rnn(self.rnn_cell_fw, self.rnn_cell_bw, inputs, dtype=tf.float32)
            input_new=tf.concat([out_fw_state,out_bw_state],2)
            (fw_out, bw_out), _ = tf.nn.bidirectional_dynamic_rnn(fw_out, bw_out, input_new, dtype=tf.float32)
            # Concatenate the forward and backward hidden states
            out = tf.concat([fw_out, bw_out], 2)

            # Apply dropout
            # out = tf.nn.dropout(out, self.keep_prob)

            return out
class OutputLayer_6(object):
    def __init__(self):
        pass

    def build_graph(self, G, M):
        with vs.variable_scope("OutputLayer"):

            # Linear downprojection layer
            w1=tf.get_variable("w1",[1,tf.shape(G)[1]+tf.shape(M)[1]], initializer=tf.contrib.layers.xavier_initializer())
            w2=tf.get_variable("w2",[1,tf.shape(G)[1]+tf.shape(M)[1]], initializer=tf.contrib.layers.xavier_initializer())
            p1=tf.nn.softmax(tf.matmul(tf.tile(w1,[tf.shape(G)[0],1]),tf.concat([G,M],1)))
            p2=tf.nn.softmax(tf.matmul(tf.tile(w1,[tf.shape(G)[0],1]),tf.concat([G,M],1)))
            return p1,p2
