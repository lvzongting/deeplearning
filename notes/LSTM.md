##1
LSTM(x,h,c) in [23] [Sequence to sequence learning with neural networks NIPS 2014]        
x: input    
h: output    
c: cell     
LSTM has two layer: one is encoder dealing with input, the other is decoder dealing with output  
```python
#encode
_, enc_state  = tf.nn.rnn(cell, encoder_inputs, dtype=dtype)    

#decode
output, state = tf.nn.seq2seq.rnn_decoder(decoder_inputs, enc_state, cell)
```

##2
+ Core network in [Recurrent Models of Visual Attention NIPS 2014]    
h_t = f_h(h_{t-1},g_t;\theta_h)     
h_t = f_h(h_{t-1})=Rect(Linear(h_{t-1})+Linear(g_t))    
+ Attention Mechanisms in [Order Matters: sequence to sequence for sets https://arxiv.org/abs/1511.06391 ]      
q_t = LSTM(q^\*_{t-1})    
e_{i,t} = f(m_i,q_t)     
a_{i,t} = exp(e_{i,t})/sum_j{exp(e_{j,t})}     
r_t = sum_i{a_{i,t}m_i}    
q^\*_t = [q_t r_t] &emsp;&emsp; #concatenate(q_t,r_t)      

The concatenation of input & state is taken as cell_input of LSTM     
```python
#[Line 503 python/ops/rnn_cell.py(class LSTMCell(RNNCell))] 
cell_inputs = array_ops.concat(1, [inputs, m_prev]) 
```
