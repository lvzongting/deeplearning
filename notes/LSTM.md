##1
LSTM(x,h,c) in [23] [Sequence to sequence learning with neural networks NIPS 2014]        
x: input    
h: output    
c: cell     
LSTM has two layer: one is encoder dealing with input, the other is decoder dealing with output  
```python
#[Line 175~177 in python/ops/seq2seq.py(def basic_rnn_seq2seq(...))]
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
#[Line 503 in python/ops/rnn_cell.py(class LSTMCell(RNNCell))] 
cell_inputs = array_ops.concat(1, [inputs, m_prev]) 
```
the state is h_{t-1} & q^\*_{t-1}, then feed  g_t & r_t as inputs.      
All above are functions from state to seq, so use seq2seq.rnn_decoder(), additional loop_function modify the input within iteration.       
```python
def get_next_input(prev, i):
  next = foo(prev)
  return next
  
outputs, _ = seq2seq.rnn_decoder(inputs, init_state, lstm_cell, loop_function=get_next_input)
```
inputs is h_0 & g^\*_0

###addition
The simplest form of RNN network generated is:      
```python
#[Line 76~83 in python/ops/rnn.py(def rnn(cell,...))]
state = cell.zero_state(...)
outputs = []
for input_ in inputs:
    output, state = cell(input_, state)
    outputs.append(output)
return (outputs, state)
```

##3
full context embeddings in [Matching Networks for One Shot Learning NIPS 2016]     
\hat{h_k}, c_k    = LSTM(f^'(\hat{x}),[h_{k-1},r_{k-1}],c_{k-1})      
h_k               = \hat{h_k} + f^'(\hat{x})     
r_{k-1}           = sum_i{a(h_{k-1},g(x_i))g(x_i)}    
a(h_{k-1},g(x_i)) = exp(h^T_{k-1}g(x_i))/sum_j{exp(h^T_{k-1}g(x_j))}     

h_k: state of LSTM, r_k: input of LSTM
Above functions need to change both input & state, so need modify the rnn_decoder as rnn_decoder2
```python
#[Line 136~150 in python/ops/seq2seq.py(def rnn_decoder(...))]
def rnn_decoder2(decoder_inputs, initial_state, cell, loop_function=None,state_function=None,scope=None):
    with variable_scope.variable_scope(scope or "rnn_decoder"):
    state = initial_state
    outputs = []
    prev_input = None
    prev_state = None
    for i, inp in enumerate(decoder_inputs):
        if loop_function  is not None and prev_input is not None:
            with variable_scope.variable_scope("loop_function", reuse=True):
            inp   =  loop_function(prev_input, i)
        if state_function is not None and prev_state is not None:
            with variable_scope.variable_scope("state_function", reuse=True):
            state = state_function(prev_state, i)        
        if i > 0:
            variable_scope.get_variable_scope().reuse_variables()
        output, state = cell(inp, state)
        outputs.append(output)
        if loop_function is not None:
            prev_input = output
            prev_state = state
    return outputs, state
```

##4
inline LSTM iteration as following:
```python
#in: lstm_input,out: outputs, state
lstm  = tf.nn.rnn_cell.BasicLSTMCell(1024,forget_bias=1.0,state_is_tuple=True)
state = lstm.zero_state(batch_size=500,dtype=tf.float32)

outputs = []
prev_input = None
prev_state = None
for i,inp in enumerate(lstm_inputs):
    if i > 0:
        with variable_scope.variable_scope("input_function", reuse=True):
        inp   = input_function(prev_input, i)
        with variable_scope.variable_scope("state_function", reuse=True):
        state = state_function(prev_state, i)
        variable_scope.get_variable_scope().reuse_variables()
    output, state = lstm(inp, state)
    outputs.append(output)
    prev_input = output
    prev_state = state   
```
