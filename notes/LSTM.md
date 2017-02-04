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
