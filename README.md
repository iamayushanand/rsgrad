# RSGRAD

An autograd framework for rust from scratch. This project is for educational purposes and also my first in rust programming language. The framework implements reverse mode autodifferentiation and is Tensor-based(it uses Vector Jacobean Products to compute gradients).

## Usage

The file `rsgrad-nn/examples/exp_addition.rs` contains a sample MLP regression network to compute sum of exponents. The output of a sample run of the network gives
```
n:0 expected: 3.6181414 got: 3.6631637
n:1000 expected: 4.0001206 got: 4.455985
n:2000 expected: 8.433083 got: 9.160059
Val running loss: 0.3371813
```

## Contact

- Ayush Anand(anand.5@iitj.ac.in)
