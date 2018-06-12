## Javascript Neural Network Starter

### Introduction

I built neural as a companion application to implement various techniques for machine learning.

Much of the content I found online to support machine learning techniques were relying on matrix
operations and pre-built libraries. In order to get a better picture of what was happening in the
math I implemented those operations functionally.

- Neural code is written from scratch
- Neural code is not implementing any libraries or even matrix operations
- Neural is only intended for use in learning

Currently this project is in development.

### Getting Started

1. Clone this repo
- git clone git@github.com:jibwa/neural.git or > git clone https://github.com/jibwa/neural.git
2. This code is intended to be run with node v9.0.3 or higher [(install Node v9.0.3 with nvm)](https://github.com/creationix/nvm/blob/master/README.md)
3. npm install
- cd neural
- npm install
4. Run the application
- npm start

### Endpoints

The application runs on port 4000 and at the moment presents a couple sample endpoints.

1. http://localhost:4000/xor/visualize
- This will return you super simple results and information about the network
2. http://localhost:4000/xor/visualize/1000
- Same as #1 except it will run 1000 iterations through the training before outputting results
3. http://localhost:4000/xor/reset
- reset the network back to random weights

### Notes

Currently only the XOR example is set up to run out of the box. Instead of just training XOR it currently trains a training set similar to XOR but with more options.

The diabetes study is not implemented in Neural yet.

Using the trainer pattern shown in the example neural allows for the use of

1. L1 Regularization
2. L2 Regularization
3. Dropout
4. Dataset batch and shuffle
