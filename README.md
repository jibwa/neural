## Javascript Neural Network Starter

### Introduction

The neural network starter is an application designed to help build, train, and visualize Neural Networks.

Currently this project is in development.

### Getting Started

0.) Clone this repo
...> git clone git@github.com:jibwa/neural.git or > git clone https://github.com/jibwa/neural.git
1.) This code is intended to be run with node v9.0.3 [(install Node v9.0.3 with nvm)](https://github.com/creationix/nvm/blob/master/README.md)2.) npm install
...> cd neural
...> npm install
3.) Run the application
...) npm start

### Endpoints

The application runs on port 4000 and at the moment presents multiple endpoints

* http://localhost:4000
... This will return you a JSON representation of the current network state
* http://localhost:4000/visualize
... This will return a (very bad at the moment) HTML representation of the network
* http://localhost:4000/visualize/1 (or any number)
... This will run a trainer though a single step and return the visualization
* http://localhost:4000/monitor
... This will return JSON results of the network training
* http://localhost:4000/monitor/1 (or any number, watch out using a number > 100, data gets big quick)
... This will train the network and record node and connections states at each activation and back prop

