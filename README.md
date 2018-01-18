## Javascript Neural Network Starter

### Introduction

The neural network starter is an application designed to help build, train, and visualize Neural Networks.

Currently this project is in development.

### Getting Started

1. Clone this repo
- git clone git@github.com:jibwa/neural.git or > git clone https://github.com/jibwa/neural.git
2. This code is intended to be run with node v9.0.3 [(install Node v9.0.3 with nvm)](https://github.com/creationix/nvm/blob/master/README.md)
3. npm install
- cd neural
- npm install
4. Run the application
- npm start

### Endpoints

The application runs on port 4000 and at the moment presents multiple endpoints

1. http://localhost:4000
- This will return you a JSON representation of the current network state
2. http://localhost:4000/visualize
- This will return a (very bad at the moment) HTML representation of the network
3. http://localhost:4000/visualize/1 (or any number)
- This will run a trainer though a single step and return the visualization
4. http://localhost:4000/monitor
- This will return JSON results of the network training
5. http://localhost:4000/monitor/1 (or any number, watch out using a number > 100, data gets big quick)
- This will train the network and record node and connections states at each activation and back prop

### Notes

At the moment the application runs only in memory. Each iteration endpoint you hit will progress the in memory network. I'll make an endpoint to reset the network soon. For now just <ctrl> c to stop and then start the application again to reset.
