import { JInputNeuron, JHiddenNeuron, JOutputNeuron } from './neurons.mjs';
import { toJSON } from '../helpers.mjs';

class JLayer {
  constructor() {
    // TODO JSON DATA
    this.connectedTo = [];
  }

  activate() {
    return this.neurons.map(neuron => neuron.activate());
  }

  // set up the neuron relationships
  project(to) {
    this.connectedTo.push(to);

    this.neurons.forEach(fromNeuron =>
      to.neurons.forEach(toNeuron =>
        fromNeuron.project(toNeuron)));
  }
  toJSON() {
    return { neurons: this.neurons.map(toJSON) };
  }
}

export class JOutputLayer extends JLayer {
  constructor(size = 0, jsonData) {
    super(jsonData);
    this.neurons = Array(size).fill().map(() => new JOutputNeuron());
  }

  propagate(target, learningRate) {
    if (this.neurons.length !== target.length) {
      throw new Error('TARGET size and LAYER size must be the same to propagate!');
    }
    this.neurons.forEach((neuron, index) =>
      neuron.propagate(target[index], learningRate));
  }
}

export class JHiddenLayer extends JLayer {
  constructor(size = 0, jsonData) {
    super(jsonData);
    this.neurons = Array(size).fill().map(() => new JHiddenNeuron());
  }

  propagate(learningRate) {
    this.neurons.forEach(neuron => neuron.propagate(learningRate));
  }
}

export class JInputLayer extends JLayer {
  constructor(size = 0, jsonData) {
    super(jsonData);
    if (jsonData) {
      this.neurons = jsonData.neurons.map(data => new JInputNeuron(data));
    } else {
      this.neurons = Array(size).fill().map(() => new JInputNeuron());
    }
  }

  activate(inputs) {
    if (inputs.length !== this.neurons.length) {
      throw new Error(`Input size and layer size must be equal to activate  I:${inputs.length}  N:${this.neurons.length}`);
    }
    return this.neurons.map((neuron, index) =>
      neuron.activate(inputs[index]));
  }
}
