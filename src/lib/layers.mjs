import { JInputNeuron, JHiddenNeuron, JOutputNeuron, JBiasNeuron } from './neurons.mjs';
import { toJSON } from '../helpers.mjs';

class JLayer {
  constructor(size, NeuronClass, createBias) {
    // TODO JSON DATA
    this.connectedTo = [];
    let bias;
    if (createBias) {
      bias = new JBiasNeuron();
    }
    const neurons = Array(size).fill().map(() => new NeuronClass());
    this.neurons = createBias ? [bias, ...neurons] : neurons;
  }

  activate() {
    return this.neurons.map(neuron => neuron.activate());
  }

  // set up the neuron relationships
  project(to) {
    this.connectedTo.push(to);

    this.neurons.forEach((fromNeuron) => {
      to.neurons.forEach((toNeuron) => {
        if (!(toNeuron instanceof JBiasNeuron)) {
          fromNeuron.project(toNeuron);
        }
      });
    });
  }
  weighSumConnections() {
    return this.neurons.map(neuron =>
      neuron.weighSumConnections());
  }
  toJSON() {
    return { neurons: this.neurons.map(toJSON) };
  }
}

export class JOutputLayer extends JLayer {
  constructor(size = 0) {
    super(size, JOutputNeuron, false);
  }

  propagateSignal(target) {
    if (this.neurons.length !== target.length) {
      throw new Error('TARGET size and LAYER size must be the same to propagate!');
    }
    return this.neurons.map((neuron, index) =>
      neuron.propagateSignal(target[index]));
  }
}

export class JHiddenLayer extends JLayer {
  constructor(size = 0) {
    super(size, JHiddenNeuron, true);
  }

  propagateSignal() {
    return this.neurons.map(neuron => neuron.propagateSignal());
  }
}

export class JInputLayer extends JLayer {
  constructor(size = 0) {
    super(size, JInputNeuron, true);
  }

  activate(inputs) {
    const { neurons } = this;
    const inputNeurons = neurons.filter(neuron => neuron instanceof JInputNeuron);
    if (inputs.length !== inputNeurons.length) {
      throw new Error(`Input size and layer size must be equal to activate  I:${inputs.length}  N:${this.neurons.length}`);
    }
    // activate bias
    inputNeurons.forEach((neuron, index) =>
      neuron.activate(inputs[index]));
  }
}
