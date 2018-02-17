import { JInputNeuron, JHiddenNeuron, JOutputNeuron, JBiasNeuron } from './neurons.mjs';
import { toJSON } from '../helpers.mjs';
import activationFunctions from './activationFunctions.mjs';

class JLayer {
  constructor(layerDef, NeuronClass) {
    const {
      numNeurons,
      activationFunction = activationFunctions.sigmoid,
      skipBias = false
    } = layerDef;

    this.connectedTo = [];

    let bias;
    if (!skipBias) {
      bias = new JBiasNeuron({ activationFunction });
    }
    const neurons = Array(numNeurons).fill().map(() => new NeuronClass({ activationFunction }));
    Object.assign(this, {
      neurons: bias ? [bias, ...neurons] : neurons,
      connectedTo: []
    });
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
  constructor(layerDef) {
    super(layerDef, JOutputNeuron);
  }

  propagateSignal(target) {
    if (this.neurons.length !== target.length) {
      throw new Error('TARGET size and LAYER size must be the same to propagate. Usually this means there is an undepected bios node');
    }
    return this.neurons.map((neuron, index) =>
      neuron.propagateSignal(target[index]));
  }
}

export class JHiddenLayer extends JLayer {
  constructor(layerDef) {
    super(layerDef, JHiddenNeuron);
  }

  propagateSignal() {
    return this.neurons.map(neuron => neuron.propagateSignal());
  }
}

export class JInputLayer extends JLayer {
  constructor(layerDef) {
    super(layerDef, JInputNeuron);
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
