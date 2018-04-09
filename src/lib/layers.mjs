import { JInputNeuron, JHiddenNeuron, JOutputNeuron, JBiasNeuron } from './neurons.mjs';
import { toJSON } from '../helpers.mjs';
import { sigmoid } from './activationFunctions.mjs';

class JLayer {
  constructor(layerDef, NeuronClass, BiasClass = JBiasNeuron) {
    const {
      numNeurons,
      activationFunction = sigmoid,
      weightMax = 1.0,
      skipBias = false
    } = layerDef;

    this.connectedTo = [];

    let bias;
    if (!skipBias) {
      bias = new BiasClass({ activationFunction });
    }
    const neurons = Array(numNeurons).fill().map(() => new NeuronClass({ activationFunction }));
    Object.assign(this, {
      neurons: bias ? [bias, ...neurons] : neurons,
      connectedTo: [],
      layerDef
    });
  }

  activate() {
    return this.neurons.map(neuron => neuron.activate());
  }

  // set up the neuron relationships
  project(to) {
    const {layerDef: { weightMax } } = this;
    this.connectedTo.push(to);

    this.neurons.forEach((fromNeuron) => {
      to.neurons.forEach((toNeuron) => {
        if (!(toNeuron instanceof JBiasNeuron)) {
          fromNeuron.project(toNeuron, weightMax);
        }
      });
    });
  }
  calculateSignal() {
    return this.neurons.map(neuron => neuron.calculateSignal());
  }
  updateConnectionSums() {
    return this.neurons.map(neuron => neuron.updateConnectionSums())
  }
  getConnectionSums(result) {
    this.neurons.forEach(neuron => neuron.getConnectionSums(result));
    return result;
  }
  toJSON() {
    return {
      layerType: this.constructor.name,
      neurons: this.neurons.map(toJSON)
    };
  }
  getWeightArrays() {
    return this.neurons.map(neuron => neuron.getWeightArray());
  }
  restoreWeights(weights) {
    return this.neurons.forEach((neuron, i) => neuron.restoreWeights(weights[i]));
  }
}

export class JOutputLayer extends JLayer {
  constructor(layerDef) {
    super(layerDef, JOutputNeuron);
  }

  calculateLoss(target) {
    if (this.neurons.length !== target.length) {
      throw new Error(`TARGET size and LAYER size must be the same to propagate. Usually this means there is an unexpected bios node. Target size: ${target.length}, Layer size: ${this.neurons.length}`);
    }
    const loss = this.neurons.reduce((acc, neuron, index) => {
      return acc + neuron.calculateLoss(target[index]);
    }, 0)
    return loss;
  }
  calculateSignal(target) {
    if (this.neurons.length !== target.length) {
      throw new Error(`TARGET size and LAYER size must be the same to propagate. Usually this means there is an unexpected bios node. Target size: ${target.length}, Layer size: ${this.neurons.length}`);
    }
    return this.neurons.map((neuron, index) => neuron.calculateSignal(target[index]));
  }
}

export class JHiddenLayer extends JLayer {
  constructor(layerDef) {
    super(layerDef, JHiddenNeuron);
  }

}

export class JInputLayer extends JLayer {
  constructor(layerDef) {
    super(layerDef, JInputNeuron);
  }

  activate(inputs) {
    const { neurons } = this;
    const inputNeurons = neurons.filter(neuron => neuron instanceof JInputNeuron);
    const biasNeurons = neurons.filter(neuron => neuron instanceof JBiasNeuron);
    if (inputs.length !== inputNeurons.length) {
      throw new Error(`Input size and layer size must be equal to activate  I:${inputs.length}  N:${this.neurons.length}`);
    }
    // activate bias
    biasNeurons.forEach(neuron => neuron.activate());
    inputNeurons.forEach((neuron, index) => {
      neuron.activate(inputs[index])
    });
  }
}
