import { JInputLayer, JOutputLayer, JHiddenLayer } from './layers.mjs';

export class JNetwork {
  constructor(layerDef) {
    const { learningRate } = layerDef;
    const input = new JInputLayer(layerDef.input);
    const lastHidden = layerDef.hidden.length - 1;
    const hidden = layerDef.hidden.map((num, index) => new JHiddenLayer(num, index === lastHidden));
    const output = new JOutputLayer(layerDef.output);
    const all = [...hidden, output];

    all.reduce((curr, next) => {
      curr.project(next);
      return next;
    }, input);
    Object.assign(this, {
      layers: { input, hidden, output },
      learningRate
    });
  }

  activate(inputs) {
    const { input, hidden, output } = this.layers;
    input.activate(inputs);
    hidden.map(layer => layer.activate());
    return output.activate();
  }

  propagateSignal(target) {
    const { layers: { output, hidden } } = this;
    return [output, ...hidden.reverse()].map(layer => layer.propagateSignal(target));
  }
  weightSumConnections() {
    const { layers: { input, hidden } } = this;
    return [input, ...hidden].map(layer => layer.weighSumConnections());
  }
  // We are going to start a new training set or group
  clearConnectionSums() {
    // eslint-disable-next-line no-return-assign
    this.getAllNeurons().forEach(neuron =>
      neuron.errorSum = 0);
  }

  updateWeights() {
    this.getAllConnections().forEach((connection) => {
      connection.w -= (this.learningRate * 0.33 * connection.errorSum);
      connection.errorSum = 0;
    });
  }

  getAllNeurons() {
    const allNeurons = [];
    this.getFlatLayers().forEach(layer =>
      allNeurons.push(...layer.neurons));
    return allNeurons;
  }

  getAllConnections() {
    const connections = [];
    this.getAllNeurons().forEach(neuron =>
      connections.push(...neuron.ins));
    return connections;
  }

  getFlatLayers() {
    const { input, hidden, output } = this.layers;
    return [input, ...hidden, output];
  }

  // inputs is optinally training set inputs to add to json
  toJSON(trainingSet = []) {
    return {
      inputOutput: trainingSet.map(([input, output]) => [input, this.activate(input), output]),
      layers: this.getFlatLayers().map(layer => layer.toJSON())
    };
  }
}
