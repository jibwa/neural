export default class TrainingMonitor {
  static neuronReport(neurons, { NID, input, sum }) {
    if (!neurons[NID]) {
      // eslint-disable-next-line no-param-reassign
      neurons[NID] = {
        inputs: [],
        sums: []
      };
    }
    neurons[NID].inputs.push(input);
    neurons[NID].sums.push(sum);
    return neurons;
  }

  static connectionReport(connections, {
    to, CID, from, weight
  }) {
    if (!connections[CID]) {
      // eslint-disable-next-line no-param-reassign
      connections[CID] = {
        to: to.NID,
        from: from.NID,
        weights: []
      };
    }
    connections[CID].weights.push(weight);
    return connections;
  }

  static layerReport(layer, { neurons = {}, connections = {} }) {
    layer.neurons.reduce(TrainingMonitor.neuronReport, neurons);
    layer.neurons.forEach(({ outs }) =>
      outs.reduce(TrainingMonitor.connectionReport, connections));
    return { neurons, connections };
  }

  constructor(trainer) {
    const { network: { layers: { input, hidden, output } } } = trainer;
    const { layerReport } = TrainingMonitor;
    Object.assign(this, {
      trainer,
      results: {
        input: {
          name: 'Input',
          ...layerReport(input, {})
        },
        hidden: hidden.map((layer, i) => ({
          name: `Hidden ${i}`,
          ...layerReport(layer, {})
        })),
        output: {
          name: 'Output',
          ...layerReport(output, {})
        }
      }
    });
  }
  monitor(iterations) {
    const { results } = this;
    const { network: { layers: { input, hidden, output } } } = this.trainer;
    //
    const { layerReport } = TrainingMonitor;
    const callback = () => {
      layerReport(input, results.input);
      hidden.forEach((layer, i) => layerReport(layer, results.hidden[i]));
      layerReport(output, results.output);
    };
    this.trainer.train(iterations, callback);
    return results;
  }
}
