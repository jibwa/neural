const logistic = (x, derivate) => {
  const fx = 1 / (1 + Math.exp(-x));
  if (!derivate) {
    return fx;
  }
  return fx * (1 - fx);
};


let neurons = 0;
let connections = 0;

class JNeuron {
  static uid() {
    neurons += 1;
    return neurons;
  }
  static rand() {
    return (Math.random() * 0.2) - 0.1;
  }
  constructor(jsonData = {}) {
    const {
      NID, input, sum, outs
    } = jsonData;
    Object.assign(this, {
      NID: NID || JNeuron.uid(),
      outs: outs || [],
      ins: [], // TODO WITH JSONDATA
      input: input || 0,
      sum: sum || 0
    });
  }

  activate() {
    const { input } = this;
    this.outs.forEach((connection) => {
      // eslint-disable-next-line no-param-reassign
      connection.input = input * connection.weight;
      return connection.input;
    });
    return input;
  }

  calculateInput() {
    this.sum = this.ins.reduce((acc, { input }) => acc + input, 0);
    return logistic(this.sum);
  }

  project(to) {
    connections += 1;
    const CID = connections;
    const { outs } = this;
    const connection = {
      to,
      CID,
      from: this,
      weight: JNeuron.rand()
    };
    outs.push(connection);
    to.ins.push(connection);
    return connection;
  }

  toJSON() {
    return {
      NID: this.NID,
      input: this.input,
      sum: this.sum,
      outs: this.outs.map(({ to: { NID }, weight }) =>
        ({ NID, weight }))
    };
  }
}

export class JInputNeuron extends JNeuron {
  activate(input) {
    this.input = input - 0.5;
    return super.activate();
  }
}

export class JOutputNeuron extends JNeuron {
  activate() {
    this.input = super.calculateInput();
    return this.input;
  }

  propagate(target, learningRate) {
    const errorMargin = target - this.input;
    const logDerivativeSum = logistic(this.sum, true);
    const deltaOutputSum = errorMargin * logDerivativeSum;
    this.ins.forEach((connection) => {
      const deltaWeight = (deltaOutputSum / connection.from.input) * learningRate;
      // eslint-disable-next-line no-param-reassign
      connection.from.deltaHiddenSum =
        (deltaOutputSum / connection.weight) * logistic(connection.from.sum);
      // eslint-disable-next-line no-param-reassign
      connection.weight += (deltaWeight);
    });
  }
}

export class JHiddenNeuron extends JNeuron {
  activate() {
    this.input = super.calculateInput();
    super.activate();
    return this.input;
  }

  propagate(learningRate) {
    this.sum += this.deltaHiddenSum;
    this.ins.forEach((connection) => {
      const deltaWeight = (this.deltaHiddenSum / connection.from.input) * learningRate;
      // eslint-disable-next-line no-param-reassign
      connection.weight += deltaWeight;
    });
  }
}
