const rectifier = x => Math.log(1 + Math.exp(x));
const deriveRectifier = x => 1 / (1 + Math.exp(-x));

const sigmoid = (x) => {
  const fx = 1 / (1 + Math.exp(-x));
  return fx;
};
const deriveSigmoid = fx => fx * (1 - fx);


let numNeurons = 0;
let numConnections = 0;

class JNeuron {
  static uid() {
    numNeurons += 1;
    return numNeurons;
  }
  static rand() {
    return (Math.random() - 0.5);
  }
  constructor() {
    Object.assign(this, {
      NID: JNeuron.uid(),
      outs: [],
      ins: [], // TODO WITH JSONDATA
      z: 0,
      s: 0
    });
  }

  project(to) {
    numConnections += 1;
    const CID = numConnections;
    const { outs } = this;
    const connection = {
      to,
      CID,
      from: this,
      w: JNeuron.rand(),
      errorSum: 0
    };
    outs.push(connection);
    to.ins.push(connection);
    return connection;
  }

  activate() {
    this.s = this.ins.reduce((acc, { w, from: { z } }) =>
      acc + (z * w), 0);
    this.z = this.activationFunction(this.s);
    return this.z;
  }

  propagateSignal() {
    const weightAcc = this.outs.reduce((acc, { to, w }) => acc + (to.errorSignal * w), 0);
    this.errorSignal = this.deriveFunction(this.z) * weightAcc;
    return this.errorSignal;
  }

  weighSumConnections() {
    const { outs, z } = this;
    outs.forEach((connection) => {
      // console.log({CID: connection.CID, z, toSignal: connection.to.errorSignal});
      connection.errorSum += (z * connection.to.errorSignal);
    });
  }

  toJSON() {
    return {
      NID: this.NID,
      z: this.z,
      s: this.s,
      errorSignal: this.errorSignal,
      bias: this.z === 1,
      outs: this.outs.map(({
        CID,
        to: { NID },
        w,
        errorSum
      }) => ({
        NID,
        w,
        CID,
        errorSum
      })),
      ins: this.ins.map(({
        CID,
        from: { NID },
        w,
        errorSum
      }) => ({
        NID,
        w,
        CID,
        errorSum
      }))
    };
  }
}

export class JBiasNeuron extends JNeuron {
constructor(lastHidden) {
    super();
    this.activationFunction = sigmoid;
    if (lastHidden) {
      this.deriveFunction = deriveSigmoid;
    } else {
      this.deriveFunction = deriveRectifier;
    }
    this.z = 1;
  }

  activate() {
    this.z = 1;
    return this.z;
  }
}

export class JInputNeuron extends JNeuron {
  constructor() {
    super();
    this.deriveFunction = deriveRectifier;
  }
  activate(input) {
    this.z = input;
  }
}

export class JOutputNeuron extends JNeuron {
  constructor() {
    super();
    this.activationFunction = sigmoid;
  }

  propagateSignal(y) {
    const { z } = this;
    this.errorSignal = z - y;
    return this.errorSignal;
  }
}

export class JHiddenNeuron extends JNeuron {
  constructor(lastHidden) {
    super();
    this.activationFunction = sigmoid;
    if (lastHidden) {
      this.deriveFunction = deriveSigmoid;
    } else {
      this.deriveFunction = deriveRectifier;
    }
    //this.activateFunction = rectifier;
  }
  // http://briandolhansky.com/blog/2013/9/27/artificial-neural-networks-backpropagation-part-4
}
