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
    this.z = sigmoid(this.s);
    this.dz = deriveSigmoid(this.z);
    return this.z;
  }

  propagateSignal() {
    const weightAcc = this.outs.reduce((acc, { to, w }) => acc + (to.errorSignal * w), 0);
    this.errorSignal = this.z * (1 - this.z) * weightAcc;
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
  // do nothing for now
  activate() {
    // this is redundant but fulfills a lint rule
    this.z = 1;
    return this.z;
  }
  constructor() {
    super();
    this.z = 1;
  }
}

export class JInputNeuron extends JNeuron {
  activate(input) {
    this.z = input;
  }
}

export class JOutputNeuron extends JNeuron {
  propagateSignal(target) {
    const { z } = this;
    this.errorSignal = z - target;
    return this.errorSignal;
  }
}

export class JHiddenNeuron extends JNeuron {
  // http://briandolhansky.com/blog/2013/9/27/artificial-neural-networks-backpropagation-part-4
}
