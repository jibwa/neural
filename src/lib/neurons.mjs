let numNeurons = 0;
let numConnections = 0;

class JNeuron {
  static uid() {
    numNeurons += 1;
    return numNeurons;
  }
  static rand() {
    return Math.random();
  }
  constructor({ activationFunction, wBound }) {
    Object.assign(this, {
      activationFunction,
      wBound,
      NID: JNeuron.uid(),
      outs: [],
      ins: [], // TODO WITH JSONDATA
      z: 0,
      s: 0
    });
  }

  project(to, weightMax) {
    numConnections += 1;
    const CID = numConnections;
    const { outs } = this;
    const connection = {
      to,
      CID,
      from: this,
      w: JNeuron.rand() * weightMax,
      errorSum: 0
    };
    outs.push(connection);
    to.ins.push(connection);
    return connection;
  }

  activate() {
    if (this.dropped) {
      return;
    }
    const s = this.ins.reduce((acc, { CID, w, dropped, from: { NID, z } }) => {
      if (dropped) {
        return acc;
      }
      return acc + (z * w);
    }, 0.0);
    this.z = this.activationFunction.f(s);
    return this.z;
  }

  sumConnectionErrors() {
    if (this.dropped) {
      return;
    }
    return this.outs.reduce((acc, { to, w, dropped }) => {
      if (dropped) {
        return acc;
      }
      return acc + (to.errorSignal * w)
    } , 0.0);
  }

  calculateSignal() {
    if (this.dropped) {
      return;
    }
    this.errorSignal = this.activationFunction.d(this.z) * this.sumConnectionErrors();
  }

  updateConnectionSums() {
    const { z, dropped } = this;
    if (dropped) {
      return;
    }
    this.outs.forEach(connection => {
      if (connection.dropped) {
        return;
      }
      connection.errorSum += z * connection.to.errorSignal;
    })
  }
  wBoundWeight(connection) {
    const { wBound } = this;
    if (wBound) {
      if (connection.w > wBound) {
        connection.w = wBound;
      }
      if (connection.w < -wBound) {
        connection.w = -wBound;
      }
    }
  }
  updateWeights(learningRate, batchSize, regularize) {
    if (this.dropped) {
      return;
    }
    this.outs.forEach(connection => {
      if (connection.dropped) {
        connection.errorSum = 0;
        return;
      }
      let regToAdd = 0.0;
      if (!regularize) {
        regToAdd = (regularize / parseFloat(batchSize)) * connection.w;
      }
      connection.w += learningRate * (((1 / batchSize) * connection.errorSum) - regToAdd);
      this.wBoundWeight(connection);
      connection.errorSum = 0;
    });
  }

  getConnectionSums(result) {
    return this.outs.forEach(connection => {
      result[connection.CID] = connection.errorSum;
    });
  }

  drop() {
    this.dropped = true;
    [...this.ins, ...this.outs].forEach(connection => {
      connection.dropped = true;
    });
  }
  restoreDrop() {
    if (this.dropped === true) {
      this.dropped = false;
      [...this.ins, ...this.outs].forEach(connection => {
        connection.dropped = false;
      });
    }
  }

  toJSON() {
    return {
      NID: this.NID,
      z: this.z,
      errorSignal: this.errorSignal,
      neuronType: this.constructor.name,
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
  getWeightArray() {
    return this.outs.map(out => out.w);
  }
  restoreWeights(weights) {
    this.outs.map((out, i) => out.w = weights[i]);
  }
}

export class JBiasNeuron extends JNeuron {
  activate() {
    if (this.dropped) {
      return;
    }
    this.z = 1;
    return this.z;
  }
  calculateSignal() {
    if (this.dropped) {
      return;
    }
    // Bias doesn't consider the derivitave of its 'value'
    this.errorSignal = this.sumConnectionErrors();
  }
  updateWeights(learningRate, batchSize) {
    if (this.dropped) {
      return;
    }
    this.outs.forEach(connection => {
      connection.w += learningRate * ((1 / batchSize) * connection.errorSum);
      this.wBoundWeight(connection);
      connection.errorSum = 0;
    });
  }
  // DONT dropout bias
  drop() {}
  restoreDrop() {}
}

export class JInputNeuron extends JNeuron {
  activate(input, dropout) {
    if (this.dropped) {
      return;
    }
    this.z = input * dropout;
  }
}

export class JOutputNeuron extends JNeuron {
  calculateSignal(y) {
    // no dropout here necause you cant drop output neurons
    const { z, NID } = this;
    // the reason we pass z to activationfunction derivative is because
    // it is less costly than calculating z again with the sums
    const errorSignal = ( y - z )
    //const errorSignal = (y - z) * this.activationFunction.d(z);
    this.errorSignal = errorSignal;
    return this.errorSignal;
  }
}

export class JHiddenNeuron extends JNeuron {

}
