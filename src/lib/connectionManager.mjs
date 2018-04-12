const differenceArrayFromSets = (origin, dropouts) => {
  const result = new Set(origin.values());
  dropouts.forEach(c => result.delete(c))
  return Array.from(result);
}
export class ConnectionManager {
  constructor() {
    this.numConnections = 0;
    this.outs = {};
    this.ins = {};
    this.conns = {};
    this.fastHash = {};
    // connections in the dropped category
    this.dropped = {};
    // connections grouped by layer
    this.layered = {};
  }
  register({ to, from, w, errorSum, layerInt }) {
    const { ins, outs, conns, layered } = this;
    const CID = this.numConnections;
    const registeredConnection = { to, from , w, errorSum, CID };
    if (!outs[from.NID]) {
      outs[from.NID] = new Set();
    }
    if (!ins[to.NID]) {
       ins[to.NID] = new Set();
    }
    if (!layered[layerInt]) {
      layered[layerInt] = new Set();
    }
    conns[CID] = registeredConnection;
    outs[from.NID].add(registeredConnection);
    ins[to.NID].add(registeredConnection);
    layered[layerInt].add(registeredConnection);
    this.numConnections += 1;
  }

  allConnections() {
    return Object.values(this.conns);
  }

  layeredConnectionWeights(fn) {
    return Object.values(this.layered).map(l =>
      Array.from(l).map(c => c.w));
  }

  // optimizes the need to scale weights for a network
  // with dropout to predict effectively
  scaleWeightsForDropout(layerInt, dropout, invert) {
    // if dropout isn't present on the layer
    if (dropout < 1) {
      let scaler = dropout;
      if (invert === true) {
        scaler = 1.0 / dropout;
      }
      const newSet = new Set();
      this.layered[layerInt].forEach(c => c.w = c.w * scaler);
    }
  }

  assembleNodeConnections(NID) {
    const { ins, outs, dropped } = this
    const inSet = ins[NID] || new Set();
    const outSet = outs[NID] || new Set();
    if (!dropped[NID]) {
      return {
        ins: Array.from(inSet || []),
        outs: Array.from(outSet || [])
      }
    }
    return {
      ins: differenceArrayFromSets(inSet, dropped[NID]),
      outs: differenceArrayFromSets(outSet, dropped[NID])
    };
  }
  getNodeConnections(NID) {
    const { fastHash } = this;
    if (!fastHash[NID]) {
      fastHash[NID] = this.assembleNodeConnections(NID)
    }
    return fastHash[NID]
  }

  dropoutConnection(conn) {
    const { conns, dropped, fastHash } = this;
    const { to, from } = conn;
    if (!dropped[to.NID]) {
      dropped[to.NID] = new Set();
    }
    if (!dropped[from.NID]) {
      dropped[from.NID] = new Set();
    }
    dropped[to.NID].add(conn);
    dropped[from.NID].add(conn);

    // make sure the fast hash gets reconstructed on the next access
    fastHash[to.NID] = undefined;
    fastHash[from.NID] = undefined;
  }
  restoreConnection(conn) {
    const { conns, dropped } = this;
    const { to, from } = conn;

    this.dropped[to.NID].delete(conn);
    this.dropped[from.NID].delete(conn);

    this.fastHash[to.NID] = undefined;
    this.fastHash[from.NID] = undefined;

  }
  dropoutNeuron(NID) {
    const { ins, outs, dropoutConnection } = this;
    if (!ins[NID] && !outs[NID]) {
      throw new Error(`Node has no inputs or outputs NID: ${NID}`);
    }
    const inSet = ins[NID] || new Set();
    const outSet = outs[NID] || new Set();
    inSet.forEach(dropoutConnection.bind(this));
    outSet.forEach(dropoutConnection.bind(this));
    return true;
  }
  restoreNeuron(NID) {
    const { ins, outs, restoreConnection } = this;
    const inSet = ins[NID] || new Set();
    const outSet = outs[NID] || new Set();
    inSet.forEach(restoreConnection.bind(this));
    outSet.forEach(restoreConnection.bind(this));
  }
}
