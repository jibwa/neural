const sigmoid = { // aka logistic
  f: sum => 1.0 / (1.0 + Math.exp(-sum)),
  d: f => f * (1.0 - f)
}
const softmax = {
  f: logit => Math.exp(logit)
  /*
    const denom = logits.reduce((acc, logit) =>
      acc + Math.exp(logit), 0);
    return logits.map(logit => Math.exp(logit) / denom);
  }*/
}
const softplus = { // aka rectrifier
  f: sum => Math.log(1.0 + Math.exp(sum)),
  d: f => 1.0 / (1.0 + Math.exp(-f))
};
const relu = {
  f: sum => sum > 0 ? sum : .001,
  d: f => f > 0 ? .0999 : .001
}

const logits = {
  f: sum => sum,
  d: f => 1
}

export { softplus, sigmoid, relu, logits, softmax };

