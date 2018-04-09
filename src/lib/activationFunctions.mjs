const sigmoid = { // aka logistic
  f: sum => 1.0 / (1.0 + Math.exp(-sum)),
  d: f => f * (1.0 - f)
}
const softplus = { // aka rectrifier
  f: sum => Math.log(1.0 + Math.exp(sum)),
  d: f => 1.0 / (1.0 + Math.exp(-f))
};
const relu = {
  f: sum => sum > 0 ? sum : 0,
  d: f => f > 0 ? 1.0 : 0
}

const logits = {
  f: sum => sum,
  d: f => 1
}

export { softplus, sigmoid, relu, logits };

