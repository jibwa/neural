const rectifier = (x, derive) => {
  if (derive) {
    return 1 / (1 + Math.exp(-x));
  }
  return Math.log(1 + Math.exp(x));
};
const sigmoid = (x, derive) => {
  if (derive) {
    return x * (1 - x);
  }
  return 1 / (1 + Math.exp(-x));
};

export default { rectifier, sigmoid };

