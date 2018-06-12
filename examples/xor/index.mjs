import XORTrainer from './XORTrainer.mjs';
let _trainer

// Initialize the trainer only if an endpoint is requested
const getTrainer = () => {
  if (!_trainer) {
    _trainer = new XORTrainer();
  }
  return _trainer;
}

const renderIndex = (req, res) => {
  const trainer = getTrainer();
  const preds = trainer.pred();
  const { network, trainingSet, totalIterations } = trainer;
  res.render('xor', { network, trainingSet, preds, totalIterations });
};

const getIterations = (req) => {
  return parseInt(req.params.iterations, 10);
}
const register = (app) => {
  const root = '/xor';

  app.get(`${root}/train/:iterations`, (req, res) => {
    const trainer = getTrainer();
    const iterations = getIterations(req);
    trainer.train(iterations);
    res.json(trainer.network.toJSON(trainer.trainingSet));
  });

  app.get(`${root}/gradient`, (req, res) => res.json(getTrainer().checkGradient()));

  app.get(`${root}/visualize`, renderIndex);

  app.get(`${root}/visualize/:iterations`, (req, res) => {
    const trainer = getTrainer();
    const iterations = getIterations(req);
    trainer.train(iterations);
    renderIndex(req, res);
  });

  app.get(`${root}/reset`, (req, res) => {
    _trainer = new XORTrainer();
    res.json({reset: true});
  });
}
export default register;
