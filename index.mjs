import express from 'express';
import expressLess from 'express-less';
import erv from 'express-react-views';
import registerExamples from './examples/';

const app = express();
if (process.version !== 'v9.3.0') {
  console.log(`This application is written for node v9.3.0, you are running: ${process.version}`);
  console.log('Please install the correct version with NVM');
  console.log('Attempting to start the APP. USE AT YOUR OWN RISK');
}

app.set('views', './examples/views');
app.set('view engine', 'jsx');
app.engine('jsx', erv.createEngine());
app.use('/style', expressLess('./less', { debug: true }));

registerExamples(app);

app.listen(4000, () => {
  console.log('Example app listening on port 4000!');
  console.log('TRAIN/VISUALIZE (1000 iterations): http://localhost:4000/xorvisualize/1000');
});
