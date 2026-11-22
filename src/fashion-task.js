import { imageTask } from './image-task.js';

// Data: uv run ml/export_fashion.py (5 slices of 2,000 images).
export default imageTask({
  taskId: 'fashion-mnist-v1',
  name: 'Fashion-MNIST',
  description: 'A small CNN learns 10 kinds of clothing from 28×28 Fashion-MNIST images.',
  file: 'fashion',
  classes: ['T-shirt', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot'],
  size: 28,
  channels: 1,
  slices: 5,
  // From ml/sim.py: Adam needs far fewer steps than SGD, and syncing every 10 steps was fastest.
  defaults: {
    rounds: 1000, batchSize: 32, localSteps: 10, optimizer: 'adam', learningRate: 0.01,
    outerLearningRate: 0.5, outerMomentum: 0.9, targetAccuracy: null, timeLimitSeconds: 50,
  },
});
