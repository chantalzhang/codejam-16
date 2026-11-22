import { imageTask } from './image-task.js';

// Data: ml/prepare_brainrot.py, then uv run ml/export_images.py brainrot. Only ~120 images
// per character, so it tops out around 60%, hence the lower target.
export default imageTask({
  taskId: 'italian-brainrot-v1',
  name: 'Italian brainrot',
  description: 'A small CNN learns 5 brainrot characters from 32×32 images. Small, noisy dataset.',
  file: 'brainrot',
  classes: ['Ballerina Cappuccina', 'Bombardino Crocodilo', 'Cappuccino Assassino', 'Tralalero Tralala', 'Tung Tung Sahur'],
  size: 32,
  channels: 3,
  displaySize: 64,
  slices: 1,
  defaults: {
    rounds: 1000, batchSize: 32, localSteps: 10, optimizer: 'adam', learningRate: 0.01,
    outerLearningRate: 0.5, outerMomentum: 0.9, targetAccuracy: 0.5, timeLimitSeconds: 120,
  },
});
