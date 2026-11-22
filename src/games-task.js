import { imageTask } from './image-task.js';

// Data: ml/prepare_games.py, then uv run ml/export_images.py games.
export default imageTask({
  taskId: 'guess-the-game-v1',
  name: 'Guess the game',
  description: 'A small CNN learns 9 games from 32×32 gameplay screenshots, tested on held-out videos.',
  file: 'games',
  classes: ['Fortnite', 'Minecraft', 'Among Us', 'Roblox', 'Terraria', 'Valorant', 'CS2', 'Dota 2', 'League'],
  size: 32,
  channels: 3,
  displaySize: 64,
  slices: 10,
  defaults: {
    rounds: 1000, batchSize: 32, localSteps: 10, optimizer: 'adam', learningRate: 0.01,
    outerLearningRate: 0.5, outerMomentum: 0.9, targetAccuracy: 0.8, timeLimitSeconds: 120,
  },
});
