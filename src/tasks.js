import brainrot from './brainrot-task.js';
import * as fashion from './fashion-task.js';
import games from './games-task.js';
import * as image from './image-task.js';
import * as linear from './linear-task.js';

// Every browser loads all tasks; the coordinator picks one. The first is the default.
export const tasks = [games, brainrot, fashion, image, linear];
