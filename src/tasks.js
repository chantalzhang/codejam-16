import * as image from './image-task.js';
import * as linear from './linear-task.js';

// Every browser loads all tasks; the coordinator picks one. The first is the default.
export const tasks = [image, linear];
