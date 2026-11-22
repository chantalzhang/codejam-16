// Draws a task's example image ({ width, height, channels: 1 or 3, pixels: 0–255 }) into a
// PNG data URL, so any task's images can be shown. CSS scales it up with crisp pixels.
export function imageUrl({ width, height, channels, pixels }) {
  const canvas = document.createElement('canvas');
  canvas.width = width;
  canvas.height = height;
  const context = canvas.getContext('2d');
  const image = context.createImageData(width, height);
  for (let i = 0; i < width * height; i++) {
    const rgb = channels === 1 ? [pixels[i], pixels[i], pixels[i]] : pixels.subarray(i * 3, i * 3 + 3);
    image.data.set([...rgb, 255], i * 4);
  }
  context.putImageData(image, 0, 0);
  return canvas.toDataURL();
}
