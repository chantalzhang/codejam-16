// Messages carry model values as raw float32 bytes in base64: ~4x smaller than JSON number
// arrays, and still plain JSON. No TF.js here, so the relay can use it too.
export function packFloats(floats) {
  const bytes = new Uint8Array(floats.buffer, floats.byteOffset, floats.byteLength);
  let binary = '';
  for (let i = 0; i < bytes.length; i += 0x8000) binary += String.fromCharCode(...bytes.subarray(i, i + 0x8000));
  return btoa(binary);
}

export function unpackFloats(text) {
  if (typeof text !== 'string') throw new Error('Expected packed values.');
  const binary = atob(text);
  if (binary.length % 4) throw new Error('Packed values have a partial number.');
  return new Float32Array(Uint8Array.from(binary, char => char.charCodeAt(0)).buffer);
}
