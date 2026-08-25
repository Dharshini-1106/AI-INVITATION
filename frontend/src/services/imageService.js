// Image handling helpers: read, resize, and preview

// Read a file and return a data URL for preview
export function fileToDataUrl(file) {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = () => resolve(reader.result);
    reader.onerror = reject;
    reader.readAsDataURL(file);
  });
}

// Convert a data URL to a Blob (useful for re-uploading previewed content)
export function dataUrlToBlob(dataUrl) {
  const [meta, b64] = dataUrl.split(',');
  const mime = meta.match(/data:(.*?);/)[1];
  const bin = atob(b64);
  const arr = new Uint8Array(bin.length);
  for (let i = 0; i < bin.length; i++) {
    arr[i] = bin.charCodeAt(i);
  }
  return new Blob([arr], { type: mime });
}

// Validate a file is an accepted image type
export function isValidImageFile(file) {
  const accepted = ['image/jpeg', 'image/png', 'image/bmp', 'image/webp', 'image/tiff', 'application/pdf'];
  return accepted.includes(file.type);
}

// Normalize a file object to a clean object with dataUrl
export async function prepareFile(file) {
  const dataUrl = await fileToDataUrl(file);
  return {
    file,
    dataUrl,
    name: file.name,
    size: file.size,
    type: file.type,
  };
}
