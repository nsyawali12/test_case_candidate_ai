function getColorLuminance(color) {
  let rgbColor = typeof color === 'string' ? hexToRgb(color) : color;
  return 0.2126 * rgbColor.r + 0.7152 * rgbColor.g + 0.0722 * rgbColor.b;
}

function hexToRgb(hex) {
  let r = parseInt(hex.substr(1, 2), 16);
  let g = parseInt(hex.substr(3, 2), 16);
  let b = parseInt(hex.substr(5, 2), 16);
  return { r, g, b };
}

function sortColorsByLuminance(colors) {
  return colors.sort((color1, color2) => getColorLuminance(color1) - getColorLuminance(color2));
}

const inputColors = ['black', '#FF0000', 'blue'];
const sortedColors = sortColorsByLuminance(inputColors);
console.log(sortedColors);
