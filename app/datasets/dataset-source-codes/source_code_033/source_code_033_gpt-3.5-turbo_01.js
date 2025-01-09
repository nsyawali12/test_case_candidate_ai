function getColorLuminance(color) {
    const rgb = parseInt(color.replace("#", ""), 16);
    const r = (rgb >> 16) & 0xff;
    const g = (rgb >> 8) & 0xff;
    const b = (rgb >> 0) & 0xff;
    return 0.2126 * r + 0.7152 * g + 0.0722 * b;
}

function sortColorsByLuminance(colors) {
    return colors.sort((color1, color2) => getColorLuminance(color1) - getColorLuminance(color2));
}

const colors = ['black', '#FF0000', 'blue'];
const sortedColors = sortColorsByLuminance(colors);
console.log(sortedColors);
