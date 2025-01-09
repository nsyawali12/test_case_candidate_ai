function sortColorsByLuminance(colors) {
    const colorToRgb = (color) => {
        if (color.charAt(0) === '#') { // HEX color
            const r = parseInt(color.substr(1, 2), 16);
            const g = parseInt(color.substr(3, 2), 16);
            const b = parseInt(color.substr(5, 2), 16);
            return { r, g, b };
        } else { // Name color
            const ctx = document.createElement('canvas').getContext('2d');
            ctx.fillStyle = color;
            return ctx.fillStyle; // Convert name to rgb
        }
    };

    const luminance = (color) => {
        const { r, g, b } = colorToRgb(color);
        // Standard RGB luminance formula
        return 0.2126 * r + 0.7152 * g + 0.0722 * b;
    };

    return colors.sort((a, b) => luminance(a) - luminance(b))
}

// Example usage
const colors = ['black', '#FF0000', 'blue'];
const sortedColors = sortColorsByLuminance(colors);
console.log(sortedColors);
