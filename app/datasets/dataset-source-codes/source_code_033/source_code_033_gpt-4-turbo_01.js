function sortColorsByLuminance(colors) {
    // Helper function to convert a color name to its HEX equivalent using a canvas
    function colorNameToHex(color) {
        const ctx = document.createElement('canvas').getContext('2d');
        ctx.fillStyle = color;
        return ctx.fillStyle;
    }

    // Converts HEX code to RGB values
    function hexToRgb(hex) {
        const r = parseInt(hex.slice(1, 3), 16);
        const g = parseInt(hex.slice(3, 5), 16);
        const b = parseInt(hex.slice(5, 7), 16);
        return [r, g, b];
    }

    // Calculates luminance of an RGB triplet
    function calculateLuminance([r, g, b]) {
        return 0.2126 * r + 0.7152 * g + 0.0722 * b;
    }

    // Converts color to RGB, then to luminance, and sorts by luminance
    return colors.map(color => {
        // Determine if the color is a hex code or a color name
        const hex = color.startsWith('#') ? color : colorNameToHex(color);      
        const rgb = hexToRgb(hex);
        const luminance = calculateLuminance(rgb);
        return { color, luminance };
    })
    .sort((a, b) => a.luminance - b.luminance)
    .map(item => item.color);
}

// Example usage:
const sortedColors = sortColorsByLuminance(['black', '#FF0000', 'blue']);
console.log(sortedColors); // Output: ['black', 'blue', '#FF0000']
