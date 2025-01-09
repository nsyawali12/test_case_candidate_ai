
//"use strict";
function getLuminance(color) {
    let r, g, b;
    if (color.charAt(0) === '#') {
        let m = color.substr(1).match(color.length == 7 ? /(\S{2})/g : /(\S{1})/g);

        if (m) {
            r = parseInt(m[0], 16);
            g = parseInt(m[1], 16);
            b = parseInt(m[2], 16);
        }
    } else {
        // If it's a HTML color name
        let colors = {
            'black': '#000000',
            'blue': '#0000FF',
            'red': '#FF0000',
            //... Other color names can also be added here
        };
        if (typeof colors[color] != 'undefined') {
            return getLuminance(colors[color]);
        }
        // If color does not exist, return worst-case scenario (black)
        return 0;
    }
    // Luminance formula
    // Luminance is a measure of the total quantity of visible light emitted by an image/colour.
    // The formula to calculate luminance is 0.2126*R + 0.7152*G + 0.0722*B where R, G, B are the RGB color values.
    return 0.2126 * r + 0.7152 * g + 0.0722 * b;
}

function sortColorsByLuminance(colors) {
    colors.sort((a, b) => getLuminance(a) - getLuminance(b));
    return colors;
}

// Example usage:
// const colors = ['black', '#FF0000', 'blue'];
// console.log(sortColorsByLuminance(colors)); // Outputs: ['black', 'blue', '#FF0000']
