
// Here is a JavaScript function that will do that:

// Function for converting hex color to RGB
function hexToRGB(h) {
    let r = 0, g = 0, b = 0;
  
    // Conversion
    r = "0x" + h[1] + h[2];
    g = "0x" + h[3] + h[4];
    b = "0x" + h[5] + h[6];
  
    return [+r,+g,+b];
}

// Calculate the luminance of the color
function luminance(r, g, b) {
    var a = [r, g, b].map(function(v) {
        v /= 255;
        return v <= 0.03928
            ? v / 12.92
            : Math.pow( (v + 0.055) / 1.055, 2.4 );
    });
    return 0.2126 * a[0] + 0.7152 * a[1] + 0.0722 * a[2];
}

// Dictionary for color to hex conversion
const colorToHex = {
    "black" : "#000000",
    "white" : "#FFFFFF",
    "red"   : "#FF0000",
    "lime"  : "#00FF00",
    "blue"  : "#0000FF",
    "yellow": "#FFFF00",
    "cyan"  : "#00FFFF",
    "magenta": "#FF00FF",
    "silver": "#C0C0C0",
    "gray"  : "#808080",
    "maroon": "#800000",
    "olive" : "#808000",
    "green" : "#008000",
    "purple": "#800080",
    "teal"  : "#008080",
    "navy"  : "#000080",
}

// the main function
function sortColorsByLuminance(colors) {
    return colors.sort(function(a, b) {
        var c1 = (a.charAt(0) === '#') ? hexToRGB(a) : hexToRGB(colorToHex[a]);
        var c2 = (b.charAt(0) === '#') ? hexToRGB(b) : hexToRGB(colorToHex[b]);
        return luminance(c1[0], c1[1], c1[2]) - luminance(c2[0], c2[1], c2[2]);
    });
}
