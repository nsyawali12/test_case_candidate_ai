import React, { useState } from 'react';

function HexToRGBConverter() {
  const [hexColor, setHexColor] = useState("");
  const [rgbColor, setRgbColor] = useState("");
  const [valid, setValid] = useState(false);

  // Function to validate and convert hex to rgb
  function convertToRGB(hex) {
    let hexPattern = /^#([A-Fa-f0-9]{6}|[A-Fa-f0-9]{3})$/;

    // validating hex color code
    if (!hexPattern.test(hex)) {
      setValid(false);
      setRgbColor("");
      return;
    }

    setValid(true);
    let r = 0, g = 0, b = 0;

    // 3 digits
    if (hex.length === 4) {
      r = "0x" + hex[1] + hex[1];
      g = "0x" + hex[2] + hex[2];
      b = "0x" + hex[3] + hex[3];

    // 6 digits
    } else if (hex.length === 7) {
      r = "0x" + hex[1] + hex[2];
      g = "0x" + hex[3] + hex[4];
      b = "0x" + hex[5] + hex[6];
    }
    setRgbColor("RGB(" + +r + ", " + +g + ", " + +b + ")");
  }

  return (
    <div>
      <input 
        type="text" 
        value={hexColor} 
        onChange={e => {
          setHexColor(e.target.value);
          convertToRGB(e.target.value);
        }} 
        placeholder="Enter Hex Color Code"
      />
      <p>{valid ? rgbColor : "Invalid hex color code!"}</p>
      {valid && <div style={{backgroundColor: hexColor, width: '50px', height: '50px'}}/>}
    </div>
  );
}

export default HexToRGBConverter;
