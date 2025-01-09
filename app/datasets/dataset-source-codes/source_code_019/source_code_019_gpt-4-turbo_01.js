import React, { useState } from 'react';

function HexToRGBConverter() {
  const [hex, setHex] = useState('');
  const [rgb, setRGB] = useState('');
  const [color, setColor] = useState('');

  const isHexValid = (hex) => /^#([A-Fa-f0-9]{6}|[A-Fa-f0-9]{3})$/.test(hex);

  const hexToRGB = (hex) => {
    let r = 0, g = 0, b = 0;
    // 3 digits
    if (hex.length === 4) {
      r = parseInt(hex[1] + hex[1], 16);
      g = parseInt(hex[2] + hex[2], 16);
      b = parseInt(hex[3] + hex[3], 16);
    }
    // 6 digits
    else if (hex.length === 7) {
      r = parseInt(hex[1] + hex[2], 16);
      g = parseInt(hex[3] + hex[4], 16);
      b = parseInt(hex[5] + hex[6], 16);
    }
    return `RGB(${r}, ${g}, ${b})`;
  }

  const handleInputChange = (e) => {
    const inputHex = e.target.value;
    if (isHexValid(inputHex)) {
      setHex(inputHex);
      const convertedRGB = hexToRGB(inputHex);
      setRGB(convertedRGB);
      setColor(inputHex);
    } else {
      setRGB('');
      setColor('');
    }
  }

  return (
    <div>
      <input
        type="text"
        value={hex}
        placeholder="#FFFFFF"
        onChange={handleInputChange}
        style={{ marginRight: '10px' }}
      />
      <span>{rgb}</span>
      <div style={{ width: '20px', height: '20px', backgroundColor: color, display: 'inline-block', marginLeft: '10px', border: '1px solid #000' }}></div>
    </div>
  );
}

export default HexToRGBConverter;
