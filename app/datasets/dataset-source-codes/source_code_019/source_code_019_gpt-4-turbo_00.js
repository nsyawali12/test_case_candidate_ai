import React, { useState } from 'react';

function HexToRGBConverter() {
  const [hex, setHex] = useState('');
  const [rgb, setRgb] = useState('');
  const [isValid, setIsValid] = useState(true);

  const handleChange = (event) => {
    const hexValue = event.target.value;
    // Check if the hex input is valid
    if (/^#([0-9A-F]{3}){1,2}$/i.test(hexValue)) {
      setHex(hexValue);
      setIsValid(true);
      convertToRGB(hexValue);
    } else {
      setIsValid(false);
      setRgb('');
    }
  };

  const convertToRGB = (hex) => {
    let r = 0, g = 0, b = 0;

    // 3 digits or 6 digits
    if (hex.length === 4) {
      r = parseInt(hex[1] + hex[1], 16);
      g = parseInt(hex[2] + hex[2], 16);
      b = parseInt(hex[3] + hex[3], 16);
    } else if (hex.length === 7) {
      r = parseInt(hex[1] + hex[2], 16);
      g = parseInt(hex[3] + hex[4], 16);
      b = parseInt(hex[5] + hex[6], 16);
    }

    setRgb(`RGB(${r}, ${g}, ${b})`);
  };

  return (
    <div>
      <input 
        type="text" 
        value={hex} 
        onChange={handleChange} 
        placeholder="#FFFFFF"
        style={{ borderColor: isValid ? 'default' : 'red' }}
      />
      {isValid && rgb && (
        <div>
          <div style={{ backgroundColor: hex, width: '50px', height: '50px', display: 'inline-block', marginLeft: '10px' }}></div>
          <span style={{ marginLeft: '10px' }}>{rgb}</span>
        </div>
      )}
    </div>
  );
}

export default HexToRGBConverter;
