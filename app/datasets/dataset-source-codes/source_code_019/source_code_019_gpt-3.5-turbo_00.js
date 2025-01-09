import React, { useState } from 'react';

const hexToRgb = (hex) => {
  if (!/^#([A-Fa-f0-9]{6}|[A-Fa-f0-9]{3})$/.test(hex)) {
    return null;
  }

  let hexValue = hex.slice(1);
  if (hexValue.length === 3) {
    hexValue = hexValue[0] + hexValue[0] + hexValue[1] + hexValue[1] + hexValue[2] + hexValue[2];
  }

  const rgb = [
    parseInt(hexValue.substring(0, 2), 16),
    parseInt(hexValue.substring(2, 4), 16),
    parseInt(hexValue.substring(4, 6), 16)
  ];

  return `RGB(${rgb.join(', ')})`;
};

const ColorConverter = () => {
  const [hexColor, setHexColor] = useState('');
  const [rgbColor, setRgbColor] = useState(null);

  const handleHexInputChange = (event) => {
    const newHexColor = event.target.value;
    setHexColor(newHexColor);

    const newRgbColor = hexToRgb(newHexColor);
    setRgbColor(newRgbColor);
  };

  return (
    <div>
      <label htmlFor="hexInput">Hex Color Code:</label>
      <input
        type="text"
        id="hexInput"
        value={hexColor}
        onChange={handleHexInputChange}
        style={{ backgroundColor: hexColor }}
      />

      {rgbColor && (
        <div>
          <p>RGB Color: {rgbColor}</p>
          <div
            style={{
              width: '50px',
              height: '50px',
              backgroundColor: hexColor,
              border: '1px solid black'
            }}
          ></div>
        </div>
      )}
    </div>
  );
};

export default ColorConverter;
