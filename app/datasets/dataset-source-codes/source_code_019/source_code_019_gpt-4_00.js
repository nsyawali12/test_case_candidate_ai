import React, { useState } from 'react';

function App() {
  const [hex, setHex] = useState('');
  const [color, setColor] = useState('');

  const hexToRgb = (hex) => {
    let rgb = [];
    if (hex[0] === '#') {
      hex = hex.substr(1);
    }

    if (hex.length === 6) {
      for (let i = 0; i < 6; i += 2) {
        rgb.push(parseInt(hex.substr(i, 2), 16));
      }
    }

    return `RGB(${rgb.join(', ')})`;
  };

  const handleInputChange = (e) => {
    let hexValue = e.target.value;
    let rgbValue = hexToRgb(e.target.value);

    if (hexValue[0] !== '#' || hexValue.length !== 7) {
      setColor('');
      setHex(hexValue);
      return;
    }

    setHex(hexValue);
    setColor(rgbValue);
  };

  return (
    <div>
      <input type="text" value={hex} onChange={handleInputChange}/>
      <p>RGB: {color}</p>
      <div style={{backgroundColor: `${hex}`}} title="Color preview" />
    </div>
  );
}

export default App;
