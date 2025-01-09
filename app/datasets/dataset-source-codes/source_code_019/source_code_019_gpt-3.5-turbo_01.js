import React, { useState } from 'react';

const HexToRgbConverter = () => {
    const [hexColor, setHexColor] = useState('');
    const [rgbColor, setRgbColor] = useState('');
    const [isValidHex, setIsValidHex] = useState(true);

    const handleHexChange = (event) => {
        const hex = event.target.value;
        setHexColor(hex);
        const isValid = /^#[0-9A-F]{6}$/i.test(hex);
        setIsValidHex(isValid);

        if (isValid) {
            const r = parseInt(hex.substring(1, 3), 16);
            const g = parseInt(hex.substring(3, 5), 16);
            const b = parseInt(hex.substring(5, 7), 16);
            setRgbColor(`RGB(${r}, ${g}, ${b})`);
        }
    };

    return (
        <div>
            <input type="text" value={hexColor} onChange={handleHexChange} placeholder="Enter a hex color code" />
            {!isValidHex && <p>Invalid hex color code</p>}
            <p>RGB Color: {rgbColor}</p>
            <div style={{ width: '50px', height: '50px', backgroundColor: hexColor }}></div>
        </div>
    );
};

export default HexToRgbConverter;
