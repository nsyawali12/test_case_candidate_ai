import React, { useState } from 'react';

const PasswordStrengthMeter = () => {
    const [password, setPassword] = useState('');

    const calculateStrength = (password) => {
        if (password.length < 6) {
            return 'Weak';
        } else if (password.length < 10 && /\d/.test(password) && /[a-zA-Z]/.test(password) && /[^A-Za-z0-9]/.test(password)) {
            return 'Medium';
        } else if (password.length >= 10 && /\d/.test(password) && /[a-zA-Z]/.test(password) && /[^A-Za-z0-9]/.test(password)) {
            return 'Strong';
        } else {
            return 'Weak';
        }
    };

    const handlePasswordChange = (e) => {
        const newPassword = e.target.value;
        setPassword(newPassword);
    };

    return (
        <div>
            <input
                type="password"
                value={password}
                onChange={handlePasswordChange}
            />
            <p>Password Strength: {calculateStrength(password)}</p>
            <div style={{ backgroundColor: calculateStrength(password) === 'Weak' ? 'red' : calculateStrength(password) === 'Medium' ? 'orange' : 'green', height: '10px', marginTop: '5px' }}></div>
        </div>
    );
};

export default PasswordStrengthMeter;
