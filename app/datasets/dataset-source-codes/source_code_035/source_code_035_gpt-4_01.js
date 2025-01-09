import React, { useState, useEffect } from 'react';

// Component
const PasswordStrengthMeter = () => {
  const [password, setPassword] = useState('');
  const [strength, setStrength] = useState('Weak');

  // Function to evaluate password strength
  const evaluateStrength = (password) => {
    let strengthValue = 'Weak';
    const strongRegex = new RegExp('^(?=.*[a-z])(?=.*[A-Z])(?=.*[0-9])(?=.*[!@#\$%\^&\*])(?=.{8,})'); // Strong: 8 Characters, uppercase, lowercase, numbers, special characters
    const mediumRegex = new RegExp('^(?=.*[a-z])(?=.*[A-Z])(?=.*[0-9])(?=.{6,})'); // Medium: 6 Characters, uppercase, lowercase, numbers

    if(strongRegex.test(password)) {
      strengthValue = 'Strong';
    } else if(mediumRegex.test(password)) {
      strengthValue = 'Medium';
    }
    return strengthValue;
  }

  // Update strength value whenever password changes
  useEffect(() => {
    setStrength(evaluateStrength(password));
  }, [password]);

  return (
    <div>
      <input
        type="password"
        value={password}
        onChange={(e) => setPassword(e.target.value)}
        placeholder='Enter your password'
      />
      <p>Strength: {strength}</p>
    </div>
  );
};

export default PasswordStrengthMeter;
