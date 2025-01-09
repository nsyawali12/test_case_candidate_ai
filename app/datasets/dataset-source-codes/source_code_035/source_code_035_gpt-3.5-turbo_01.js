import React, { useState } from 'react';

const PasswordStrengthMeter = () => {
  const [password, setPassword] = useState('');
  
  const getPasswordStrength = (password) => {
    if (password.length < 6) {
      return 'Weak';
    } else if (password.length < 10) {
      if (/\d/.test(password) || /[!@#$%^&*]/.test(password)) {
        return 'Medium';
      } else {
        return 'Weak';
      }
    } else {
      if (/\d/.test(password) && /[!@#$%^&*]/.test(password)) {
        return 'Strong';
      } else {
        return 'Medium';
      }
    }
  };

  const handleChange = (e) => {
    setPassword(e.target.value);
  };

  return (
    <div>
      <input type="password" value={password} onChange={handleChange} />
      <div>
        <strong>Password Strength: </strong>{getPasswordStrength(password)}
      </div>
      <div style={{ marginTop: '10px', backgroundColor: getPasswordStrength(password) === 'Weak' ? 'red' : getPasswordStrength(password) === 'Medium' ? 'yellow' : 'green', width: '100px', height: '20px' }}>
        {/* Visual indicator to show password strength */}
      </div>
    </div>
  );
};

export default PasswordStrengthMeter;
