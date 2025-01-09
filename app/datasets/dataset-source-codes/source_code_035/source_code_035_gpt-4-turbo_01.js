import React, { useState, useEffect } from 'react';

// CSS styles for different strength levels
const styles = {
  weak: {
    color: 'red'
  },
  medium: {
    color: 'orange'
  },
  strong: {
    color: 'green'
  }
};

// Determines the strength of the password
const getPasswordStrength = password => {
  let strength = 0;
  if (password.length > 5) strength += 1;
  if (password.length > 8) strength += 1;
  if (/\d/.test(password)) strength += 1;
  if (/[!@#$%^&*(),.?":{}|<>]/.test(password)) strength += 1;

  switch (strength) {
    case 0:
    case 1:
      return 'weak';
    case 2:
    case 3:
      return 'medium';
    case 4:
      return 'strong';
    default:
      return 'weak';
  }
};

// React functional component
const PasswordStrengthMeter = () => {
  const [password, setPassword] = useState('');
  const [strength, setStrength] = useState('weak');

  // Update strength as the password changes
  useEffect(() => {
    setStrength(getPasswordStrength(password));
  }, [password]);

  return (
    <div>
      <input
        type="password"
        value={password}
        onChange={e => setPassword(e.target.value)}
        placeholder="Enter your password"
      />
      <div style={styles[strength]}>
        Password Strength: {strength.toUpperCase()}
      </div>
    </div>
  );
};

export default PasswordStrengthMeter;
