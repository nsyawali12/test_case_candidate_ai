import React, { useState, useEffect } from 'react';

function PasswordStrengthMeter() {
  const [password, setPassword] = useState('');
  const [strength, setStrength] = useState('Weak');

  useEffect(() => {
    evaluateStrength(password);
  }, [password]);

  // Function to evaluate strength of the password
  const evaluateStrength = (password) => {
    const len = password.length;
    const hasNumbers = /\d/.test(password);
    const hasSpecialChars = /[!@#$%^&*(),.?":{}|<>]/.test(password);

    if (len > 8 && hasNumbers && hasSpecialChars) {
      setStrength('Strong');
    } else if (len > 5 && (hasNumbers || hasSpecialChars)) {
      setStrength('Medium');
    } else {
      setStrength('Weak');
    }
  };

  // Handle password input change
  const handleChange = (event) => {
    setPassword(event.target.value);
  };

  return (
    <div>
      <input type="password" value={password} onChange={handleChange} />
      <div>
        Strength: <span style={{ fontWeight: 'bold', color: getStrengthColor(strength) }}>{strength}</span>
      </div>
    </div>
  );
}

// Function to return color based on password strength
function getStrengthColor(strength) {
  switch (strength) {
    case 'Strong':
      return 'green';
    case 'Medium':
      return 'orange';
    default:
      return 'red';
  }
}

export default PasswordStrengthMeter;
