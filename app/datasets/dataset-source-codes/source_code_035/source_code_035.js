import React, { useState } from 'react';
import "./style.css";

const PwdStrengthMeter = () => {
  const [pwd, setPwd] = useState('');
  const [strength, setStrength] = useState('');

  const evaluateStrength = (p) => {
    const len = p.length;
    const hasNum = /\d/.test(p);
    const hasSpecial = /[!@#$%^&*(),.?":{}|<>]/g.test(p);
    const hasMixed = /[a-z]/.test(p) && /[A-Z]/.test(p);

    if (len > 8 && hasNum && hasSpecial && hasMixed) return 'Strong';
    if (len >= 6 && (hasNum || hasSpecial) && hasMixed) return 'Medium';
    return 'Weak';
  };

  const handleChange = (e) => {
    const { value } = e.target;
    setPwd(value);
    setStrength(evaluateStrength(value));
  };

  return (
    <div>
      <input type="password" value={pwd} onChange={handleChange} />
      <div>Strength: {strength}</div>
    </div>
  );
};

export default PwdStrengthMeter;