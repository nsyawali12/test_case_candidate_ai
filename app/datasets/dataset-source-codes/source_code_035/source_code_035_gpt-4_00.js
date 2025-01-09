import React, { useState, useEffect } from 'react';

// functional component
const PasswordStrengthMeter = () => {

  const [password, setPassword] = useState("");
  const [strength, setStrength] = useState("Weak");

  // Utilize useEffect to update the strength state when password changes
  useEffect(() => {

    const lowerCase = /[a-z]/g;
    const upperCase = /[A-Z]/g;
    const numeric = /[0-9]/g;
    const special = /[@$!%*#?&]/g;

    if(password.match(lowerCase) && password.match(upperCase) && password.match(numeric) && 
        password.match(special) && password.length >= 8) {
      setStrength("Strong");
    } else if(password.match(lowerCase) || password.match(upperCase) || password.match(numeric) || 
              password.match(special)) {
      setStrength("Medium");
    } else {
      setStrength("Weak");
    }

  }, [password]);

  // Handle the password state when the user types
  const handleChange = (e) => {
    setPassword(e.target.value);
  };

  // Return a dynamic visual indicator of the strength state
  return (
    <div className="password-strength-meter">
      <input type="password" value={password} onChange={handleChange} placeholder="Enter your password" />
      <p> Password strength: <b> {strength} </b> </p>
      <div className={strength}></div>
    </div>
  )
}

export default PasswordStrengthMeter;
