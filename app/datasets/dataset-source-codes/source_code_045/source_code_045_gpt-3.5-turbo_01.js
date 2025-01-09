import React, { useState, useEffect } from 'react';

const CountdownTimer = ({ predefinedTime }) => {
  const [remainingTime, setRemainingTime] = useState(predefinedTime * 60);

  useEffect(() => {
    const timer = setInterval(() => {
      setRemainingTime((prevTime) => {
        if (prevTime <= 0) {
          clearInterval(timer);
          return 0;
        }
        return prevTime - 1;
      });
    }, 1000);

    return () => clearInterval(timer);
  }, []);

  const formatTime = (time) => {
    const minutes = Math.floor(time / 60).toString().padStart(2, '0');
    const seconds = (time % 60).toString().padStart(2, '0');
    return `${minutes}:${seconds}`;
  };

  return (
    <div>
      <h1>{formatTime(remainingTime)}</h1>
      {remainingTime <= 0 && <p>Time is up</p>}
    </div>
  );
};

export default CountdownTimer;
