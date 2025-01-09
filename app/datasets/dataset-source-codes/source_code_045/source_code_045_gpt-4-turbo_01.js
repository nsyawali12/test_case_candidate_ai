import React, { useState, useEffect } from 'react';

const CountdownTimer = ({ predefinedTime }) => {
  // Convert minutes to seconds
  const initialSeconds = predefinedTime > 0 ? predefinedTime * 60 : 0;
  const [secondsLeft, setSecondsLeft] = useState(initialSeconds);
  const [isActive, setIsActive] = useState(initialSeconds > 0);

  useEffect(() => {
    let timer = null;

    if (isActive) {
      timer = setInterval(() => {
        setSecondsLeft(seconds => {
          if (seconds > 0) return seconds - 1;
          setIsActive(false);
          return 0;
        });
      }, 1000);
    }

    // Clear interval on component unmount
    return () => clearInterval(timer);
  }, [isActive]);

  // Format time to MM:SS
  const formatTime = () => {
    const minutes = Math.floor(secondsLeft / 60);
    const seconds = secondsLeft % 60;
    return `${minutes.toString().padStart(2, '0')}:${seconds.toString().padStart(2, '0')}`;
  };

  return (
    <div>
      <div>{formatTime()}</div>
      {!isActive && secondsLeft === 0 && <div>Time is up</div>}
    </div>
  );
};

export default CountdownTimer;
