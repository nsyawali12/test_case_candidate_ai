import React, { useState, useEffect } from 'react';

const CountdownTimer = ({ predefinedTime }) => {
  // Convert minutes to seconds
  const totalSeconds = predefinedTime > 0 ? predefinedTime * 60 : 0;
  const [secondsLeft, setSecondsLeft] = useState(totalSeconds);
  const [timerActive, setTimerActive] = useState(totalSeconds > 0);

  useEffect(() => {
    let interval = null;

    if (timerActive) {
      interval = setInterval(() => {
        setSecondsLeft((prevSeconds) => {
          if (prevSeconds <= 1) {
            clearInterval(interval);
            setTimerActive(false);
            return 0;
          }
          return prevSeconds - 1;
        });
      }, 1000);
    }

    return () => clearInterval(interval);
  }, [timerActive]);

  const formatTime = (seconds) => {
    const minutes = Math.floor(seconds / 60);
    const remainingSeconds = seconds % 60;
    return `${minutes.toString().padStart(2, '0')}:${remainingSeconds.toString().padStart(2, '0')}`;
  };

  return (
    <div>
      <div>{timerActive ? formatTime(secondsLeft) : "Time is up"}</div>
    </div>
  );
};

export default CountdownTimer;
