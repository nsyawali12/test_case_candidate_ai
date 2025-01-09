import React, { useState, useEffect } from 'react';

// define the timer component
function Timer({ predefinedTime }) {
  // set the initial state
  const [timeLeft, setTimeLeft] = useState(predefinedTime * 60);

  // update every second
  useEffect(() => {
    // return early if timeLeft is invalid or already zero
    if (isNaN(timeLeft) || timeLeft <= 0) return;

    // countdown logic
    const timer = setInterval(() => {
      setTimeLeft(timeLeft => timeLeft - 1);
    }, 1000);

    // cleanup function
    return () => clearInterval(timer);
  }, [predefinedTime, timeLeft]); // dependencies

  // format the timeLeft into mm:ss
  const minutes = Math.floor(timeLeft / 60);
  const seconds = timeLeft - minutes * 60;
  const formattedTimeLeft = `${minutes.toString().padStart(2, 0)}:${seconds.toString().padStart(2, 0)}`;

  return (
    <div>
      <p>{timeLeft > 0 ? formattedTimeLeft : "Time is up"}</p>
    </div>
  );
}

export default Timer;
