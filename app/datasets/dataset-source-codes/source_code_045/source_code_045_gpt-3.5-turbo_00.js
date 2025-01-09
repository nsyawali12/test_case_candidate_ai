import React, { useState, useEffect } from 'react';

const CountdownTimer = ({ predefinedTime }) => {
    const [time, setTime] = useState(predefinedTime * 60);

    useEffect(() => {
        const interval = setInterval(() => {
            setTime(prevTime => {
                if (prevTime > 0) {
                    return prevTime - 1;
                } else {
                    clearInterval(interval);
                    return prevTime;
                }
            });
        }, 1000);

        return () => clearInterval(interval);
    }, []);

    const formatTime = () => {
        const minutes = Math.floor(time / 60).toString().padStart(2, '0');
        const seconds = (time % 60).toString().padStart(2, '0');
        return `${minutes}:${seconds}`;
    };

    return (
        <div>
            <h1>Countdown Timer: {formatTime()}</h1>
            {time === 0 && <p>Time is up</p>}
        </div>
    );
};

export default CountdownTimer;
