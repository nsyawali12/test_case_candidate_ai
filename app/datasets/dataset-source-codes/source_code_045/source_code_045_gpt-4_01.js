import React, { useState, useEffect } from 'react';

function CountdownTimer(props) {
    // Convert minutes to seconds
    const [timeLeft, setTimeLeft] = useState(props.predefinedTime * 60); 

    useEffect(() => {
        // Return nothing if time is zero or less
        if (props.predefinedTime <= 0) {
            return;
        }

        const intervalId = setInterval(() => {
            setTimeLeft(timeLeft => timeLeft - 1);
        }, 1000);

        return () => clearInterval(intervalId); // Cleanup on unmount
    }, [props.predefinedTime]);

    const minutes = Math.floor(timeLeft / 60); // Convert time to minutes
    const seconds = timeLeft % 60; // Get remaining seconds

    return (
        <div>
            {timeLeft <= 0 ? (
                <p>Time is up</p>
            ) : (
                <p>
                    Time left: {minutes.toString().padStart(2, '0')}:
                    {seconds.toString().padStart(2, '0')}
                </p>
            )}
        </div>
    ); 
}

export default CountdownTimer;

/* Usage: 
<CountdownTimer predefinedTime={5} /> 
*/
