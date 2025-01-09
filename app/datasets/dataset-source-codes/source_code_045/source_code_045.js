<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Cronometru cu numărătoare inversă</title>
  <style>
    body {
      font-family: Arial, sans-serif;
      margin: 20px;
    }

    .container {
      display: flex;
      flex-direction: column;
      align-items: center;
      justify-content: center;
      min-height: 100vh;
    }

    .timer {
      font-size: 48px;
      font-weight: bold;
      margin: 0 20px;
    }

    .buttons {
      display: flex;
      justify-content: space-between;
      width: 200px;
      margin: 20px 0;
    }

    button {
      padding: 10px 20px;
      font-size: 16px;
      border: 1px solid #ccc;
      border-radius: 4px;
      cursor: pointer;
    }

    button:hover {
      background-color: #eee;
    }

    .button-start {
      background-color: #009933;
      color: #fff;
    }

    .button-stop {
      background-color: #ff4444;
      color: #fff;
    }
  </style>
</head>
<body>
  <div class="container">
    <h1>Cronometru cu numărătoare inversă</h1>
    <p id="timer">00:00</p>
    <div class="buttons">
      <button id="start-button" class="button-start">Start</button>
      <button id="stop-button" class="button-stop">Stop</button>
    </div>
  </div>

  <script>
    const timerElement = document.getElementById("timer");
    const startButton = document.getElementById("start-button");
    const stopButton = document.getElementById("stop-button");

    let intervalId;

    startButton.addEventListener("click", () => {
      // Setează timpul inițial
      let seconds = 60;

      // Funcția de actualizare a cronometrului
      const updateTimer = () => {
        // Decrementează numărul de secunde
        seconds--;

        // Afișează timpul rămas
        const minutes = Math.floor(seconds / 60);
        const remainingSeconds = seconds % 60;
        timerElement.textContent = `${minutes.toString().padStart(2, "0")}:${remainingSeconds.toString().padStart(2, "0")}`;

        // Oprește cronometrul când ajunge la 0
        if (seconds === 0) {
          clearInterval(intervalId);
        }
      };

      // Pornește cronometrul
      intervalId = setInterval(updateTimer, 1000);
    });

    stopButton.addEventListener("click", () => {
      // Oprește cronometrul
      clearInterval(intervalId);
    });
  </script>
</body>
</html>