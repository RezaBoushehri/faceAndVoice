let mediaRecorder;
let audioChunks = [];

document.getElementById('recordButton').addEventListener('click', async () => {
    const username = document.getElementById('username').value;
    document.getElementById('form-username').value = username;

    const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
    mediaRecorder = new MediaRecorder(stream);

    mediaRecorder.start();
    document.getElementById('recordButton').disabled = true;
    document.getElementById('stopButton').disabled = false;

    mediaRecorder.addEventListener('dataavailable', event => {
        audioChunks.push(event.data);
    });

    mediaRecorder.addEventListener('stop', async () => {
        const audioBlob = new Blob(audioChunks, { type: 'audio/wav' });
        audioChunks = []; // Reset audioChunks for the next recording

        const formData = new FormData();
        formData.append('audio', audioBlob, 'audio.wav');
        formData.append('username', username);
        
        // Append audio type as a string
        formData.append('audioType', audioBlob.type); // Save the audio type as a string

        console.log(formData);

        const response = await fetch('/register', {
            method: 'POST',
            body: formData,
        });

        const result = await response.json();
        document.getElementById('result').innerText = result.message;

        document.getElementById('recordButton').disabled = false;
        document.getElementById('stopButton').disabled = true;
    });
});

document.getElementById('stopButton').addEventListener('click', () => {
    mediaRecorder.stop();
});
