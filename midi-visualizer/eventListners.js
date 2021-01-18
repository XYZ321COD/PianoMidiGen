tempoInput.addEventListener('change', () => visualizer.tempo = tempoInput.value);
playBtn.addEventListener('click', () => startOrStop());
visualizer.addEventListener('visualizer-ready', () => {
  playBtn.disabled = false;

});
dropZone.addEventListener('drop', loadFile)
dropZone.addEventListener('dragover', dragOverHandler)

function loadFile(e) {
  e.preventDefault()
  const file = e.dataTransfer.files[0];
  visualizer.loadFile(file);
  return false;
}

function dragOverHandler(ev) {
  console.log('File(s) in drop zone');
  ev.preventDefault();
}

function startOrStop() {
  if (visualizer.isPlaying()) {
    visualizer.stop();
    document.getElementById("icon").className = "fas fa-play";
  } else {
    visualizer.tempo = tempoInput.value;
    visualizer.start();
    document.getElementById("icon").className = "fas fa-stop";
  }
}
