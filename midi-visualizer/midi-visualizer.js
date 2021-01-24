let templ = document.createElement('template');
templ.innerHTML = `
<style>
  :host { display: block; }
  #container {
    overflow-x: auto;
  }
</style>
<div id="container">
  <canvas id="canvas"></canvas>
</div>
`

if (window.ShadyCSS)
  window.ShadyCSS.prepareTemplate(templ, 'midi-visualizer');

class MidiVisualizer extends HTMLElement {
  constructor() {
    super();
    this.attachShadow({mode: 'open'});
    this.shadowRoot.appendChild(document.importNode(templ.content, true));

    this.$ = {};
    this.$.container = this.shadowRoot.getElementById('container');
    this.$.canvas = this.shadowRoot.getElementById('canvas');

    this._visualizer = null;
    this._player = new mm.SoundFontPlayer(
        'https://storage.googleapis.com/download.magenta.tensorflow.org/soundfonts_js/sgm_plus'
    );
    this._player.callbackObject = {
      run: (note) => {
        const currentNotePosition = this._visualizer.drawSequence(note);

        const containerWidth = this.$.container.getBoundingClientRect().width;
        if (currentNotePosition > (this.$.container.scrollLeft + containerWidth)) {
          this.$.container.scrollLeft = currentNotePosition - 20;
        }
      },
      stop: () => {}
    };
  }

  get url() { return this.getAttribute('url'); }
  set url(value) {
    this.setAttribute('url', value);
    this._fetchMidi();
  }

  get tempo() { return this._player.desiredQPM; }
  set tempo(value) {
    this.setAttribute('tempo', value);
    this._player.setTempo(value);
  }

  get noteSequence() { return this._noteSequence; }
  set noteSequence(value) {
    if (value != this._noteSequence) {
      this._noteSequence = value;
      this._initializeVisualizer();
    }
  }

  isPlaying() { return this._player.isPlaying(); }

  start() {
    this.$.container.scrollLeft = 0;
    mm.Player.tone.context.resume();
    this._player.start(this.noteSequence);
  }

  stop() {
    this._player.stop();
  }

  loadFile(blob) {
    this._parseMidiFile(blob);
  }

  static get observedAttributes() { return ['url', 'tempo']; }
  attributeChangedCallback(attr, oldValue, newValue) {
    if (oldValue === newValue) return;
    this[attr] = newValue;
  }

  _fetchMidi() {
    fetch(this.url)
    .then((response) => {
      return response.blob();
    })
    .then((blob) => {
      this._parseMidiFile(blob);
    })
    .catch(function(error) {
      console.log('Error', error.message);
    });
  }

  _parseMidiFile(file) {
    const reader = new FileReader();

    reader.onload = async (e) => {
      this.noteSequence = mm.midiToSequenceProto(e.target.result);
      this._player.setTempo(this.noteSequence.tempos[0].qpm);
    };

    reader.readAsBinaryString(file);
  }

  async _initializeVisualizer() {
    this._visualizer = new NoteSequenceDrawing(this.noteSequence, this.$.canvas);
    await this._player.loadSamples(this.noteSequence);
    this.dispatchEvent(new CustomEvent('visualizer-ready'));
  }
}

class NoteSequenceDrawing {
  constructor(sequence, canvas) {
    this.config = {
      noteHeight: 6,
      noteSpacing: 1,
      pixelsPerTimeStep: 30, 
      minPitch: 100,
      maxPitch: 1
    }

    this.noteSequence = sequence;

    this.ctx = canvas.getContext('2d');
    const size = this.getCanvasSize();

    this.height = size.height;
    this.ctx.canvas.width  = size.width;
    this.ctx.canvas.height = size.height;

    this.drawSequence();
  }

  drawSequence(currentNote) {
    this.ctx.clearRect(0, 0, this.ctx.canvas.width, this.ctx.canvas.height);
    let currentNotePosition;

    for (let i = 0; i < this.noteSequence.notes.length; i++) {
      const note = this.noteSequence.notes[i];

      const offset = this.config.noteSpacing * (i + 1);
      const x = (note.startTime * this.config.pixelsPerTimeStep) + offset;
      const w = (note.endTime - note.startTime) * this.config.pixelsPerTimeStep;


      const y = this.height - ((note.pitch - this.config.minPitch) * this.config.noteHeight);

      const opacity = note.velocity / 100 + 0.2;
      if (this.isPaintingCurrentNote(note, currentNote)) {
        this.ctx.fillStyle=`rgba(99, 22, 119, ${opacity})`; 
      } else if (i <= this.primerNotes) {
        this.ctx.fillStyle=`rgba(111, 201, 198, ${opacity})`; 
      }
      else {
        this.ctx.fillStyle=`rgba(8, 41, 64, ${opacity})`;  
      }
      this.ctx.fillRect(x, y, w, this.config.noteHeight);

      if (this.isPaintingCurrentNote(note, currentNote)) {
        currentNotePosition = x;
      }
    }
    return currentNotePosition;
   }

  getCanvasSize() {
    for (let note of this.noteSequence.notes) {
      if (note.pitch < this.config.minPitch) {
        this.config.minPitch = note.pitch;
      }
      if (note.pitch > this.config.maxPitch) {
        this.config.maxPitch = note.pitch;
      }
    }

    this.config.minPitch -= 2;
    this.config.maxPitch += 2;

    const height = (this.config.maxPitch - this.config.minPitch) * this.config.noteHeight;

    const numNotes = this.noteSequence.notes.length;
    const lastNote = this.noteSequence.notes[numNotes - 1];
    const width = (numNotes * this.config.noteSpacing) + (lastNote.endTime * this.config.pixelsPerTimeStep);

    return {width, height};
  }

  isPaintingCurrentNote(note, currentNote) {
    return currentNote &&
          ((note.startTime == currentNote.startTime) ||
           (note.endTime >= currentNote.endTime) &&
          (note.startTime <= currentNote.startTime))
  }
}

window.customElements.define('midi-visualizer', MidiVisualizer);
