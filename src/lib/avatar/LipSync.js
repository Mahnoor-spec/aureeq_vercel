export class LipSync {
    constructor(audioContext) {
        this.audio = audioContext;
        this.analyser = this.audio.createAnalyser();
        this.timeDomainData = new Float32Array(2048);
    }

    update() {
        this.analyser.getFloatTimeDomainData(this.timeDomainData);

        let volume = 0.0;
        for (let i = 0; i < 2048; i++) {
            volume = Math.max(volume, Math.abs(this.timeDomainData[i]));
        }

        // Amica's sigmoid curve for responsiveness
        volume = 1 / (1 + Math.exp(-45 * volume + 5));
        if (volume < 0.1) volume = 0;

        return { volume };
    }

    async playAudio(buffer) {
        const source = this.audio.createBufferSource();
        source.buffer = buffer;
        source.connect(this.audio.destination);
        source.connect(this.analyser);
        source.start();

        return new Promise((resolve) => {
            source.onended = resolve;
        });
    }
}
