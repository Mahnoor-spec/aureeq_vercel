import * as THREE from 'three';
import { GLTFLoader } from 'three/examples/jsm/loaders/GLTFLoader';
import { VRMLoaderPlugin, VRMUtils } from '@pixiv/three-vrm';
import { LipSync } from './LipSync';

export class AvatarRenderer {
    constructor(canvas) {
        this.canvas = canvas;
        this.scene = new THREE.Scene();
        this.camera = new THREE.PerspectiveCamera(30, canvas.clientWidth / canvas.clientHeight, 0.1, 20.0);
        this.renderer = new THREE.WebGLRenderer({ canvas, alpha: true, antialias: true });
        this.clock = new THREE.Clock();
        this.currentVRM = null;

        // Audio / LipSync - Lazy initialized
        this.audioCtx = null;
        this.lipSync = null;

        // Animation State
        this.blinkTimer = 0;
        this.nextBlinkTime = 2; // seconds

        this.init();
    }

    init() {
        this.renderer.setSize(this.canvas.clientWidth, this.canvas.clientHeight);
        this.renderer.setPixelRatio(window.devicePixelRatio);

        // Camera: Adjusted for a centered portrait view
        this.camera.position.set(0.0, 0.7, 1.4);
        this.camera.lookAt(0.0, 0.5, 0.0);

        const light = new THREE.DirectionalLight(0xffffff, 1.5);
        light.position.set(1.0, 1.0, 1.0).normalize();
        this.scene.add(light);

        const ambient = new THREE.AmbientLight(0xffffff, 0.9);
        this.scene.add(ambient);

        window.addEventListener('resize', () => this.onResize());
        this.animate();
    }

    onResize() {
        this.camera.aspect = this.canvas.clientWidth / this.canvas.clientHeight;
        this.camera.updateProjectionMatrix();
        this.renderer.setSize(this.canvas.clientWidth, this.canvas.clientHeight);
    }

    loadVRM(url) {
        const loader = new GLTFLoader();
        loader.register((parser) => {
            return new VRMLoaderPlugin(parser);
        });

        loader.load(url, (gltf) => {
            const vrm = gltf.userData.vrm;
            VRMUtils.removeUnnecessaryVertices(gltf.scene);
            VRMUtils.combineSkeletons(gltf.scene);

            // previous Math.PI was showing the back.
            vrm.scene.rotation.y = 0;

            // Position: Lowered more to look perfectly centered in the frame
            vrm.scene.position.set(0, -1.1, 0);

            this.scene.add(vrm.scene);
            this.currentVRM = vrm;
            console.log('✅ VRM Loaded successfully and added to scene at:', vrm.scene.position);
        }, (progress) => {
            console.log('VRM Loading progress:', (progress.loaded / progress.total * 100) + '%');
        }, (error) => {
            console.error('❌ Error loading VRM:', error);
        });
    }

    async speak(audioBuffer) {
        if (!this.audioCtx) {
            this.audioCtx = new (window.AudioContext || window.webkitAudioContext)();
            this.lipSync = new LipSync(this.audioCtx);
        }

        if (this.audioCtx.state === 'suspended') {
            await this.audioCtx.resume();
        }
        await this.lipSync.playAudio(audioBuffer);
    }

    async speakFromUrl(url) {
        try {
            if (!this.audioCtx) {
                this.audioCtx = new (window.AudioContext || window.webkitAudioContext)();
                this.lipSync = new LipSync(this.audioCtx);
            }
            const response = await fetch(url);
            const arrayBuffer = await response.arrayBuffer();
            const audioBuffer = await this.audioCtx.decodeAudioData(arrayBuffer);
            await this.speak(audioBuffer);
        } catch (error) {
            console.error("Error playing audio:", error);
        }
    }

    animate() {
        requestAnimationFrame(this.animate.bind(this));
        const delta = this.clock.getDelta();

        if (this.currentVRM) {
            this.currentVRM.update(delta);

            // 1. Idle Breathing (Sine on Chest)
            // 1. Idle Breathing (Spine + Arms)
            const s = Math.sin(this.clock.elapsedTime);

            const spine = this.currentVRM.humanoid.getNormalizedBoneNode('spine');
            if (spine) {
                spine.rotation.x = s * 0.02;
                spine.rotation.y = s * 0.01;
            }

            const leftArm = this.currentVRM.humanoid.getNormalizedBoneNode('leftUpperArm');
            const rightArm = this.currentVRM.humanoid.getNormalizedBoneNode('rightUpperArm');

            if (leftArm) {
                // Correcting Rotation Axis based on feedback
                // Left Arm: Negative Z to rotate down (CW)
                leftArm.rotation.z = -1.2 + (s * 0.02);
            }
            if (rightArm) {
                // Right Arm: Positive Z to rotate down (CCW)
                rightArm.rotation.z = 1.2 - (s * 0.02);
            }

            // 2. Blinking
            this.blinkTimer += delta;
            if (this.blinkTimer > this.nextBlinkTime) {
                this.blinkTimer = 0;
                this.nextBlinkTime = Math.random() * 4 + 2;
                this.currentVRM.expressionManager.setValue('blink', 1);
                setTimeout(() => {
                    if (this.currentVRM) this.currentVRM.expressionManager.setValue('blink', 0);
                }, 150);
            }

            // 3. LipSync
            if (this.lipSync) {
                const { volume } = this.lipSync.update();
                this.currentVRM.expressionManager.setValue('aa', volume);
            }
        }

        this.renderer.render(this.scene, this.camera);
    }
}
