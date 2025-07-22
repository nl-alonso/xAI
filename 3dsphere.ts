// main.ts
import * as THREE from 'https://unpkg.com/three@0.168.0/build/three.module.js?module';
import { OrbitControls } from 'https://unpkg.com/three@0.168.0/examples/jsm/controls/OrbitControls.js?module';

interface ParticleData {
    base_r: number;
    original_base_r: number;
    theta: number;
    phi: number;
    amp: number;
    freq: number;
    phase: number;
    pulsePhase: number;
    pulseDirection: number;
}

private createParticles() {
    const particleCount = 5000;
    const baseRadius = 20.5;
    const radiusNoise = 4;
    const ampMax = 2;
    const freqBase = 0.5;

    const geometry = new THREE.BufferGeometry();
    const positions = new Float32Array(particleCount * 3);
    const colors = new Float32Array(particleCount * 3);

    for (let i = 0; i < particleCount; i++) {
        // Spherical coordinate calculation
        const theta = Math.random() * 2 * Math.PI;
        const phi = Math.acos(2 * Math.random() - 1);
        const base_r = baseRadius + (Math.random() - 0.5) * radiusNoise;
        const amp = Math.random() * ampMax;
        const freq = freqBase + Math.random() * 0.5;
        const phase = Math.random() * 2 * Math.PI;

        this.particleData.push({ 
            base_r, 
            original_base_r: base_r, 
            theta, 
            phi, 
            amp, 
            freq, 
            phase,
            pulsePhase: Math.random() * 2 * Math.PI,
            pulseDirection: Math.random() > 0.5 ? 1 : -1
        });

        // Convert spherical to cartesian coordinates
        positions[i * 3] = base_r * Math.sin(phi) * Math.cos(theta);
        positions[i * 3 + 1] = base_r * Math.sin(phi) * Math.sin(theta);
        positions[i * 3 + 2] = base_r * Math.cos(phi);

        // Set particle colors
        colors[i * 3] = 0.8;
        colors[i * 3 + 1] = 0.8;
        colors[i * 3 + 2] = 0.8;
    }

    geometry.setAttribute('position', new THREE.BufferAttribute(positions, 3));
    geometry.setAttribute('color', new THREE.BufferAttribute(colors, 3));

    const material = new THREE.PointsMaterial({
        size: 0.12,
        vertexColors: true,
        blending: THREE.AdditiveBlending,
        transparent: true,
        depthWrite: false
    });

    this.particles = new THREE.Points(geometry, material);
    this.particles.position.y = 7;
    this.scene.add(this.particles);
}

private animate() {
    requestAnimationFrame(this.animate.bind(this));
    
    const time = this.clock.getElapsedTime();
    const positions = this.particles.geometry.attributes.position.array as Float32Array;
    
    for (let i = 0; i < this.particleData.length; i++) {
        const pd = this.particleData[i];
        
        // Calculate radial oscillation
        const r = pd.base_r + pd.amp * Math.sin(time * pd.freq + pd.phase);
        
        // Apply spherical coordinate transformations
        positions[i * 3] = r * Math.sin(pd.phi) * Math.cos(pd.theta);
        positions[i * 3 + 1] = r * Math.sin(pd.phi) * Math.sin(pd.theta);
        positions[i * 3 + 2] = r * Math.cos(pd.phi);
    }
    
    this.particles.geometry.attributes.position.needsUpdate = true;
    
    // Continuous rotation
    this.particles.rotation.y += 0.002;
    
    this.renderer.render(this.scene, this.camera);
}

// Initialize the particle system
new ParticleSystem();