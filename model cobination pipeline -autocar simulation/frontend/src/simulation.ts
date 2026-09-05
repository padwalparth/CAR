// src/simulation.ts
import * as THREE from 'three';
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls';

// Utility to parse query parameters
function getQueryParam(param: string): string | null {
  const urlSearch = new URLSearchParams(window.location.search);
  return urlSearch.get(param);
}

// Main entry
async function initSimulation() {
  // Container
  const container = document.getElementById('sim-container') as HTMLDivElement;

  const scene = new THREE.Scene();
  scene.background = new THREE.Color(0x111111);

  const camera = new THREE.PerspectiveCamera(45, container.clientWidth / container.clientHeight, 0.1, 1000);
  camera.position.set(0, 50, 100);

  const renderer = new THREE.WebGLRenderer({ antialias: true });
  renderer.setSize(container.clientWidth, container.clientHeight);
  container.appendChild(renderer.domElement);

  const controls = new OrbitControls(camera, renderer.domElement);
  controls.enableDamping = true;
  controls.dampingFactor = 0.05;

  // Light
  const ambient = new THREE.AmbientLight(0xffffff, 0.7);
  scene.add(ambient);
  const dirLight = new THREE.DirectionalLight(0xffffff, 0.5);
  dirLight.position.set(100, 100, 100);
  scene.add(dirLight);

  // Ground plane (placeholder)
  const planeGeom = new THREE.PlaneGeometry(200, 200);
  const planeMat = new THREE.MeshStandardMaterial({ color: 0x333333, side: THREE.DoubleSide });
  const plane = new THREE.Mesh(planeGeom, planeMat);
  plane.rotation.x = -Math.PI / 2;
  scene.add(plane);

  // Fetch inference result (expects session_id query param)
  const sessionId = getQueryParam('session_id');
  if (!sessionId) {
    console.warn('No session_id provided in URL. Simulation will show placeholder scene.');
    animate();
    return;
  }

  try {
    const resp = await fetch(`http://localhost:8000/api/inference/${sessionId}`);
    if (!resp.ok) throw new Error('Failed to fetch inference data');
    const data = await resp.json();
    const { yolo, road_segmentation, potholes } = data.result;

    // Load road mask texture if available
    if (road_segmentation?.mask_url) {
      const loader = new THREE.TextureLoader();
      loader.load(road_segmentation.mask_url, (tex) => {
        const maskMat = new THREE.MeshBasicMaterial({ map: tex, transparent: true, opacity: 0.6 });
        const maskPlane = new THREE.Mesh(planeGeom, maskMat);
        maskPlane.rotation.x = -Math.PI / 2;
        scene.add(maskPlane);
      });
    }

    // Helper to create a box for detection
    const createBox = (bbox: any, color: number) => {
      const { x1, y1, x2, y2 } = bbox;
      const width = x2 - x1;
      const height = y2 - y1;
      const depth = 5; // arbitrary depth for 3D effect
      const geometry = new THREE.BoxGeometry(width, depth, height);
      const material = new THREE.MeshStandardMaterial({ color, opacity: 0.7, transparent: true });
      const box = new THREE.Mesh(geometry, material);
      // Position: center of bbox, y at half depth
      box.position.set(x1 + width / 2 - 100, depth / 2, -(y1 + height / 2 - 100));
      scene.add(box);
    };

    // Add YOLO detections as red boxes
    if (Array.isArray(yolo?.detections)) {
      yolo.detections.forEach((det: any) => {
        createBox(det.bbox, 0xff0000);
      });
    }

    // Add pothole detections as orange boxes
    if (Array.isArray(potholes?.detections)) {
      potholes.detections.forEach((det: any) => {
        createBox(det.bbox, 0xffa500);
      });
    }
  } catch (e) {
    console.error('Error loading simulation data:', e);
  }

  animate();

  function animate() {
    requestAnimationFrame(animate);
    controls.update();
    renderer.render(scene, camera);
  }
}

initSimulation();
