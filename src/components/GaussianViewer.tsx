"use client";

import { useEffect, useRef, useState } from "react";
import { motion, AnimatePresence } from "framer-motion";
import {
  Maximize2,
  Download,
  MousePointer2,
  X,
  RotateCcw,
  Move,
} from "lucide-react";
import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";

interface GaussianViewerProps {
  plyUrl: string;
}

export default function GaussianViewer({ plyUrl }: GaussianViewerProps) {
  const sceneContainerRef = useRef<HTMLDivElement>(null);
  const containerRef = useRef<HTMLDivElement>(null);
  const viewerRef = useRef<unknown>(null);
  const rendererRef = useRef<THREE.WebGLRenderer | null>(null);
  const controlsRef = useRef<OrbitControls | null>(null);
  const [isLoading, setIsLoading] = useState(true);
  const [loadProgress, setLoadProgress] = useState(0);
  const [error, setError] = useState<string | null>(null);
  const [isExpanded, setIsExpanded] = useState(false);

  useEffect(() => {
    if (!containerRef.current || !plyUrl) return;

    let disposed = false;
    let animationFrameId: number;

    const initGaussianSplatViewer = async () => {
      try {
        setIsLoading(true);
        setError(null);
        setLoadProgress(0);

        if (containerRef.current) {
          containerRef.current.innerHTML = "";
        }

        const GaussianSplats3D = await import("@mkkellogg/gaussian-splats-3d");

        if (disposed || !containerRef.current) return;

        const container = containerRef.current;
        const width = container.clientWidth;
        const height = container.clientHeight;

        // Create renderer
        const renderer = new THREE.WebGLRenderer({
          antialias: true,
          alpha: true,
        });
        renderer.setSize(width, height);
        renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
        container.appendChild(renderer.domElement);
        rendererRef.current = renderer;

        // Create camera
        const camera = new THREE.PerspectiveCamera(45, width / height, 0.01, 500);
        camera.position.set(0, 0, -3);
        camera.up.set(0, -1, 0);
        camera.lookAt(0, 0, 0);

        // Create controls
        const controls = new OrbitControls(camera, renderer.domElement);
        controls.enableDamping = true;
        controls.dampingFactor = 0.1;
        controls.rotateSpeed = 0.8;
        controls.enableZoom = false; // We handle zoom ourselves
        controls.enablePan = true;
        controls.panSpeed = 0.8;
        controls.screenSpacePanning = true;
        controls.target.set(0, 0, 0);
        controls.minPolarAngle = 0.1;
        controls.maxPolarAngle = Math.PI - 0.1;
        controls.touches = {
          ONE: THREE.TOUCH.ROTATE,
          TWO: THREE.TOUCH.DOLLY_PAN,
        };
        controls.enabled = true;
        controlsRef.current = controls;

        // Custom wheel handler for flying through the scene
        const handleWheel = (e: WheelEvent) => {
          e.preventDefault();
          const dollySpeed = 0.002;
          const delta = e.deltaY * dollySpeed;
          const forward = new THREE.Vector3();
          camera.getWorldDirection(forward);
          camera.position.addScaledVector(forward, -delta);
          controls.target.addScaledVector(forward, -delta);
        };

        renderer.domElement.addEventListener("wheel", handleWheel, {
          passive: false,
        });

        // Create Gaussian Splat viewer
        const viewer = new GaussianSplats3D.Viewer({
          renderer: renderer,
          camera: camera,
          selfDrivenMode: false,
          useBuiltInControls: false,
          sharedMemoryForWorkers: false,
          dynamicScene: false,
          sceneRevealMode: GaussianSplats3D.SceneRevealMode.Gradual,
          antialiased: true,
          focalAdjustment: 1.0,
        });

        viewerRef.current = { viewer, camera, renderer, controls };

        await viewer.addSplatScene(plyUrl, {
          splatAlphaRemovalThreshold: 5,
          showLoadingUI: false,
          progressiveLoad: true,
          onProgress: (progress: number) => {
            setLoadProgress(Math.min(100, Math.round(progress)));
          },
        });

        if (disposed) return;

        setIsLoading(false);

        // Animation loop
        const animate = () => {
          if (disposed) return;
          animationFrameId = requestAnimationFrame(animate);
          controls.update();
          viewer.update();
          viewer.render();
        };
        animate();

        // Handle resize
        const handleResize = () => {
          if (!containerRef.current || disposed) return;
          const newWidth = containerRef.current.clientWidth;
          const newHeight = containerRef.current.clientHeight;
          camera.aspect = newWidth / newHeight;
          camera.updateProjectionMatrix();
          renderer.setSize(newWidth, newHeight);
        };

        window.addEventListener("resize", handleResize);

        return () => {
          window.removeEventListener("resize", handleResize);
          renderer.domElement.removeEventListener("wheel", handleWheel);
        };
      } catch (err) {
        console.error("Error initializing viewer:", err);
        if (!disposed) {
          setError("Failed to load 3D scene. The model may still be processing.");
          setIsLoading(false);
        }
      }
    };

    initGaussianSplatViewer();

    return () => {
      disposed = true;

      if (animationFrameId) {
        cancelAnimationFrame(animationFrameId);
      }

      if (viewerRef.current) {
        const viewerObj = viewerRef.current as {
          viewer?: { dispose: () => void };
        };
        if (viewerObj.viewer?.dispose) {
          viewerObj.viewer.dispose();
        }
      }

      if (rendererRef.current) {
        rendererRef.current.dispose();
      }

      if (controlsRef.current) {
        controlsRef.current.dispose();
      }

      viewerRef.current = null;
      rendererRef.current = null;
      controlsRef.current = null;
    };
  }, [plyUrl]);

  const handleExpand = () => setIsExpanded(true);
  const handleCollapse = () => setIsExpanded(false);

  const handleDownload = () => {
    const link = document.createElement("a");
    link.href = plyUrl;
    link.download = "scene.ply";
    link.click();
  };

  const handleReset = () => {
    const viewer = viewerRef.current as {
      camera?: THREE.PerspectiveCamera;
      controls?: OrbitControls;
    } | null;

    if (viewer?.camera && viewer?.controls) {
      viewer.camera.position.set(0, 0, -3);
      viewer.camera.up.set(0, -1, 0);
      viewer.controls.target.set(0, 0, 0);
      viewer.camera.lookAt(0, 0, 0);
      viewer.controls.update();
    }
  };

  // Handle resize when expanded/collapsed
  useEffect(() => {
    const handleResize = () => {
      if (!containerRef.current || !rendererRef.current) return;
      const newWidth = containerRef.current.clientWidth;
      const newHeight = containerRef.current.clientHeight;

      if (newWidth === 0 || newHeight === 0) return;

      rendererRef.current.setSize(newWidth, newHeight);

      const viewer = viewerRef.current as {
        camera?: THREE.PerspectiveCamera;
      } | null;
      if (viewer?.camera) {
        viewer.camera.aspect = newWidth / newHeight;
        viewer.camera.updateProjectionMatrix();
      }
    };

    const timeouts = [50, 100, 200, 300].map((delay) =>
      setTimeout(handleResize, delay),
    );

    if (isExpanded) {
      window.addEventListener("resize", handleResize);
    }

    return () => {
      timeouts.forEach(clearTimeout);
      window.removeEventListener("resize", handleResize);
    };
  }, [isExpanded]);

  // Close on escape
  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      if (e.key === "Escape" && isExpanded) {
        setIsExpanded(false);
      }
    };

    document.addEventListener("keydown", handleKeyDown);
    return () => document.removeEventListener("keydown", handleKeyDown);
  }, [isExpanded]);

  // Prevent body scroll when modal is open
  useEffect(() => {
    if (isExpanded) {
      document.body.style.overflow = "hidden";
    } else {
      document.body.style.overflow = "";
    }
    return () => {
      document.body.style.overflow = "";
    };
  }, [isExpanded]);

  return (
    <>
      {/* Backdrop when expanded */}
      <AnimatePresence>
        {isExpanded && (
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            transition={{ duration: 0.2 }}
            className="fixed inset-0 z-40 bg-black/60 backdrop-blur-sm"
            onClick={handleCollapse}
          />
        )}
      </AnimatePresence>

      <motion.div
        initial={{ opacity: 0, scale: 0.95 }}
        animate={{ opacity: 1, scale: 1 }}
        transition={{ duration: 0.5 }}
        className="w-full"
      >
        <div
          ref={sceneContainerRef}
          className={`scene-container overflow-hidden rounded-2xl border border-[var(--border)] ${
            isExpanded ? "expanded z-50" : "relative w-full aspect-[16/10]"
          }`}
        >
          <div ref={containerRef} className="absolute inset-0" />

          {/* Loading overlay */}
          {isLoading && (
            <motion.div
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              className={`absolute inset-0 flex flex-col items-center justify-center backdrop-blur-sm z-10 ${
                isExpanded ? "bg-[#1a1a1a]/90" : "bg-[var(--surface)]/90"
              }`}
            >
              <p className={`text-lg font-medium mb-2 ${isExpanded ? "text-white" : ""}`}>
                Loading 3D Scene
              </p>
              <div
                className={`w-48 h-1.5 rounded-full overflow-hidden ${
                  isExpanded ? "bg-white/10" : "bg-[var(--surface-elevated)]"
                }`}
              >
                <motion.div
                  className={`h-full rounded-full ${
                    isExpanded ? "bg-white" : "bg-[var(--accent)]"
                  }`}
                  initial={{ width: 0 }}
                  animate={{ width: `${loadProgress}%` }}
                  transition={{ duration: 0.3 }}
                />
              </div>
              <p
                className={`text-sm mt-2 ${
                  isExpanded ? "text-white/60" : "text-[var(--text-muted)]"
                }`}
              >
                {loadProgress}%
              </p>
            </motion.div>
          )}

          {/* Error overlay */}
          {error && (
            <motion.div
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              className={`absolute inset-0 flex flex-col items-center justify-center backdrop-blur-sm z-10 ${
                isExpanded ? "bg-[#1a1a1a]/90" : "bg-[var(--surface)]/90"
              }`}
            >
              <p className="text-lg font-medium mb-2 text-red-400">Error</p>
              <p
                className={`text-sm text-center max-w-sm ${
                  isExpanded ? "text-white/60" : "text-[var(--text-muted)]"
                }`}
              >
                {error}
              </p>
            </motion.div>
          )}

          {/* Controls overlay */}
          {!isLoading && !error && (
            <>
              {/* Top right controls */}
              <div className={`absolute z-20 flex gap-2 ${isExpanded ? "top-4 right-4" : "top-4 right-4"}`}>
                {isExpanded && (
                  <button
                    type="button"
                    onClick={handleCollapse}
                    className="w-8 h-8 rounded-full bg-white/10 hover:bg-white/20 backdrop-blur-sm border border-white/20 flex items-center justify-center transition-all cursor-pointer"
                    title="Close"
                  >
                    <X className="w-4 h-4 text-white" />
                  </button>
                )}

                <button
                  type="button"
                  onClick={handleReset}
                  className={`flex items-center justify-center transition-all cursor-pointer ${
                    isExpanded
                      ? "w-8 h-8 rounded-full bg-white/10 hover:bg-white/20 backdrop-blur-sm border border-white/20"
                      : "p-2.5 rounded-xl glass hover:bg-white/10"
                  }`}
                  title="Reset view"
                >
                  <RotateCcw className={isExpanded ? "w-4 h-4 text-white" : "w-4 h-4"} />
                </button>

                {!isExpanded && (
                  <button
                    type="button"
                    onClick={handleExpand}
                    className="p-2.5 rounded-xl glass hover:bg-white/10 transition-colors cursor-pointer"
                    title="Expand"
                  >
                    <Maximize2 className="w-4 h-4" />
                  </button>
                )}

                <button
                  type="button"
                  onClick={handleDownload}
                  className={`flex items-center justify-center transition-all cursor-pointer ${
                    isExpanded
                      ? "w-8 h-8 rounded-full bg-white/10 hover:bg-white/20 backdrop-blur-sm border border-white/20"
                      : "p-2.5 rounded-xl glass hover:bg-white/10"
                  }`}
                  title="Download PLY"
                >
                  <Download className={isExpanded ? "w-4 h-4 text-white" : "w-4 h-4"} />
                </button>
              </div>

              {/* Bottom left instructions */}
              <div className={`absolute z-20 ${isExpanded ? "bottom-6 left-6" : "bottom-4 left-4"}`}>
                <div
                  className={`rounded-xl px-4 py-3 flex items-center gap-3 text-xs ${
                    isExpanded
                      ? "text-white/60 bg-white/10 backdrop-blur-sm border border-white/20"
                      : "glass text-[var(--text-muted)]"
                  }`}
                >
                  <span className="flex items-center gap-1.5">
                    <MousePointer2 className="w-3.5 h-3.5" />
                    Drag to rotate
                  </span>
                  <span className="flex items-center gap-1.5">
                    <Move className="w-3.5 h-3.5" />
                    Scroll to fly through
                  </span>
                </div>
              </div>
            </>
          )}
        </div>
      </motion.div>
    </>
  );
}
