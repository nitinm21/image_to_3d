"use client";

import { useState, useCallback } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { ArrowLeft } from "lucide-react";
import ImageUpload from "@/components/ImageUpload";
import GaussianViewer from "@/components/GaussianViewer";
import ProcessingStatus, { ProcessingStage } from "@/components/ProcessingStatus";

export default function Home() {
  const [stage, setStage] = useState<ProcessingStage>("idle");
  const [progress, setProgress] = useState(0);
  const [error, setError] = useState<string | undefined>();
  const [imagePreview, setImagePreview] = useState<string | undefined>();
  const [plyUrl, setPlyUrl] = useState<string | null>(null);

  const handleImageSelect = useCallback(async (file: File) => {
    // Create preview
    const reader = new FileReader();
    reader.onload = () => {
      setImagePreview(reader.result as string);
    };
    reader.readAsDataURL(file);

    // Reset state
    setStage("uploading");
    setProgress(0);
    setError(undefined);
    setPlyUrl(null);

    try {
      // Simulate upload progress
      setProgress(30);

      // Create FormData and send to API
      const formData = new FormData();
      formData.append("image", file);

      setProgress(50);
      setStage("processing");

      const response = await fetch("/api/process", {
        method: "POST",
        body: formData,
      });

      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.error || "Processing failed");
      }

      const data = await response.json();

      if (!data.success) {
        throw new Error(data.error || "Processing failed");
      }

      // Set the PLY URL (this is a data URL for local storage)
      setPlyUrl(data.plyUrl);
      setStage("complete");
      setProgress(100);
    } catch (err) {
      console.error("Processing error:", err);
      setError(err instanceof Error ? err.message : "An error occurred");
      setStage("error");
    }
  }, []);

  const handleReset = () => {
    setStage("idle");
    setProgress(0);
    setError(undefined);
    setImagePreview(undefined);
    setPlyUrl(null);
  };

  return (
    <main className="min-h-screen flex flex-col">

      {/* Main content */}
      <div className="flex-1 flex flex-col">
        <div className="max-w-5xl w-full mx-auto px-6 py-16 flex-1">
          <AnimatePresence mode="wait">
            {/* Initial state - show upload */}
            {stage === "idle" && (
              <motion.div
                key="upload"
                initial={{ opacity: 0, y: 24 }}
                animate={{ opacity: 1, y: 0 }}
                exit={{ opacity: 0, y: -16 }}
                transition={{ duration: 0.4, ease: [0.22, 1, 0.36, 1] }}
                className="max-w-xl mx-auto"
              >
                {/* Hero text */}
                <div className="text-center mb-10">
                  <motion.h1
                    initial={{ opacity: 0, y: 16 }}
                    animate={{ opacity: 1, y: 0 }}
                    transition={{ delay: 0.1, duration: 0.5 }}
                    className="text-display text-4xl md:text-5xl mb-4 text-[var(--text-primary)]"
                    style={{ fontFamily: "var(--font-syne)" }}
                  >
                    Image to 3D
                  </motion.h1>
                  <motion.p
                    initial={{ opacity: 0, y: 16 }}
                    animate={{ opacity: 1, y: 0 }}
                    transition={{ delay: 0.2, duration: 0.5 }}
                    className="text-[var(--text-muted)] text-base leading-relaxed max-w-md mx-auto"
                  >
                    Transform any photograph into an interactive 3D scene.
                    Powered by neural radiance fields and Gaussian splatting.
                  </motion.p>
                </div>

                {/* Upload component */}
                <motion.div
                  initial={{ opacity: 0, y: 16 }}
                  animate={{ opacity: 1, y: 0 }}
                  transition={{ delay: 0.3, duration: 0.5 }}
                >
                  <ImageUpload onImageSelect={handleImageSelect} />
                </motion.div>
              </motion.div>
            )}

            {/* Processing state */}
            {(stage === "uploading" || stage === "processing") && (
              <motion.div
                key="processing"
                initial={{ opacity: 0, y: 24 }}
                animate={{ opacity: 1, y: 0 }}
                exit={{ opacity: 0, y: -16 }}
                transition={{ duration: 0.4, ease: [0.22, 1, 0.36, 1] }}
              >
                <ProcessingStatus
                  stage={stage}
                  progress={progress}
                  imagePreview={imagePreview}
                />
              </motion.div>
            )}

            {/* Error state */}
            {stage === "error" && (
              <motion.div
                key="error"
                initial={{ opacity: 0, y: 24 }}
                animate={{ opacity: 1, y: 0 }}
                exit={{ opacity: 0, y: -16 }}
                transition={{ duration: 0.4, ease: [0.22, 1, 0.36, 1] }}
              >
                <ProcessingStatus
                  stage={stage}
                  error={error}
                  imagePreview={imagePreview}
                />

                <div className="flex justify-center mt-8">
                  <button
                    onClick={handleReset}
                    className="btn-secondary"
                  >
                    <ArrowLeft className="w-4 h-4" />
                    Try again
                  </button>
                </div>
              </motion.div>
            )}

            {/* Complete state - show 3D viewer */}
            {stage === "complete" && plyUrl && (
              <motion.div
                key="viewer"
                initial={{ opacity: 0, y: 24 }}
                animate={{ opacity: 1, y: 0 }}
                exit={{ opacity: 0, y: -16 }}
                transition={{ duration: 0.4, ease: [0.22, 1, 0.36, 1] }}
              >
                <div className="flex items-center justify-between mb-6">
                  <button
                    onClick={handleReset}
                    className="btn-secondary"
                  >
                    <ArrowLeft className="w-4 h-4" />
                    New image
                  </button>

                  <div className="flex items-center gap-2 px-3 py-1.5 rounded-full bg-[var(--success-muted)] border border-[var(--success)]/20">
                    <div className="status-dot success" />
                    <span className="text-xs text-[var(--success)] font-medium">
                      3D scene ready
                    </span>
                  </div>
                </div>

                <GaussianViewer plyUrl={plyUrl} />
              </motion.div>
            )}
          </AnimatePresence>
        </div>
      </div>

    </main>
  );
}
