"use client";

import { motion } from "framer-motion";
import { Loader2, Check, AlertCircle } from "lucide-react";

export type ProcessingStage = "idle" | "uploading" | "processing" | "complete" | "error";

interface ProcessingStatusProps {
  stage: ProcessingStage;
  progress?: number;
  error?: string;
  imagePreview?: string;
}

export default function ProcessingStatus({
  stage,
  progress = 0,
  error,
  imagePreview,
}: ProcessingStatusProps) {
  if (stage === "idle") return null;

  const stages = [
    { key: "uploading", label: "Uploading image" },
    { key: "processing", label: "Generating 3D scene" },
  ];

  const currentIndex = stages.findIndex((s) => s.key === stage);

  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      className="w-full max-w-lg mx-auto"
    >
      <div className="bg-[var(--surface)] border border-[var(--border-subtle)] rounded-2xl p-6">
        {/* Image preview */}
        {imagePreview && (
          <div className="mb-6">
            <div className="relative w-full aspect-video rounded-xl overflow-hidden border border-[var(--border)]">
              <img
                src={imagePreview}
                alt="Processing"
                className="absolute inset-0 w-full h-full object-cover"
              />
              {stage === "processing" && (
                <div className="absolute inset-0 bg-black/50 backdrop-blur-sm flex items-center justify-center">
                  <div className="flex items-center gap-3 px-4 py-2 rounded-full bg-black/30 backdrop-blur-sm">
                    <Loader2 className="w-4 h-4 text-[var(--accent)] animate-spin" />
                    <span className="text-sm font-medium text-white">Creating 3D scene...</span>
                  </div>
                </div>
              )}
            </div>
          </div>
        )}

        {/* Error state */}
        {stage === "error" && (
          <div className="flex flex-col items-center gap-4 py-4">
            <div className="w-12 h-12 rounded-full bg-[var(--error-muted)] flex items-center justify-center">
              <AlertCircle className="w-5 h-5 text-[var(--error)]" />
            </div>
            <div className="text-center">
              <p className="font-medium text-[var(--error)] mb-1">Processing Failed</p>
              <p className="text-sm text-[var(--text-muted)] max-w-sm">
                {error || "An error occurred while processing your image."}
              </p>
            </div>
          </div>
        )}

        {/* Progress stages */}
        {stage !== "error" && stage !== "complete" && (
          <div className="space-y-4">
            {stages.map((s, index) => {
              const isActive = s.key === stage;
              const isComplete = index < currentIndex;

              return (
                <div key={s.key} className="flex items-center gap-4">
                  {/* Status icon */}
                  <div
                    className={`w-8 h-8 rounded-full flex items-center justify-center flex-shrink-0 transition-colors ${
                      isComplete
                        ? "bg-[var(--success-muted)]"
                        : isActive
                        ? "bg-[var(--accent-subtle)]"
                        : "bg-[var(--surface-elevated)]"
                    }`}
                  >
                    {isComplete ? (
                      <Check className="w-4 h-4 text-[var(--success)]" />
                    ) : isActive ? (
                      <Loader2 className="w-4 h-4 text-[var(--accent)] animate-spin" />
                    ) : (
                      <div className="w-1.5 h-1.5 rounded-full bg-[var(--text-faint)]" />
                    )}
                  </div>

                  {/* Label and progress */}
                  <div className="flex-1">
                    <p
                      className={`text-sm font-medium transition-colors ${
                        isComplete
                          ? "text-[var(--success)]"
                          : isActive
                          ? "text-[var(--text-primary)]"
                          : "text-[var(--text-faint)]"
                      }`}
                    >
                      {s.label}
                    </p>
                    {isActive && (
                      <div className="mt-2.5">
                        <div className="h-1 rounded-full overflow-hidden bg-[var(--surface-elevated)]">
                          <motion.div
                            className="h-full rounded-full bg-[var(--accent)] progress-pulse"
                            initial={{ width: 0 }}
                            animate={{ width: `${Math.max(progress, 10)}%` }}
                            transition={{ duration: 0.3 }}
                          />
                        </div>
                      </div>
                    )}
                  </div>
                </div>
              );
            })}
          </div>
        )}

        {/* Complete state */}
        {stage === "complete" && (
          <div className="flex items-center gap-4 py-2">
            <div className="w-8 h-8 rounded-full bg-[var(--success-muted)] flex items-center justify-center">
              <Check className="w-4 h-4 text-[var(--success)]" />
            </div>
            <p className="text-sm font-medium text-[var(--success)]">
              3D scene generated successfully
            </p>
          </div>
        )}
      </div>
    </motion.div>
  );
}
