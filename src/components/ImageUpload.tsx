"use client";

import { useCallback, useState } from "react";
import { useDropzone, FileRejection } from "react-dropzone";
import { Upload, X, Image as ImageIcon } from "lucide-react";
import { motion, AnimatePresence } from "framer-motion";

const MAX_FILE_SIZE = 10 * 1024 * 1024; // 10MB limit

interface ImageUploadProps {
  onImageSelect: (file: File) => void;
  disabled?: boolean;
}

export default function ImageUpload({
  onImageSelect,
  disabled,
}: ImageUploadProps) {
  const [preview, setPreview] = useState<string | null>(null);
  const [fileName, setFileName] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);

  const onDrop = useCallback(
    (acceptedFiles: File[], fileRejections: FileRejection[]) => {
      setError(null);

      if (fileRejections.length > 0) {
        const rejection = fileRejections[0];
        const errorCode = rejection.errors[0]?.code;

        if (errorCode === "file-too-large") {
          const sizeMB = (rejection.file.size / (1024 * 1024)).toFixed(1);
          setError(`File too large (${sizeMB}MB). Maximum size is 10MB.`);
        } else if (errorCode === "file-invalid-type") {
          setError("Invalid file type. Please use PNG, JPG, or WEBP.");
        } else {
          setError(rejection.errors[0]?.message || "File could not be uploaded.");
        }
        return;
      }

      if (acceptedFiles.length > 0 && !disabled) {
        const file = acceptedFiles[0];
        setFileName(file.name);

        const reader = new FileReader();
        reader.onload = () => {
          setPreview(reader.result as string);
        };
        reader.readAsDataURL(file);
        onImageSelect(file);
      }
    },
    [onImageSelect, disabled],
  );

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: {
      "image/*": [".png", ".jpg", ".jpeg", ".webp"],
    },
    maxFiles: 1,
    maxSize: MAX_FILE_SIZE,
    disabled,
  });

  const clearPreview = (e: React.MouseEvent) => {
    e.stopPropagation();
    setPreview(null);
    setFileName(null);
    setError(null);
  };

  return (
    <motion.div
      initial={{ opacity: 0, y: 16 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.5, ease: [0.22, 1, 0.36, 1] }}
      className="w-full"
    >
      <div
        {...getRootProps()}
        className={`upload-portal relative min-h-[340px] flex flex-col justify-center cursor-pointer ${
          isDragActive ? "active" : ""
        } ${disabled ? "opacity-50 cursor-not-allowed" : ""}`}
      >
        <input {...getInputProps()} />

        {/* Grid overlay */}
        <div className="grid-overlay" />

        {/* Crosshair */}
        <div className="crosshair" />

        {/* Corner brackets */}
        <div className="corner-bracket tl" />
        <div className="corner-bracket tr" />
        <div className="corner-bracket bl" />
        <div className="corner-bracket br" />

        <AnimatePresence mode="wait">
          {preview ? (
            <motion.div
              key="preview"
              initial={{ opacity: 0, scale: 0.96 }}
              animate={{ opacity: 1, scale: 1 }}
              exit={{ opacity: 0, scale: 0.96 }}
              transition={{ duration: 0.3, ease: [0.22, 1, 0.36, 1] }}
              className="relative z-10 px-6"
            >
              <div className="relative w-full max-w-sm mx-auto">
                {/* Image container with glow effect */}
                <div className="relative">
                  <div className="absolute -inset-1 bg-gradient-to-b from-[var(--accent)]/20 to-transparent rounded-2xl blur-xl" />
                  <div className="relative aspect-[4/3] rounded-xl overflow-hidden border border-[var(--border)] bg-[var(--surface)]">
                    <img
                      src={preview}
                      alt="Preview"
                      className="absolute inset-0 w-full h-full object-cover"
                    />

                    {/* Gradient overlay at bottom */}
                    <div className="absolute inset-x-0 bottom-0 h-20 bg-gradient-to-t from-black/70 via-black/30 to-transparent" />

                    {/* File info */}
                    <div className="absolute bottom-0 left-0 right-0 p-4">
                      <div className="flex items-center gap-2.5">
                        <div className="w-8 h-8 rounded-lg bg-white/10 backdrop-blur-sm flex items-center justify-center">
                          <ImageIcon size={14} className="text-white" />
                        </div>
                        <div className="flex-1 min-w-0">
                          <p className="text-sm font-medium text-white truncate">
                            {fileName}
                          </p>
                          <p className="text-xs text-white/60">Ready to process</p>
                        </div>
                      </div>
                    </div>

                    {/* Clear button */}
                    {!disabled && (
                      <motion.button
                        initial={{ opacity: 0, scale: 0.8 }}
                        animate={{ opacity: 1, scale: 1 }}
                        transition={{ delay: 0.2 }}
                        onClick={clearPreview}
                        className="absolute top-3 right-3 w-8 h-8 rounded-full bg-black/50 backdrop-blur-sm hover:bg-black/70 transition-all flex items-center justify-center cursor-pointer group"
                      >
                        <X size={14} className="text-white/80 group-hover:text-white transition-colors" />
                      </motion.button>
                    )}
                  </div>
                </div>

                {/* Processing indicator */}
                <motion.div
                  initial={{ opacity: 0, y: 8 }}
                  animate={{ opacity: 1, y: 0 }}
                  transition={{ delay: 0.3 }}
                  className="mt-5 flex items-center justify-center gap-2"
                >
                  <div className="status-dot processing" />
                  <span className="text-sm text-[var(--text-muted)]">
                    Processing will begin automatically
                  </span>
                </motion.div>
              </div>
            </motion.div>
          ) : (
            <motion.div
              key="dropzone"
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              exit={{ opacity: 0 }}
              className="relative z-10 flex flex-col items-center gap-6 px-6 py-8"
            >
              {/* Icon */}
              <motion.div
                animate={isDragActive ? { scale: 1.08, y: -4 } : { scale: 1, y: 0 }}
                transition={{ duration: 0.3, ease: [0.22, 1, 0.36, 1] }}
              >
                <div className="relative">
                  {/* Glow effect behind icon */}
                  <motion.div
                    className="absolute -inset-4 rounded-2xl bg-[var(--accent)]/10 blur-xl"
                    animate={isDragActive ? { opacity: 1, scale: 1.2 } : { opacity: 0, scale: 1 }}
                    transition={{ duration: 0.3 }}
                  />
                  <div className="icon-box relative">
                    <Upload
                      size={22}
                      className={`transition-colors duration-300 ${
                        isDragActive ? "text-[var(--accent)]" : "text-[var(--text-secondary)]"
                      }`}
                    />
                  </div>
                </div>
              </motion.div>

              {/* Text content */}
              <div className="text-center">
                <motion.h3
                  className="text-lg font-semibold text-[var(--text-primary)]"
                  animate={isDragActive ? { scale: 1.02 } : { scale: 1 }}
                  transition={{ duration: 0.2 }}
                >
                  {isDragActive ? "Drop to begin" : "Drop image here"}
                </motion.h3>
                <p className="text-sm text-[var(--text-muted)] mt-1">
                  or click to browse
                </p>
              </div>

              {/* Error message */}
              <AnimatePresence>
                {error && (
                  <motion.div
                    initial={{ opacity: 0, y: 8, scale: 0.96 }}
                    animate={{ opacity: 1, y: 0, scale: 1 }}
                    exit={{ opacity: 0, y: -8, scale: 0.96 }}
                    className="px-4 py-2.5 rounded-lg bg-[var(--error-muted)] border border-[var(--error)]/20"
                  >
                    <p className="text-sm text-[var(--error)]">{error}</p>
                  </motion.div>
                )}
              </AnimatePresence>
            </motion.div>
          )}
        </AnimatePresence>
      </div>

    </motion.div>
  );
}

