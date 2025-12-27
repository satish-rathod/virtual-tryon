'use client';

import { useEffect, useCallback } from 'react';
import Image from 'next/image';
import { X, Clock } from 'lucide-react';
import { motion, AnimatePresence } from 'framer-motion';

interface ImageLightboxProps {
    src: string;
    alt: string;
    title: string;
    timestamp?: string;
    onClose: () => void;
}

export function ImageLightbox({
    src,
    alt,
    title,
    timestamp,
    onClose,
}: ImageLightboxProps) {
    // Handle escape key
    const handleKeyDown = useCallback(
        (event: KeyboardEvent) => {
            if (event.key === 'Escape') {
                onClose();
            }
        },
        [onClose]
    );

    useEffect(() => {
        document.addEventListener('keydown', handleKeyDown);
        // Prevent body scroll when lightbox is open
        document.body.style.overflow = 'hidden';

        return () => {
            document.removeEventListener('keydown', handleKeyDown);
            document.body.style.overflow = '';
        };
    }, [handleKeyDown]);

    return (
        <AnimatePresence>
            <motion.div
                initial={{ opacity: 0 }}
                animate={{ opacity: 1 }}
                exit={{ opacity: 0 }}
                className="fixed inset-0 z-50 flex items-center justify-center"
                role="dialog"
                aria-modal="true"
                aria-labelledby="lightbox-title"
            >
                {/* Backdrop */}
                <motion.div
                    initial={{ opacity: 0 }}
                    animate={{ opacity: 1 }}
                    exit={{ opacity: 0 }}
                    className="absolute inset-0 bg-black/90"
                    onClick={onClose}
                />

                {/* Content */}
                <motion.div
                    initial={{ scale: 0.95, opacity: 0 }}
                    animate={{ scale: 1, opacity: 1 }}
                    exit={{ scale: 0.95, opacity: 0 }}
                    className="relative z-10 w-full max-w-4xl max-h-[90vh] mx-4"
                >
                    {/* Close button */}
                    <button
                        onClick={onClose}
                        className="absolute -top-12 right-0 text-white/70 hover:text-white p-2 rounded-full hover:bg-white/10 transition-colors"
                        aria-label="Close lightbox"
                    >
                        <X className="h-6 w-6" />
                    </button>

                    {/* Image container */}
                    <div className="relative aspect-[3/4] bg-black/50 rounded-xl overflow-hidden">
                        <Image
                            src={src}
                            alt={alt}
                            fill
                            className="object-contain"
                            sizes="(max-width: 768px) 100vw, 800px"
                            priority
                        />
                    </div>

                    {/* Metadata footer */}
                    <div className="mt-4 text-center text-white">
                        <h2 id="lightbox-title" className="text-lg font-semibold">
                            {title}
                        </h2>
                        {timestamp && (
                            <div className="flex items-center justify-center gap-1.5 mt-1 text-sm text-white/70">
                                <Clock className="h-3.5 w-3.5" />
                                {timestamp}
                            </div>
                        )}
                    </div>
                </motion.div>
            </motion.div>
        </AnimatePresence>
    );
}
