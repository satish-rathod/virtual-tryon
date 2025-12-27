'use client';

import { useRef, useState } from 'react';
import { useRouter } from 'next/navigation';
import { Button } from '@/components/ui/button';
import { uploadSaree, generateViews } from '@/lib/api';
import { Upload, Loader2 } from 'lucide-react';
import { motion, AnimatePresence } from 'framer-motion';

type UploadState = 'idle' | 'uploading' | 'generating' | 'error';

export function UploadButton() {
    const [state, setState] = useState<UploadState>('idle');
    const [progress, setProgress] = useState(0);
    const [errorMessage, setErrorMessage] = useState<string>('');
    const fileInputRef = useRef<HTMLInputElement>(null);
    const router = useRouter();

    const handleClick = () => {
        fileInputRef.current?.click();
    };

    const handleFileChange = async (event: React.ChangeEvent<HTMLInputElement>) => {
        const file = event.target.files?.[0];
        if (!file) return;

        // Reset file input so the same file can be selected again
        event.target.value = '';

        try {
            // Step 1: Upload the file
            setState('uploading');
            setProgress(0);
            setErrorMessage('');

            // Simulate progress for upload (actual progress would need XMLHttpRequest)
            const progressInterval = setInterval(() => {
                setProgress((prev) => Math.min(prev + 10, 90));
            }, 100);

            const uploadResult = await uploadSaree(file);

            clearInterval(progressInterval);
            setProgress(100);

            // Step 2: Trigger initial generation with standard mode
            setState('generating');
            setProgress(0);

            // Auto-trigger generation with mode: standard
            await generateViews(uploadResult.saree_id, 'standard');

            // Step 3: Navigate to folder view
            router.push(`/gallery/${uploadResult.saree_id}`);
        } catch (error) {
            setState('error');
            setErrorMessage(error instanceof Error ? error.message : 'Upload failed');

            // Reset to idle after 3 seconds
            setTimeout(() => {
                setState('idle');
                setErrorMessage('');
            }, 3000);
        }
    };

    const getButtonContent = () => {
        switch (state) {
            case 'uploading':
                return (
                    <>
                        <Loader2 className="h-4 w-4 animate-spin" />
                        Uploading... {progress}%
                    </>
                );
            case 'generating':
                return (
                    <>
                        <Loader2 className="h-4 w-4 animate-spin" />
                        Generating standard views...
                    </>
                );
            case 'error':
                return 'Upload failed';
            default:
                return (
                    <>
                        <Upload className="h-4 w-4" />
                        Upload Saree
                    </>
                );
        }
    };

    return (
        <>
            <input
                ref={fileInputRef}
                type="file"
                accept="image/*"
                className="hidden"
                onChange={handleFileChange}
                data-testid="file-input"
            />
            <Button
                onClick={handleClick}
                disabled={state !== 'idle' && state !== 'error'}
                variant={state === 'error' ? 'destructive' : 'default'}
                data-testid="upload-button"
            >
                <AnimatePresence mode="wait">
                    <motion.span
                        key={state}
                        initial={{ opacity: 0, y: 5 }}
                        animate={{ opacity: 1, y: 0 }}
                        exit={{ opacity: 0, y: -5 }}
                        className="flex items-center gap-2"
                    >
                        {getButtonContent()}
                    </motion.span>
                </AnimatePresence>
            </Button>

            {/* Error message tooltip */}
            <AnimatePresence>
                {errorMessage && (
                    <motion.div
                        initial={{ opacity: 0, y: 10 }}
                        animate={{ opacity: 1, y: 0 }}
                        exit={{ opacity: 0, y: 10 }}
                        className="absolute top-full mt-2 right-0 bg-destructive text-destructive-foreground text-sm px-3 py-1.5 rounded-md shadow-lg"
                    >
                        {errorMessage}
                    </motion.div>
                )}
            </AnimatePresence>
        </>
    );
}
