'use client';

import { useRef, useState } from 'react';
import { useRouter } from 'next/navigation';
import { useMutation, useQueryClient } from '@tanstack/react-query';
import { Button } from '@/components/ui/button';
import { uploadSaree, generateViews } from '@/lib/api';
import { Upload, Loader2 } from 'lucide-react';
import { motion, AnimatePresence } from 'framer-motion';
import { toast } from 'sonner';

type UploadState = 'idle' | 'uploading' | 'generating' | 'error';

export function UploadButton() {
    const [uploadState, setUploadState] = useState<UploadState>('idle');
    const [progress, setProgress] = useState(0);
    const fileInputRef = useRef<HTMLInputElement>(null);
    const router = useRouter();
    const queryClient = useQueryClient();

    const { mutate: handleUploadFlow, isPending } = useMutation({
        mutationFn: async (file: File) => {
            // Step 1: Upload
            setUploadState('uploading');
            setProgress(0);

            // Simulate progress (since fetch doesn't give us upload progress easily)
            const progressInterval = setInterval(() => {
                setProgress((prev) => {
                    if (prev >= 90) return prev;
                    return prev + 10;
                });
            }, 100);

            try {
                const uploadResult = await uploadSaree(file);
                clearInterval(progressInterval);
                setProgress(100);

                // Step 2: Generate
                setUploadState('generating');
                await generateViews(uploadResult.saree_id, 'standard');

                return uploadResult;
            } catch (error) {
                clearInterval(progressInterval);
                throw error;
            }
        },
        onSuccess: (data) => {
            toast.success('Saree uploaded and generation started');
            setUploadState('idle');

            // Invalidate gallery to show new item
            queryClient.invalidateQueries({ queryKey: ['gallery'] });

            router.push(`/gallery/${data.saree_id}`);
        },
        onError: (error) => {
            setUploadState('error');
            const msg = error instanceof Error ? error.message : 'Upload failed';
            toast.error(msg);

            // Reset to idle after delay
            setTimeout(() => {
                setUploadState('idle');
            }, 3000);
        },
    });

    const handleClick = () => {
        fileInputRef.current?.click();
    };

    const handleFileChange = (event: React.ChangeEvent<HTMLInputElement>) => {
        const file = event.target.files?.[0];
        if (!file) return;

        // Reset file input
        event.target.value = '';

        handleUploadFlow(file);
    };

    const getButtonContent = () => {
        switch (uploadState) {
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
                disabled={isPending}
                variant={uploadState === 'error' ? 'destructive' : 'default'}
                data-testid="upload-button"
            >
                <AnimatePresence mode="wait">
                    <motion.span
                        key={uploadState}
                        initial={{ opacity: 0, y: 5 }}
                        animate={{ opacity: 1, y: 0 }}
                        exit={{ opacity: 0, y: -5 }}
                        className="flex items-center gap-2"
                    >
                        {getButtonContent()}
                    </motion.span>
                </AnimatePresence>
            </Button>
        </>
    );
}
