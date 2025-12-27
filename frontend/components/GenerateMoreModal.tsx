'use client';

import { useState } from 'react';
import { useRouter } from 'next/navigation';
import {
    Dialog,
    DialogContent,
    DialogDescription,
    DialogHeader,
    DialogTitle,
    DialogFooter,
} from '@/components/ui/dialog';
import { Button } from '@/components/ui/button';
import { generateViews } from '@/lib/api';
import { Wand2, RotateCcw, Loader2 } from 'lucide-react';

interface GenerateMoreModalProps {
    sareeId: string;
    isOpen: boolean;
    onClose: () => void;
    hasFailures: boolean;
}

type GenerateState = 'idle' | 'generating' | 'success' | 'error';

export function GenerateMoreModal({
    sareeId,
    isOpen,
    onClose,
    hasFailures,
}: GenerateMoreModalProps) {
    const [state, setState] = useState<GenerateState>('idle');
    const [errorMessage, setErrorMessage] = useState('');
    const router = useRouter();

    const handleGenerateExtend = async () => {
        try {
            setState('generating');
            setErrorMessage('');

            await generateViews(sareeId, 'extend');

            setState('success');
            router.refresh();

            // Close modal after short delay
            setTimeout(() => {
                onClose();
                setState('idle');
            }, 1000);
        } catch (error) {
            setState('error');
            setErrorMessage(error instanceof Error ? error.message : 'Generation failed');
        }
    };

    const handleRetryFailed = async () => {
        if (!hasFailures) return;

        try {
            setState('generating');
            setErrorMessage('');

            await generateViews(sareeId, 'retry_failed');

            setState('success');
            router.refresh();

            // Close modal after short delay
            setTimeout(() => {
                onClose();
                setState('idle');
            }, 1000);
        } catch (error) {
            setState('error');
            setErrorMessage(error instanceof Error ? error.message : 'Retry failed');
        }
    };

    const handleOpenChange = (open: boolean) => {
        if (!open && state !== 'generating') {
            onClose();
            setState('idle');
            setErrorMessage('');
        }
    };

    return (
        <Dialog open={isOpen} onOpenChange={handleOpenChange}>
            <DialogContent className="sm:max-w-md">
                <DialogHeader>
                    <DialogTitle>Generate additional views</DialogTitle>
                    <DialogDescription>
                        Additional views are generated using predefined model angles.
                    </DialogDescription>
                </DialogHeader>

                <div className="space-y-4 py-4">
                    {/* Generate remaining views option */}
                    <Button
                        onClick={handleGenerateExtend}
                        disabled={state === 'generating'}
                        className="w-full justify-start h-auto py-4"
                        variant="outline"
                        data-testid="generate-extend-button"
                    >
                        <div className="flex items-start gap-3">
                            {state === 'generating' ? (
                                <Loader2 className="h-5 w-5 animate-spin shrink-0 mt-0.5" />
                            ) : (
                                <Wand2 className="h-5 w-5 shrink-0 mt-0.5" />
                            )}
                            <div className="text-left">
                                <div className="font-medium">Generate remaining views</div>
                                <div className="text-sm text-muted-foreground font-normal">
                                    Create additional model poses for this saree
                                </div>
                            </div>
                        </div>
                    </Button>

                    {/* Retry failed views option */}
                    <Button
                        onClick={handleRetryFailed}
                        disabled={state === 'generating' || !hasFailures}
                        className="w-full justify-start h-auto py-4"
                        variant="outline"
                        data-testid="retry-failed-button"
                        title={!hasFailures ? 'No failed views to retry' : undefined}
                    >
                        <div className="flex items-start gap-3">
                            {state === 'generating' ? (
                                <Loader2 className="h-5 w-5 animate-spin shrink-0 mt-0.5" />
                            ) : (
                                <RotateCcw className="h-5 w-5 shrink-0 mt-0.5" />
                            )}
                            <div className="text-left">
                                <div className="font-medium">Retry failed views</div>
                                <div className="text-sm text-muted-foreground font-normal">
                                    {hasFailures
                                        ? 'Re-attempt generation for views that failed'
                                        : 'No failed views to retry'}
                                </div>
                            </div>
                        </div>
                    </Button>
                </div>

                {/* Error message */}
                {errorMessage && (
                    <div className="text-sm text-destructive text-center py-2">
                        {errorMessage}
                    </div>
                )}

                {/* Success message */}
                {state === 'success' && (
                    <div className="text-sm text-emerald-600 text-center py-2">
                        Generation started successfully!
                    </div>
                )}

                <DialogFooter>
                    <Button
                        variant="ghost"
                        onClick={onClose}
                        disabled={state === 'generating'}
                    >
                        Cancel
                    </Button>
                </DialogFooter>
            </DialogContent>
        </Dialog>
    );
}
