'use client';

import { useRouter } from 'next/navigation';
import { useMutation, useQueryClient } from '@tanstack/react-query';
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
import { toast } from 'sonner';

interface GenerateMoreModalProps {
    sareeId: string;
    isOpen: boolean;
    onClose: () => void;
    hasFailures: boolean;
    onGenerationStarted?: () => void;
}

export function GenerateMoreModal({
    sareeId,
    isOpen,
    onClose,
    hasFailures,
    onGenerationStarted,
}: GenerateMoreModalProps) {
    const router = useRouter();
    const queryClient = useQueryClient();

    const { mutate: generate, isPending } = useMutation({
        mutationFn: ({ mode }: { mode: 'extend' | 'retry_failed' }) =>
            generateViews(sareeId, mode),
        onSuccess: (_data, variables) => {
            const action = variables.mode === 'extend' ? 'Generation' : 'Retry';
            toast.success(`${action} started successfully`);

            // Invalidate gallery query to reflect new status
            queryClient.invalidateQueries({ queryKey: ['gallery'] });

            if (onGenerationStarted) {
                onGenerationStarted();
            }

            router.refresh();
            onClose();
        },
        onError: (error) => {
            toast.error(error instanceof Error ? error.message : 'Operation failed');
        },
    });

    const handleGenerateExtend = () => {
        generate({ mode: 'extend' });
    };

    const handleRetryFailed = () => {
        if (!hasFailures) return;
        generate({ mode: 'retry_failed' });
    };

    const handleOpenChange = (open: boolean) => {
        if (!open && !isPending) {
            onClose();
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
                        disabled={isPending}
                        className="w-full justify-start h-auto py-4"
                        variant="outline"
                        data-testid="generate-extend-button"
                    >
                        <div className="flex items-start gap-3">
                            {isPending ? (
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
                        disabled={isPending || !hasFailures}
                        className="w-full justify-start h-auto py-4"
                        variant="outline"
                        data-testid="retry-failed-button"
                        title={!hasFailures ? 'No failed views to retry' : undefined}
                    >
                        <div className="flex items-start gap-3">
                            {isPending ? (
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

                <DialogFooter>
                    <Button
                        variant="ghost"
                        onClick={onClose}
                        disabled={isPending}
                    >
                        Cancel
                    </Button>
                </DialogFooter>
            </DialogContent>
        </Dialog>
    );
}
