'use client';

import { Badge } from '@/components/ui/badge';
import { cn } from '@/lib/utils';
import { Loader2 } from 'lucide-react';

export type Status = 'success' | 'partial' | 'failed' | 'running' | 'queued' | 'pending';

interface StatusBadgeProps {
    status: Status;
    className?: string;
}

const statusConfig: Record<Status, { label: string; variant: 'success' | 'warning' | 'destructive' | 'info' | 'secondary' }> = {
    success: { label: 'Success', variant: 'success' },
    partial: { label: 'Partial', variant: 'warning' },
    failed: { label: 'Failed', variant: 'destructive' },
    running: { label: 'Running', variant: 'info' },
    queued: { label: 'Queued', variant: 'secondary' },
    pending: { label: 'Pending', variant: 'secondary' },
};

export function StatusBadge({ status, className }: StatusBadgeProps) {
    const config = statusConfig[status];

    return (
        <Badge variant={config.variant} className={cn('gap-1', className)}>
            {status === 'running' && (
                <Loader2 className="h-3 w-3 animate-spin" />
            )}
            {config.label}
        </Badge>
    );
}
