'use client';

import { useState, useEffect, useCallback } from 'react';
import Link from 'next/link';
import { Button } from '@/components/ui/button';
import { StatusBadge, type Status } from '@/components/StatusBadge';
import { ArtifactTabs } from '@/components/ArtifactTabs';
import { GenerationCard } from '@/components/GenerationCard';
import { GenerateMoreModal } from '@/components/GenerateMoreModal';
import { getSareeDetails, getJobStatus, type SareeDetails, type JobStatus } from '@/lib/api';
import { ArrowLeft, Wand2, Calendar, Hash, Loader2 } from 'lucide-react';

interface FolderContentProps {
    sareeId: string;
}

export function FolderContent({ sareeId }: FolderContentProps) {
    const [sareeDetails, setSareeDetails] = useState<SareeDetails | null>(null);
    const [loading, setLoading] = useState(true);
    const [error, setError] = useState<string | null>(null);
    const [modalOpen, setModalOpen] = useState(false);
    const [isPolling, setIsPolling] = useState(false);

    // Fetch saree details
    const fetchDetails = useCallback(async () => {
        try {
            const data = await getSareeDetails(sareeId);
            setSareeDetails(data);
            setError(null);
            return data;
        } catch (err) {
            setError(err instanceof Error ? err.message : 'Failed to load saree details');
            return null;
        }
    }, [sareeId]);

    // Initial load
    useEffect(() => {
        let isMounted = true;

        async function loadData() {
            setLoading(true);
            const data = await fetchDetails();
            if (isMounted) {
                setLoading(false);
                // Check if any generation is running and start polling
                if (data && hasRunningJob(data)) {
                    setIsPolling(true);
                }
            }
        }

        loadData();

        return () => {
            isMounted = false;
        };
    }, [fetchDetails]);

    // Polling for running jobs
    useEffect(() => {
        if (!isPolling) return;

        const interval = setInterval(async () => {
            const data = await fetchDetails();
            if (data && !hasRunningJob(data)) {
                setIsPolling(false);
            }
        }, 2000);

        return () => clearInterval(interval);
    }, [isPolling, fetchDetails]);

    // Check if there's a running job
    function hasRunningJob(details: SareeDetails): boolean {
        return details.generations.some(
            (gen) => gen.status === 'running' || gen.status === 'queued'
        );
    }

    // Format date
    const formattedDate = sareeDetails
        ? new Date(sareeDetails.created_at).toLocaleDateString('en-US', {
            month: 'long',
            day: 'numeric',
            year: 'numeric',
        })
        : '';

    // Short ID
    const shortId = sareeId.substring(0, 8);

    // Get latest status from most recent generation
    const latestStatus = sareeDetails?.generations[sareeDetails.generations.length - 1]?.status;

    if (loading) {
        return null; // Suspense boundary handles loading
    }

    if (error || !sareeDetails) {
        return (
            <div className="text-center py-16">
                <p className="text-destructive mb-4">{error || 'Saree not found'}</p>
                <Button asChild variant="outline">
                    <Link href="/gallery">
                        <ArrowLeft className="h-4 w-4 mr-2" />
                        Back to Gallery
                    </Link>
                </Button>
            </div>
        );
    }

    return (
        <div className="space-y-8">
            {/* Header */}
            <div className="flex flex-col gap-4 md:flex-row md:items-start md:justify-between">
                <div className="space-y-2">
                    <div className="flex items-center gap-2">
                        <Button asChild variant="ghost" size="sm" className="mr-2">
                            <Link href="/gallery">
                                <ArrowLeft className="h-4 w-4" />
                            </Link>
                        </Button>
                        <div className="flex items-center gap-2 text-lg font-mono text-muted-foreground">
                            <Hash className="h-4 w-4" />
                            {shortId}
                        </div>
                        {latestStatus && <StatusBadge status={latestStatus as Status} />}
                    </div>
                    <div className="flex items-center gap-1.5 text-sm text-muted-foreground ml-11">
                        <Calendar className="h-3.5 w-3.5" />
                        {formattedDate}
                    </div>
                </div>

                <Button onClick={() => setModalOpen(true)} data-testid="generate-more-button">
                    <Wand2 className="h-4 w-4 mr-2" />
                    Generate More Views
                </Button>
            </div>

            {/* Polling indicator */}
            {isPolling && (
                <div className="flex items-center gap-2 text-sm text-muted-foreground bg-muted px-4 py-2 rounded-lg">
                    <Loader2 className="h-4 w-4 animate-spin" />
                    Generating standard views... Updates will appear automatically.
                </div>
            )}

            {/* Artifacts Section */}
            <section>
                <h2 className="text-lg font-semibold mb-4">Saree Artifacts</h2>
                <ArtifactTabs sareeId={sareeId} artifacts={sareeDetails.artifacts} />
            </section>

            {/* Generations Section */}
            <section>
                <h2 className="text-lg font-semibold mb-4">Generations</h2>
                {sareeDetails.generations.length === 0 ? (
                    <div className="text-center py-8 text-muted-foreground bg-muted rounded-xl">
                        <p>No generations yet. Click "Generate More Views" to start.</p>
                    </div>
                ) : (
                    <div className="space-y-4">
                        {sareeDetails.generations.map((generation) => (
                            <GenerationCard
                                key={generation.generation_id}
                                sareeId={sareeId}
                                generation={generation}
                            />
                        ))}
                    </div>
                )}
            </section>

            {/* Generate More Modal */}
            <GenerateMoreModal
                sareeId={sareeId}
                isOpen={modalOpen}
                onClose={() => setModalOpen(false)}
                hasFailures={sareeDetails.has_failures}
                onGenerationStarted={() => {
                    setIsPolling(true);
                    fetchDetails();
                }}
            />
        </div>
    );
}
