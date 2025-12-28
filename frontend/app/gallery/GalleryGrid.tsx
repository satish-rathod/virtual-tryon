'use client';

import { useState } from 'react';
import { useQuery } from '@tanstack/react-query';
import { GalleryTile } from '@/components/GalleryTile';
import { GenerateMoreModal } from '@/components/GenerateMoreModal';
import { getGallery, type GalleryItem } from '@/lib/api';
import { Upload, FolderOpen } from 'lucide-react';
import { Button } from '@/components/ui/button';
import { Skeleton } from '@/components/ui/skeleton';
import Link from 'next/link';
import { toast } from 'sonner';

export function GalleryGrid() {
    const [modalOpen, setModalOpen] = useState(false);
    const [selectedSareeId, setSelectedSareeId] = useState<string | null>(null);

    const { data: gallery = [], isLoading, error, isError } = useQuery({
        queryKey: ['gallery'],
        queryFn: getGallery,
        refetchInterval: 5000, // Poll every 5s to check for updates
    });

    if (isError) {
        // We can show a toast here, but for persistent error state, UI is better
        // toast.error(`Failed to load gallery: ${error.message}`);
    }

    const handleGenerateMore = (sareeId: string) => {
        setSelectedSareeId(sareeId);
        setModalOpen(true);
    };

    const handleCloseModal = () => {
        setModalOpen(false);
        setSelectedSareeId(null);
    };

    // Check if selected saree has failures
    const selectedItem = gallery.find((item) => item.saree_id === selectedSareeId);
    const hasFailures = selectedItem?.latest_status === 'failed' || selectedItem?.latest_status === 'partial';

    if (isLoading) {
        return (
            <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4 gap-6">
                {Array.from({ length: 8 }).map((_, i) => (
                    <div key={i} className="flex flex-col gap-2">
                        <Skeleton className="aspect-[3/4] w-full rounded-lg" />
                        <div className="space-y-2">
                            <Skeleton className="h-4 w-[250px]" />
                            <Skeleton className="h-4 w-[200px]" />
                        </div>
                    </div>
                ))}
            </div>
        );
    }

    if (isError) {
        return (
            <div className="text-center py-16">
                <p className="text-destructive mb-4">
                    {error instanceof Error ? error.message : 'Failed to load gallery'}
                </p>
                <Button onClick={() => window.location.reload()}>Retry</Button>
            </div>
        );
    }

    if (gallery.length === 0) {
        return (
            <div className="text-center py-16">
                <FolderOpen className="h-16 w-16 mx-auto text-muted-foreground/50 mb-4" />
                <h2 className="text-lg font-medium mb-2">No sarees yet</h2>
                <p className="text-muted-foreground mb-6">
                    Upload a saree image to get started
                </p>
                <Button asChild>
                    <Link href="/">
                        <Upload className="h-4 w-4 mr-2" />
                        Upload Saree
                    </Link>
                </Button>
            </div>
        );
    }

    return (
        <>
            <div
                className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4 gap-6"
                data-testid="gallery-grid"
            >
                {gallery.map((item) => (
                    <GalleryTile
                        key={item.saree_id}
                        item={item}
                        onGenerateMore={handleGenerateMore}
                    />
                ))}
            </div>

            {/* Generate More Modal */}
            {selectedSareeId && (
                <GenerateMoreModal
                    sareeId={selectedSareeId}
                    isOpen={modalOpen}
                    onClose={handleCloseModal}
                    hasFailures={hasFailures}
                />
            )}
        </>
    );
}
