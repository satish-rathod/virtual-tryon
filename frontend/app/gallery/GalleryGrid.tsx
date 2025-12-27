'use client';

import { useState, useEffect } from 'react';
import { GalleryTile } from '@/components/GalleryTile';
import { GenerateMoreModal } from '@/components/GenerateMoreModal';
import { getGallery, type GalleryItem } from '@/lib/api';
import { Upload, FolderOpen } from 'lucide-react';
import { Button } from '@/components/ui/button';
import Link from 'next/link';

export function GalleryGrid() {
    const [gallery, setGallery] = useState<GalleryItem[]>([]);
    const [loading, setLoading] = useState(true);
    const [error, setError] = useState<string | null>(null);
    const [modalOpen, setModalOpen] = useState(false);
    const [selectedSareeId, setSelectedSareeId] = useState<string | null>(null);

    useEffect(() => {
        async function fetchGallery() {
            try {
                setLoading(true);
                const data = await getGallery();
                setGallery(data);
                setError(null);
            } catch (err) {
                setError(err instanceof Error ? err.message : 'Failed to load gallery');
            } finally {
                setLoading(false);
            }
        }

        fetchGallery();
    }, []);

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

    if (loading) {
        return null; // Suspense will handle loading state
    }

    if (error) {
        return (
            <div className="text-center py-16">
                <p className="text-destructive mb-4">{error}</p>
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
