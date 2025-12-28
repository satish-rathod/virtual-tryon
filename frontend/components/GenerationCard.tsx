'use client';

import { useState } from 'react';
import Image from 'next/image';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { StatusBadge, type Status } from '@/components/StatusBadge';
import { Skeleton } from '@/components/ui/skeleton';
import { ImageLightbox } from '@/components/ImageLightbox';
import { getArtifactUrl, type Generation } from '@/lib/api';
import { Clock, RotateCcw } from 'lucide-react';
import { motion } from 'framer-motion';

interface GenerationCardProps {
    sareeId: string;
    generation: Generation;
}

export function GenerationCard({ sareeId, generation }: GenerationCardProps) {
    const [lightboxImage, setLightboxImage] = useState<{
        src: string;
        alt: string;
        viewNumber: number;
    } | null>(null);

    // Format timestamp
    const formattedTime = new Date(generation.timestamp).toLocaleString('en-US', {
        month: 'short',
        day: 'numeric',
        year: 'numeric',
        hour: 'numeric',
        minute: '2-digit',
    });

    return (
        <>
            <Card className="overflow-hidden">
                <CardHeader className="pb-4">
                    <div className="flex items-center justify-between">
                        <CardTitle className="text-base font-medium">
                            {generation.label}
                        </CardTitle>
                        <StatusBadge status={generation.status as Status} />
                    </div>
                    <div className="flex items-center gap-4 text-sm text-muted-foreground">
                        <div className="flex items-center gap-1.5">
                            <Clock className="h-3.5 w-3.5" />
                            {formattedTime}
                        </div>
                        {generation.retry_count > 0 && (
                            <div className="flex items-center gap-1.5">
                                <RotateCcw className="h-3.5 w-3.5" />
                                {generation.retry_count} {generation.retry_count === 1 ? 'retry' : 'retries'}
                            </div>
                        )}
                    </div>
                </CardHeader>

                <CardContent>
                    {/* Views grid */}
                    <div className="grid grid-cols-2 sm:grid-cols-3 md:grid-cols-4 gap-3">
                        {generation.views.map((view) => (
                            <motion.button
                                key={view.view_number}
                                initial={{ opacity: 0, scale: 0.95 }}
                                animate={{ opacity: 1, scale: 1 }}
                                transition={{ delay: view.view_number * 0.05 }}
                                onClick={() =>
                                    setLightboxImage({
                                        src: view.image_url.startsWith('http')
                                            ? view.image_url
                                            : getArtifactUrl(sareeId, view.image_url),
                                        alt: `View ${view.view_number}`,
                                        viewNumber: view.view_number,
                                    })
                                }
                                className="group relative aspect-[3/4] bg-muted rounded-lg overflow-hidden focus:outline-none focus:ring-2 focus:ring-ring focus:ring-offset-2"
                                data-testid={`view-${view.view_number}`}
                            >
                                {view.status === 'success' ? (
                                    <>
                                        <Image
                                            src={
                                                view.image_url.startsWith('http')
                                                    ? view.image_url
                                                    : getArtifactUrl(sareeId, view.image_url)
                                            }
                                            alt={`View ${view.view_number}`}
                                            fill
                                            className="object-cover transition-transform group-hover:scale-105"
                                            sizes="(max-width: 640px) 50vw, (max-width: 1024px) 25vw, 200px"
                                        />
                                        <div className="absolute inset-0 bg-gradient-to-t from-black/50 to-transparent opacity-0 group-hover:opacity-100 transition-opacity" />
                                    </>
                                ) : view.status === 'pending' ? (
                                    <Skeleton className="absolute inset-0 w-full h-full" />
                                ) : (
                                    <div className="absolute inset-0 flex items-center justify-center">
                                        <StatusBadge status={view.status as Status} />
                                    </div>
                                )}

                                {/* View label */}
                                <div className="absolute bottom-2 left-2 right-2">
                                    <span className="text-xs font-medium text-white bg-black/50 px-2 py-0.5 rounded">
                                        View {view.view_number}
                                    </span>
                                </div>
                            </motion.button>
                        ))}
                    </div>
                </CardContent>
            </Card>

            {/* Lightbox */}
            {lightboxImage && (
                <ImageLightbox
                    src={lightboxImage.src}
                    alt={lightboxImage.alt}
                    title={`View ${lightboxImage.viewNumber}`}
                    timestamp={formattedTime}
                    onClose={() => setLightboxImage(null)}
                />
            )}
        </>
    );
}
