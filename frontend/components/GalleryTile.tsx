'use client';

import Image from 'next/image';
import Link from 'next/link';
import { Card, CardContent, CardFooter } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { StatusBadge, type Status } from '@/components/StatusBadge';
import { getThumbnailUrl, type GalleryItem } from '@/lib/api';
import { FolderOpen, Wand2, Calendar, Hash, Layers } from 'lucide-react';
import { motion } from 'framer-motion';

interface GalleryTileProps {
    item: GalleryItem;
    onGenerateMore: (sareeId: string) => void;
}

export function GalleryTile({ item, onGenerateMore }: GalleryTileProps) {
    // Format date to readable string
    const formattedDate = new Date(item.created_at).toLocaleDateString('en-US', {
        month: 'short',
        day: 'numeric',
        year: 'numeric',
    });

    // Short ID (first 8 characters)
    const shortId = item.saree_id.substring(0, 8);

    return (
        <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.3 }}
        >
            <Card className="overflow-hidden hover:shadow-md transition-shadow">
                {/* Thumbnail */}
                <div className="relative aspect-[4/5] bg-muted">
                    <Image
                        src={getThumbnailUrl(item.saree_id)}
                        alt={`Saree ${shortId}`}
                        fill
                        className="object-cover"
                        sizes="(max-width: 640px) 100vw, (max-width: 1024px) 50vw, 33vw"
                        placeholder="blur"
                        blurDataURL="data:image/jpeg;base64,/9j/4AAQSkZJRgABAQAAAQABAAD/2wBDAAMCAgMCAgMDAwMEAwMEBQgFBQQEBQoHBwYIDAoMCwsKCwsNDhIQDQ4RDgsLEBYQERMUFRUVDA8XGBYUGBIUFRT/2wBDAQMEBAUEBQkFBQkUDQsNFBQUFBQUFBQUFBQUFBQUFBQUFBQUFBQUFBQUFBQUFBQUFBQUFBQUFBQUFBQUFBQUFBT/wAARCAAIAAoDASIAAhEBAxEB/8QAFgABAQEAAAAAAAAAAAAAAAAAAAcI/8QAIhAAAgICAgICAwAAAAAAAAAAAQIDBAUGABESIQcTMVFh/8QAFQEBAQAAAAAAAAAAAAAAAAAAAAX/xAAZEQADAQEBAAAAAAAAAAAAAAAAAQIREjH/2gAMAwEAAhEDEEEAAA=="
                    />
                </div>

                <CardContent className="p-4 space-y-3">
                    {/* ID and Status */}
                    <div className="flex items-center justify-between">
                        <div className="flex items-center gap-1.5 text-sm font-mono text-muted-foreground">
                            <Hash className="h-3.5 w-3.5" />
                            {shortId}
                        </div>
                        <StatusBadge status={item.latest_status as Status} />
                    </div>

                    {/* Metadata */}
                    <div className="flex items-center gap-4 text-xs text-muted-foreground">
                        <div className="flex items-center gap-1">
                            <Calendar className="h-3.5 w-3.5" />
                            {formattedDate}
                        </div>
                        <div className="flex items-center gap-1">
                            <Layers className="h-3.5 w-3.5" />
                            {item.generation_count} {item.generation_count === 1 ? 'generation' : 'generations'}
                        </div>
                    </div>
                </CardContent>

                <CardFooter className="p-4 pt-0 gap-2">
                    <Button asChild variant="outline" size="sm" className="flex-1">
                        <Link href={`/gallery/${item.saree_id}`}>
                            <FolderOpen className="h-4 w-4 mr-1.5" />
                            Open
                        </Link>
                    </Button>
                    <Button
                        variant="secondary"
                        size="sm"
                        className="flex-1"
                        onClick={() => onGenerateMore(item.saree_id)}
                        data-testid={`generate-more-${item.saree_id}`}
                    >
                        <Wand2 className="h-4 w-4 mr-1.5" />
                        More Views
                    </Button>
                </CardFooter>
            </Card>
        </motion.div>
    );
}
