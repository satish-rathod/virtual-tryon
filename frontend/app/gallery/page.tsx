import { Suspense } from 'react';
import { GalleryGrid } from './GalleryGrid';
import { Loader2 } from 'lucide-react';

export const dynamic = 'force-dynamic';

export default function GalleryPage() {
    return (
        <div className="container px-4 md:px-8 py-8">
            <div className="mb-8">
                <h1 className="text-2xl font-semibold tracking-tight">Saree Gallery</h1>
                <p className="text-muted-foreground mt-1">
                    Browse your saree uploads and generated views
                </p>
            </div>

            <Suspense fallback={<GalleryLoading />}>
                <GalleryGrid />
            </Suspense>
        </div>
    );
}

function GalleryLoading() {
    return (
        <div className="flex items-center justify-center py-16">
            <Loader2 className="h-8 w-8 animate-spin text-muted-foreground" />
        </div>
    );
}
