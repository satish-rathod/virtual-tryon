import { Suspense } from 'react';
import { FolderContent } from './FolderContent';
import { Loader2 } from 'lucide-react';

export const dynamic = 'force-dynamic';

interface FolderPageProps {
    params: Promise<{ saree_id: string }>;
}

export default async function FolderPage({ params }: FolderPageProps) {
    const { saree_id } = await params;

    return (
        <div className="container px-4 md:px-8 py-8">
            <Suspense fallback={<FolderLoading />}>
                <FolderContent sareeId={saree_id} />
            </Suspense>
        </div>
    );
}

function FolderLoading() {
    return (
        <div className="flex items-center justify-center py-16">
            <Loader2 className="h-8 w-8 animate-spin text-muted-foreground" />
        </div>
    );
}
