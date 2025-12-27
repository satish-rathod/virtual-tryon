'use client';

import Image from 'next/image';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { getArtifactUrl, type SareeDetails } from '@/lib/api';
import { motion } from 'framer-motion';

interface ArtifactTabsProps {
    sareeId: string;
    artifacts: SareeDetails['artifacts'];
}

export function ArtifactTabs({ sareeId, artifacts }: ArtifactTabsProps) {
    const hasCleanedArtifact = Boolean(artifacts.cleaned);
    const hasFlattenedArtifact = Boolean(artifacts.flattened);
    const hasPartsArtifacts = Boolean(artifacts.parts);

    return (
        <Tabs defaultValue="original" className="w-full">
            <TabsList className="grid w-full grid-cols-4 max-w-md">
                <TabsTrigger value="original">Original</TabsTrigger>
                <TabsTrigger value="cleaned" disabled={!hasCleanedArtifact}>
                    Cleaned
                </TabsTrigger>
                <TabsTrigger value="flattened" disabled={!hasFlattenedArtifact}>
                    Flattened
                </TabsTrigger>
                <TabsTrigger value="parts" disabled={!hasPartsArtifacts}>
                    Parts
                </TabsTrigger>
            </TabsList>

            <TabsContent value="original" className="mt-6">
                <ArtifactImage
                    src={getArtifactUrl(sareeId, artifacts.original)}
                    alt="Original saree image"
                />
            </TabsContent>

            <TabsContent value="cleaned" className="mt-6">
                {artifacts.cleaned && (
                    <ArtifactImage
                        src={getArtifactUrl(sareeId, artifacts.cleaned)}
                        alt="Cleaned saree (background removed)"
                    />
                )}
            </TabsContent>

            <TabsContent value="flattened" className="mt-6">
                {artifacts.flattened && (
                    <ArtifactImage
                        src={getArtifactUrl(sareeId, artifacts.flattened)}
                        alt="Flattened saree"
                    />
                )}
            </TabsContent>

            <TabsContent value="parts" className="mt-6">
                {artifacts.parts && (
                    <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                        {artifacts.parts.body && (
                            <PartImage
                                src={getArtifactUrl(sareeId, artifacts.parts.body)}
                                label="Body"
                            />
                        )}
                        {artifacts.parts.pallu && (
                            <PartImage
                                src={getArtifactUrl(sareeId, artifacts.parts.pallu)}
                                label="Pallu"
                            />
                        )}
                        {artifacts.parts.top_border && (
                            <PartImage
                                src={getArtifactUrl(sareeId, artifacts.parts.top_border)}
                                label="Top Border"
                            />
                        )}
                        {artifacts.parts.bottom_border && (
                            <PartImage
                                src={getArtifactUrl(sareeId, artifacts.parts.bottom_border)}
                                label="Bottom Border"
                            />
                        )}
                    </div>
                )}
            </TabsContent>
        </Tabs>
    );
}

function ArtifactImage({ src, alt }: { src: string; alt: string }) {
    return (
        <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            className="relative aspect-[3/4] max-w-md bg-muted rounded-xl overflow-hidden"
        >
            <Image
                src={src}
                alt={alt}
                fill
                className="object-contain"
                sizes="(max-width: 768px) 100vw, 400px"
            />
        </motion.div>
    );
}

function PartImage({ src, label }: { src: string; label: string }) {
    return (
        <div className="space-y-2">
            <div className="relative aspect-square bg-muted rounded-lg overflow-hidden">
                <Image
                    src={src}
                    alt={`${label} part`}
                    fill
                    className="object-contain"
                    sizes="150px"
                />
            </div>
            <p className="text-sm text-center text-muted-foreground">{label}</p>
        </div>
    );
}
