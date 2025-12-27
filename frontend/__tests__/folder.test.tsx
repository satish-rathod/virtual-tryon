/**
 * Unit tests for FolderContent component
 */

import React from 'react';
import { render, screen, waitFor } from '@testing-library/react';
import { FolderContent } from '@/app/gallery/[saree_id]/FolderContent';
import * as api from '@/lib/api';

// Mock the API module
jest.mock('@/lib/api');
const mockApi = api as jest.Mocked<typeof api>;

// Mock next/navigation
jest.mock('next/navigation', () => ({
    useRouter: () => ({
        push: jest.fn(),
        replace: jest.fn(),
        refresh: jest.fn(),
    }),
}));

// Mock next/link
jest.mock('next/link', () => {
    return ({ children, href }: { children: React.ReactNode; href: string }) => (
        <a href={href}>{children}</a>
    );
});

describe('FolderContent', () => {
    beforeEach(() => {
        jest.clearAllMocks();
    });

    const mockSareeDetails = {
        saree_id: 'test-saree-123',
        created_at: '2024-01-15T10:30:00Z',
        artifacts: {
            original: 'original.jpg',
            cleaned: 'S_clean.png',
            flattened: 'S_flat.png',
            parts: {
                body: 'parts/body.png',
                pallu: 'parts/pallu.png',
                top_border: 'parts/top_border.png',
                bottom_border: 'parts/bottom_border.png',
            },
        },
        generations: [
            {
                generation_id: 'gen-001',
                label: 'Standard Views' as const,
                mode: 'standard' as const,
                status: 'success' as const,
                timestamp: '2024-01-15T10:35:00Z',
                views: [
                    { view_number: 1, image_url: 'generations/gen_01/final_view_01.png', status: 'success' as const },
                    { view_number: 2, image_url: 'generations/gen_01/final_view_02.png', status: 'success' as const },
                    { view_number: 3, image_url: 'generations/gen_01/final_view_03.png', status: 'success' as const },
                    { view_number: 4, image_url: 'generations/gen_01/final_view_04.png', status: 'success' as const },
                ],
                retry_count: 0,
            },
        ],
        has_failures: false,
    };

    it('renders saree artifacts tabs', async () => {
        mockApi.getSareeDetails.mockResolvedValueOnce(mockSareeDetails);
        mockApi.getArtifactUrl.mockImplementation(
            (id, path) => `http://localhost:8000/api/artifacts/${id}/${path}`
        );

        render(<FolderContent sareeId="test-saree-123" />);

        await waitFor(() => {
            expect(screen.getByText('Saree Artifacts')).toBeInTheDocument();
        });

        // Check artifact tabs
        expect(screen.getByRole('tab', { name: 'Original' })).toBeInTheDocument();
        expect(screen.getByRole('tab', { name: 'Cleaned' })).toBeInTheDocument();
        expect(screen.getByRole('tab', { name: 'Flattened' })).toBeInTheDocument();
        expect(screen.getByRole('tab', { name: 'Parts' })).toBeInTheDocument();
    });

    it('renders generation cards', async () => {
        mockApi.getSareeDetails.mockResolvedValueOnce(mockSareeDetails);
        mockApi.getArtifactUrl.mockImplementation(
            (id, path) => `http://localhost:8000/api/artifacts/${id}/${path}`
        );

        render(<FolderContent sareeId="test-saree-123" />);

        await waitFor(() => {
            expect(screen.getByText('Generations')).toBeInTheDocument();
        });

        // Check generation card
        expect(screen.getByText('Standard Views')).toBeInTheDocument();
    });

    it('renders view labels as "View 1", "View 2", etc. (not pose IDs)', async () => {
        mockApi.getSareeDetails.mockResolvedValueOnce(mockSareeDetails);
        mockApi.getArtifactUrl.mockImplementation(
            (id, path) => `http://localhost:8000/api/artifacts/${id}/${path}`
        );

        render(<FolderContent sareeId="test-saree-123" />);

        await waitFor(() => {
            expect(screen.getByText('View 1')).toBeInTheDocument();
        });

        expect(screen.getByText('View 2')).toBeInTheDocument();
        expect(screen.getByText('View 3')).toBeInTheDocument();
        expect(screen.getByText('View 4')).toBeInTheDocument();

        // Ensure no pose identifiers are exposed (should not have "pose_01" style labels)
        expect(screen.queryByText(/pose_\d+/i)).not.toBeInTheDocument();
        expect(screen.queryByText(/pose \d+/i)).not.toBeInTheDocument();
    });

    it('renders Generate More Views button', async () => {
        mockApi.getSareeDetails.mockResolvedValueOnce(mockSareeDetails);
        mockApi.getArtifactUrl.mockImplementation(
            (id, path) => `http://localhost:8000/api/artifacts/${id}/${path}`
        );

        render(<FolderContent sareeId="test-saree-123" />);

        await waitFor(() => {
            expect(screen.getByTestId('generate-more-button')).toBeInTheDocument();
        });

        expect(screen.getByText('Generate More Views')).toBeInTheDocument();
    });

    it('shows error state when API fails', async () => {
        mockApi.getSareeDetails.mockRejectedValueOnce(new Error('Saree not found'));

        render(<FolderContent sareeId="invalid-id" />);

        await waitFor(() => {
            expect(screen.getByText('Saree not found')).toBeInTheDocument();
        });

        expect(screen.getByText('Back to Gallery')).toBeInTheDocument();
    });

    it('shows short saree ID in header', async () => {
        mockApi.getSareeDetails.mockResolvedValueOnce(mockSareeDetails);
        mockApi.getArtifactUrl.mockImplementation(
            (id, path) => `http://localhost:8000/api/artifacts/${id}/${path}`
        );

        render(<FolderContent sareeId="test-saree-123" />);

        await waitFor(() => {
            // Short ID (first 8 characters)
            expect(screen.getByText('test-sar')).toBeInTheDocument();
        });
    });

    it('shows empty state when no generations exist', async () => {
        const mockNoGenerations = {
            ...mockSareeDetails,
            generations: [],
        };
        mockApi.getSareeDetails.mockResolvedValueOnce(mockNoGenerations);
        mockApi.getArtifactUrl.mockImplementation(
            (id, path) => `http://localhost:8000/api/artifacts/${id}/${path}`
        );

        render(<FolderContent sareeId="test-saree-123" />);

        await waitFor(() => {
            expect(screen.getByText(/No generations yet/)).toBeInTheDocument();
        });
    });
});
