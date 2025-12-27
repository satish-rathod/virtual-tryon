/**
 * Unit tests for GalleryGrid component
 */

import React from 'react';
import { render, screen, waitFor } from '@testing-library/react';
import { GalleryGrid } from '@/app/gallery/GalleryGrid';
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

describe('GalleryGrid', () => {
    beforeEach(() => {
        jest.clearAllMocks();
    });

    it('renders gallery tiles from API response', async () => {
        const mockGalleryData = [
            {
                saree_id: 'saree-123',
                created_at: '2024-01-15T10:30:00Z',
                thumbnail: 'original.jpg',
                generation_count: 2,
                latest_status: 'success' as const,
            },
            {
                saree_id: 'saree-456',
                created_at: '2024-01-16T14:45:00Z',
                thumbnail: 'original.jpg',
                generation_count: 1,
                latest_status: 'partial' as const,
            },
        ];

        mockApi.getGallery.mockResolvedValueOnce(mockGalleryData);
        mockApi.getThumbnailUrl.mockImplementation(
            (id) => `http://localhost:8000/api/artifacts/${id}/original.jpg`
        );

        render(<GalleryGrid />);

        await waitFor(() => {
            expect(screen.getByTestId('gallery-grid')).toBeInTheDocument();
        });

        // Check that tiles are rendered
        expect(screen.getByText('saree-12')).toBeInTheDocument(); // short ID
        expect(screen.getByText('saree-45')).toBeInTheDocument();

        // Check generation counts
        expect(screen.getByText('2 generations')).toBeInTheDocument();
        expect(screen.getByText('1 generation')).toBeInTheDocument();
    });

    it('renders empty state when no sarees exist', async () => {
        mockApi.getGallery.mockResolvedValueOnce([]);

        render(<GalleryGrid />);

        await waitFor(() => {
            expect(screen.getByText('No sarees yet')).toBeInTheDocument();
        });

        expect(screen.getByText('Upload a saree image to get started')).toBeInTheDocument();
    });

    it('shows error message on API failure', async () => {
        mockApi.getGallery.mockRejectedValueOnce(new Error('Network error'));

        render(<GalleryGrid />);

        await waitFor(() => {
            expect(screen.getByText('Network error')).toBeInTheDocument();
        });
    });

    it('renders status badges correctly', async () => {
        const mockGalleryData = [
            {
                saree_id: 'saree-success',
                created_at: '2024-01-15T10:30:00Z',
                thumbnail: 'original.jpg',
                generation_count: 1,
                latest_status: 'success' as const,
            },
            {
                saree_id: 'saree-failed',
                created_at: '2024-01-15T10:30:00Z',
                thumbnail: 'original.jpg',
                generation_count: 1,
                latest_status: 'failed' as const,
            },
        ];

        mockApi.getGallery.mockResolvedValueOnce(mockGalleryData);
        mockApi.getThumbnailUrl.mockImplementation(
            (id) => `http://localhost:8000/api/artifacts/${id}/original.jpg`
        );

        render(<GalleryGrid />);

        await waitFor(() => {
            expect(screen.getByText('Success')).toBeInTheDocument();
            expect(screen.getByText('Failed')).toBeInTheDocument();
        });
    });

    it('renders Open and More Views buttons for each tile', async () => {
        const mockGalleryData = [
            {
                saree_id: 'saree-123',
                created_at: '2024-01-15T10:30:00Z',
                thumbnail: 'original.jpg',
                generation_count: 1,
                latest_status: 'success' as const,
            },
        ];

        mockApi.getGallery.mockResolvedValueOnce(mockGalleryData);
        mockApi.getThumbnailUrl.mockImplementation(
            (id) => `http://localhost:8000/api/artifacts/${id}/original.jpg`
        );

        render(<GalleryGrid />);

        await waitFor(() => {
            expect(screen.getByText('Open')).toBeInTheDocument();
            expect(screen.getByText('More Views')).toBeInTheDocument();
        });
    });
});
