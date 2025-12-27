/**
 * Integration test simulating full upload → generate → poll flow using MSW
 */

import React from 'react';
import { render, screen, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { UploadButton } from '@/components/UploadButton';

// Mock the entire API module for integration testing
jest.mock('@/lib/api', () => {
    let callCount = 0;

    return {
        uploadSaree: jest.fn().mockResolvedValue({
            saree_id: 'integration-test-saree',
            upload_path: 'artifacts/integration-test-saree/original.jpg',
        }),
        generateViews: jest.fn().mockResolvedValue({
            job_id: 'integration-test-job',
            status: 'queued',
        }),
        getJobStatus: jest.fn().mockImplementation(() => {
            callCount++;
            // Simulate job progressing through stages
            if (callCount === 1) {
                return Promise.resolve({
                    job_id: 'integration-test-job',
                    status: 'running',
                    progress: 25,
                    current_stage: 'flatten',
                });
            }
            if (callCount === 2) {
                return Promise.resolve({
                    job_id: 'integration-test-job',
                    status: 'running',
                    progress: 75,
                    current_stage: 'compose',
                });
            }
            return Promise.resolve({
                job_id: 'integration-test-job',
                status: 'success',
                progress: 100,
                artifacts: ['final_view_01.png', 'final_view_02.png', 'final_view_03.png', 'final_view_04.png'],
            });
        }),
        getGallery: jest.fn().mockResolvedValue([]),
        getSareeDetails: jest.fn().mockResolvedValue({
            saree_id: 'integration-test-saree',
            created_at: new Date().toISOString(),
            artifacts: {
                original: 'original.jpg',
            },
            generations: [
                {
                    generation_id: 'gen-001',
                    label: 'Standard Views',
                    mode: 'standard',
                    status: 'success',
                    timestamp: new Date().toISOString(),
                    views: [
                        { view_number: 1, image_url: 'final_view_01.png', status: 'success' },
                        { view_number: 2, image_url: 'final_view_02.png', status: 'success' },
                        { view_number: 3, image_url: 'final_view_03.png', status: 'success' },
                        { view_number: 4, image_url: 'final_view_04.png', status: 'success' },
                    ],
                    retry_count: 0,
                },
            ],
            has_failures: false,
        }),
        getArtifactUrl: jest.fn().mockImplementation(
            (id: string, path: string) => `http://localhost:8000/api/artifacts/${id}/${path}`
        ),
        getThumbnailUrl: jest.fn().mockImplementation(
            (id: string) => `http://localhost:8000/api/artifacts/${id}/original.jpg`
        ),
        getLogs: jest.fn().mockResolvedValue({
            job_id: 'integration-test-job',
            retry_log: [],
            failure_reasons: [],
        }),
        pollJobStatus: jest.fn().mockResolvedValue({
            job_id: 'integration-test-job',
            status: 'success',
            progress: 100,
        }),
    };
});

// Mock router
const mockPush = jest.fn();
jest.mock('next/navigation', () => ({
    useRouter: () => ({
        push: mockPush,
        replace: jest.fn(),
        refresh: jest.fn(),
    }),
}));

describe('Integration: Upload → Generate → Success flow', () => {
    beforeEach(() => {
        jest.clearAllMocks();
    });

    it('completes full upload flow and navigates to folder view', async () => {
        const api = require('@/lib/api');

        render(<UploadButton />);

        // Step 1: Click upload button and select file
        const fileInput = screen.getByTestId('file-input');
        const testFile = new File(['test image content'], 'test-saree.jpg', {
            type: 'image/jpeg',
        });

        await userEvent.upload(fileInput, testFile);

        // Step 2: Verify upload API was called
        await waitFor(() => {
            expect(api.uploadSaree).toHaveBeenCalledWith(testFile);
        });

        // Step 3: Verify generate API was called with standard mode
        await waitFor(() => {
            expect(api.generateViews).toHaveBeenCalledWith('integration-test-saree', 'standard');
        });

        // Step 4: Verify navigation to folder view
        await waitFor(() => {
            expect(mockPush).toHaveBeenCalledWith('/gallery/integration-test-saree');
        });
    });

    it('enforces deterministic generation mode (standard only) for initial upload', async () => {
        const api = require('@/lib/api');

        render(<UploadButton />);

        const fileInput = screen.getByTestId('file-input');
        const testFile = new File(['content'], 'saree.jpg', { type: 'image/jpeg' });

        await userEvent.upload(fileInput, testFile);

        // The generate call must use 'standard' mode - this is a non-negotiable constraint
        await waitFor(() => {
            expect(api.generateViews).toHaveBeenCalledWith(
                expect.any(String),
                'standard' // MUST be 'standard' for initial generation
            );
        });

        // Verify it was NOT called with any other mode
        expect(api.generateViews).not.toHaveBeenCalledWith(expect.any(String), 'extend');
        expect(api.generateViews).not.toHaveBeenCalledWith(expect.any(String), 'retry_failed');
    });
});
