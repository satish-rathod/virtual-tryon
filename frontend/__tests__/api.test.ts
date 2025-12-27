/**
 * Unit tests for API helper functions
 */

import {
    uploadSaree,
    generateViews,
    getJobStatus,
    getGallery,
    getSareeDetails,
    getLogs,
    getArtifactUrl,
    getThumbnailUrl,
} from '@/lib/api';

// Mock fetch globally
const mockFetch = jest.fn();
global.fetch = mockFetch;

describe('API Functions', () => {
    beforeEach(() => {
        mockFetch.mockClear();
    });

    describe('uploadSaree', () => {
        it('should upload file and return saree_id', async () => {
            const mockResponse = {
                saree_id: 'test-uuid-123',
                upload_path: 'artifacts/test-uuid-123/original.jpg',
            };

            mockFetch.mockResolvedValueOnce({
                ok: true,
                json: async () => mockResponse,
            });

            const file = new File(['test content'], 'test.jpg', { type: 'image/jpeg' });
            const result = await uploadSaree(file);

            expect(result).toEqual(mockResponse);
            expect(mockFetch).toHaveBeenCalledTimes(1);
            expect(mockFetch).toHaveBeenCalledWith(
                'http://localhost:8000/api/upload',
                expect.objectContaining({
                    method: 'POST',
                    body: expect.any(FormData),
                })
            );
        });

        it('should throw error on upload failure', async () => {
            mockFetch.mockResolvedValueOnce({
                ok: false,
                text: async () => 'Upload error message',
            });

            const file = new File(['test'], 'test.jpg', { type: 'image/jpeg' });
            await expect(uploadSaree(file)).rejects.toThrow('Upload failed: Upload error message');
        });
    });

    describe('generateViews', () => {
        it('should trigger generation with standard mode by default', async () => {
            const mockResponse = { job_id: 'job-123', status: 'queued' };

            mockFetch.mockResolvedValueOnce({
                ok: true,
                json: async () => mockResponse,
            });

            const result = await generateViews('saree-123');

            expect(result).toEqual(mockResponse);
            expect(mockFetch).toHaveBeenCalledWith(
                'http://localhost:8000/api/generate',
                expect.objectContaining({
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ saree_id: 'saree-123', mode: 'standard' }),
                })
            );
        });

        it('should trigger generation with extend mode', async () => {
            const mockResponse = { job_id: 'job-456', status: 'queued' };

            mockFetch.mockResolvedValueOnce({
                ok: true,
                json: async () => mockResponse,
            });

            const result = await generateViews('saree-123', 'extend');

            expect(result).toEqual(mockResponse);
            expect(mockFetch).toHaveBeenCalledWith(
                'http://localhost:8000/api/generate',
                expect.objectContaining({
                    body: JSON.stringify({ saree_id: 'saree-123', mode: 'extend' }),
                })
            );
        });

        it('should trigger generation with retry_failed mode', async () => {
            const mockResponse = { job_id: 'job-789', status: 'queued' };

            mockFetch.mockResolvedValueOnce({
                ok: true,
                json: async () => mockResponse,
            });

            const result = await generateViews('saree-123', 'retry_failed');

            expect(result).toEqual(mockResponse);
            expect(mockFetch).toHaveBeenCalledWith(
                'http://localhost:8000/api/generate',
                expect.objectContaining({
                    body: JSON.stringify({ saree_id: 'saree-123', mode: 'retry_failed' }),
                })
            );
        });
    });

    describe('getJobStatus', () => {
        it('should fetch job status', async () => {
            const mockResponse = {
                job_id: 'job-123',
                status: 'running',
                progress: 50,
                current_stage: 'ai_flatten',
            };

            mockFetch.mockResolvedValueOnce({
                ok: true,
                json: async () => mockResponse,
            });

            const result = await getJobStatus('job-123');

            expect(result).toEqual(mockResponse);
            expect(mockFetch).toHaveBeenCalledWith('http://localhost:8000/api/status/job-123');
        });
    });

    describe('getGallery', () => {
        it('should fetch gallery list', async () => {
            const mockResponse = [
                {
                    saree_id: 'saree-1',
                    created_at: '2024-01-01T00:00:00Z',
                    thumbnail: 'original.jpg',
                    generation_count: 1,
                    latest_status: 'success',
                },
            ];

            mockFetch.mockResolvedValueOnce({
                ok: true,
                json: async () => mockResponse,
            });

            const result = await getGallery();

            expect(result).toEqual(mockResponse);
            expect(mockFetch).toHaveBeenCalledWith(
                'http://localhost:8000/api/gallery',
                expect.objectContaining({ cache: 'no-store' })
            );
        });
    });

    describe('getSareeDetails', () => {
        it('should fetch saree details', async () => {
            const mockResponse = {
                saree_id: 'saree-123',
                created_at: '2024-01-01T00:00:00Z',
                artifacts: {
                    original: 'original.jpg',
                    cleaned: 'S_clean.png',
                },
                generations: [],
                has_failures: false,
            };

            mockFetch.mockResolvedValueOnce({
                ok: true,
                json: async () => mockResponse,
            });

            const result = await getSareeDetails('saree-123');

            expect(result).toEqual(mockResponse);
            expect(mockFetch).toHaveBeenCalledWith(
                'http://localhost:8000/api/gallery/saree-123',
                expect.objectContaining({ cache: 'no-store' })
            );
        });
    });

    describe('getLogs', () => {
        it('should fetch job logs', async () => {
            const mockResponse = {
                job_id: 'job-123',
                retry_log: [],
                failure_reasons: [],
            };

            mockFetch.mockResolvedValueOnce({
                ok: true,
                json: async () => mockResponse,
            });

            const result = await getLogs('job-123');

            expect(result).toEqual(mockResponse);
            expect(mockFetch).toHaveBeenCalledWith('http://localhost:8000/api/logs/job-123');
        });
    });

    describe('URL builders', () => {
        it('should build artifact URL correctly', () => {
            const url = getArtifactUrl('saree-123', 'generations/gen_01/final_view_01.png');
            expect(url).toBe('http://localhost:8000/api/artifacts/saree-123/generations/gen_01/final_view_01.png');
        });

        it('should build thumbnail URL correctly', () => {
            const url = getThumbnailUrl('saree-123');
            expect(url).toBe('http://localhost:8000/api/artifacts/saree-123/original.jpg');
        });
    });
});
