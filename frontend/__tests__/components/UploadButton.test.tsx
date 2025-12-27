/**
 * Unit tests for UploadButton component
 */

import React from 'react';
import { render, screen, fireEvent, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { UploadButton } from '@/components/UploadButton';
import * as api from '@/lib/api';

// Mock the API module
jest.mock('@/lib/api');
const mockApi = api as jest.Mocked<typeof api>;

// Mock next/navigation
const mockPush = jest.fn();
jest.mock('next/navigation', () => ({
    useRouter: () => ({
        push: mockPush,
        replace: jest.fn(),
        refresh: jest.fn(),
    }),
}));

describe('UploadButton', () => {
    beforeEach(() => {
        jest.clearAllMocks();
    });

    it('renders upload button', () => {
        render(<UploadButton />);
        expect(screen.getByTestId('upload-button')).toBeInTheDocument();
        expect(screen.getByText('Upload Saree')).toBeInTheDocument();
    });

    it('triggers file input when button is clicked', async () => {
        render(<UploadButton />);
        const fileInput = screen.getByTestId('file-input') as HTMLInputElement;
        const button = screen.getByTestId('upload-button');

        // Spy on click of file input
        const clickSpy = jest.spyOn(fileInput, 'click');

        await userEvent.click(button);

        expect(clickSpy).toHaveBeenCalled();
    });

    it('calls upload API and then generate API with standard mode on file selection', async () => {
        const mockUploadResponse = {
            saree_id: 'test-saree-id',
            upload_path: 'artifacts/test-saree-id/original.jpg',
        };
        const mockGenerateResponse = {
            job_id: 'test-job-id',
            status: 'queued' as const,
        };

        mockApi.uploadSaree.mockResolvedValueOnce(mockUploadResponse);
        mockApi.generateViews.mockResolvedValueOnce(mockGenerateResponse);

        render(<UploadButton />);

        const fileInput = screen.getByTestId('file-input');
        const testFile = new File(['test content'], 'saree.jpg', { type: 'image/jpeg' });

        await userEvent.upload(fileInput, testFile);

        await waitFor(() => {
            expect(mockApi.uploadSaree).toHaveBeenCalledWith(testFile);
        });

        await waitFor(() => {
            expect(mockApi.generateViews).toHaveBeenCalledWith('test-saree-id', 'standard');
        });
    });

    it('navigates to folder view after successful upload and generate', async () => {
        const mockUploadResponse = {
            saree_id: 'test-saree-id',
            upload_path: 'artifacts/test-saree-id/original.jpg',
        };
        const mockGenerateResponse = {
            job_id: 'test-job-id',
            status: 'queued' as const,
        };

        mockApi.uploadSaree.mockResolvedValueOnce(mockUploadResponse);
        mockApi.generateViews.mockResolvedValueOnce(mockGenerateResponse);

        render(<UploadButton />);

        const fileInput = screen.getByTestId('file-input');
        const testFile = new File(['test content'], 'saree.jpg', { type: 'image/jpeg' });

        await userEvent.upload(fileInput, testFile);

        await waitFor(() => {
            expect(mockPush).toHaveBeenCalledWith('/gallery/test-saree-id');
        });
    });

    it('shows loading state during upload', async () => {
        // Make upload hang indefinitely
        mockApi.uploadSaree.mockImplementation(
            () => new Promise(() => { })
        );

        render(<UploadButton />);

        const fileInput = screen.getByTestId('file-input');
        const testFile = new File(['test content'], 'saree.jpg', { type: 'image/jpeg' });

        fireEvent.change(fileInput, { target: { files: [testFile] } });

        await waitFor(() => {
            expect(screen.getByText(/Uploading/)).toBeInTheDocument();
        });
    });

    it('shows generating state after upload completes', async () => {
        const mockUploadResponse = {
            saree_id: 'test-saree-id',
            upload_path: 'artifacts/test-saree-id/original.jpg',
        };

        mockApi.uploadSaree.mockResolvedValueOnce(mockUploadResponse);
        // Make generate hang indefinitely
        mockApi.generateViews.mockImplementation(
            () => new Promise(() => { })
        );

        render(<UploadButton />);

        const fileInput = screen.getByTestId('file-input');
        const testFile = new File(['test content'], 'saree.jpg', { type: 'image/jpeg' });

        fireEvent.change(fileInput, { target: { files: [testFile] } });

        await waitFor(() => {
            expect(screen.getByText(/Generating standard views/)).toBeInTheDocument();
        });
    });
});
