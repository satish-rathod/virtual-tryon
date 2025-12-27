/**
 * Unit tests for GenerateMoreModal component
 */

import React from 'react';
import { render, screen, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { GenerateMoreModal } from '@/components/GenerateMoreModal';
import * as api from '@/lib/api';

// Mock the API module
jest.mock('@/lib/api');
const mockApi = api as jest.Mocked<typeof api>;

// Mock next/navigation
const mockRefresh = jest.fn();
jest.mock('next/navigation', () => ({
    useRouter: () => ({
        push: jest.fn(),
        replace: jest.fn(),
        refresh: mockRefresh,
    }),
}));

describe('GenerateMoreModal', () => {
    const defaultProps = {
        sareeId: 'test-saree-id',
        isOpen: true,
        onClose: jest.fn(),
        hasFailures: false,
    };

    beforeEach(() => {
        jest.clearAllMocks();
    });

    it('renders modal with title and description', () => {
        render(<GenerateMoreModal {...defaultProps} />);

        expect(screen.getByText('Generate additional views')).toBeInTheDocument();
        expect(
            screen.getByText('Additional views are generated using predefined model angles.')
        ).toBeInTheDocument();
    });

    it('renders both action buttons', () => {
        render(<GenerateMoreModal {...defaultProps} />);

        expect(screen.getByText('Generate remaining views')).toBeInTheDocument();
        expect(screen.getByText('Retry failed views')).toBeInTheDocument();
    });

    it('calls generate API with extend mode when "Generate remaining views" is clicked', async () => {
        const mockGenerateResponse = {
            job_id: 'job-123',
            status: 'queued' as const,
        };
        mockApi.generateViews.mockResolvedValueOnce(mockGenerateResponse);

        render(<GenerateMoreModal {...defaultProps} />);

        const extendButton = screen.getByTestId('generate-extend-button');
        await userEvent.click(extendButton);

        await waitFor(() => {
            expect(mockApi.generateViews).toHaveBeenCalledWith('test-saree-id', 'extend');
        });
    });

    it('calls generate API with retry_failed mode when "Retry failed views" is clicked', async () => {
        const mockGenerateResponse = {
            job_id: 'job-456',
            status: 'queued' as const,
        };
        mockApi.generateViews.mockResolvedValueOnce(mockGenerateResponse);

        render(<GenerateMoreModal {...defaultProps} hasFailures={true} />);

        const retryButton = screen.getByTestId('retry-failed-button');
        await userEvent.click(retryButton);

        await waitFor(() => {
            expect(mockApi.generateViews).toHaveBeenCalledWith('test-saree-id', 'retry_failed');
        });
    });

    it('disables retry button when no failures exist', () => {
        render(<GenerateMoreModal {...defaultProps} hasFailures={false} />);

        const retryButton = screen.getByTestId('retry-failed-button');
        expect(retryButton).toBeDisabled();
    });

    it('enables retry button when failures exist', () => {
        render(<GenerateMoreModal {...defaultProps} hasFailures={true} />);

        const retryButton = screen.getByTestId('retry-failed-button');
        expect(retryButton).not.toBeDisabled();
    });

    it('calls onClose when Cancel button is clicked', async () => {
        const onClose = jest.fn();
        render(<GenerateMoreModal {...defaultProps} onClose={onClose} />);

        const cancelButton = screen.getByText('Cancel');
        await userEvent.click(cancelButton);

        expect(onClose).toHaveBeenCalled();
    });

    it('shows success message after generation starts', async () => {
        const mockGenerateResponse = {
            job_id: 'job-123',
            status: 'queued' as const,
        };
        mockApi.generateViews.mockResolvedValueOnce(mockGenerateResponse);

        render(<GenerateMoreModal {...defaultProps} />);

        const extendButton = screen.getByTestId('generate-extend-button');
        await userEvent.click(extendButton);

        await waitFor(() => {
            expect(screen.getByText('Generation started successfully!')).toBeInTheDocument();
        });
    });

    it('refreshes router after successful generation', async () => {
        const mockGenerateResponse = {
            job_id: 'job-123',
            status: 'queued' as const,
        };
        mockApi.generateViews.mockResolvedValueOnce(mockGenerateResponse);

        render(<GenerateMoreModal {...defaultProps} />);

        const extendButton = screen.getByTestId('generate-extend-button');
        await userEvent.click(extendButton);

        await waitFor(() => {
            expect(mockRefresh).toHaveBeenCalled();
        });
    });
});
