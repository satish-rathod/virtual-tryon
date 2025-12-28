/**
 * API client for Saree Virtual Try-On backend
 * All endpoints follow the contracts in docs/API.md
 */

const API_BASE = process.env.NEXT_PUBLIC_API_BASE || 'http://localhost:8000';

// ============================================================================
// Types
// ============================================================================

export interface UploadResponse {
    saree_id: string;
    upload_path: string;
}

export interface GenerateResponse {
    job_id: string;
    status: 'queued' | 'running' | 'success' | 'failed';
}

export interface JobStatus {
    job_id: string;
    status: 'queued' | 'running' | 'success' | 'partial' | 'failed';
    progress: number;
    current_stage?: string;
    artifacts?: string[];
    metrics_url?: string;
}

export interface GalleryItem {
    saree_id: string;
    created_at: string;
    thumbnail: string;
    generation_count: number;
    latest_status: 'success' | 'partial' | 'failed' | 'running' | 'queued';
}

export interface Generation {
    generation_id: string;
    label: 'Standard Views' | 'Additional Views' | 'Retry Failed';
    mode: 'standard' | 'extend' | 'retry_failed';
    status: 'success' | 'partial' | 'failed' | 'running' | 'queued';
    timestamp: string;
    views: ViewArtifact[];
    retry_count: number;
    metrics_url?: string;
}

export interface ViewArtifact {
    view_number: number;
    image_url: string;
    status: 'success' | 'failed' | 'pending';
}

export interface SareeDetails {
    saree_id: string;
    created_at: string;
    artifacts: {
        original: string;
        cleaned?: string;
        flattened?: string;
        parts?: {
            body?: string;
            pallu?: string;
            top_border?: string;
            bottom_border?: string;
        };
    };
    generations: Generation[];
    has_failures: boolean;
}

export interface LogsResponse {
    job_id: string;
    retry_log: Array<{
        attempt: number;
        reason: string;
        timestamp: string;
    }>;
    failure_reasons: string[];
}

// ============================================================================
// API Functions
// ============================================================================

/**
 * Upload a saree image
 * POST /api/upload (multipart form-data)
 */
export async function uploadSaree(file: File): Promise<UploadResponse> {
    const formData = new FormData();
    formData.append('file', file);

    const response = await fetch(`${API_BASE}/api/upload`, {
        method: 'POST',
        body: formData,
    });

    if (!response.ok) {
        const error = await response.text();
        throw new Error(`Upload failed: ${error}`);
    }

    return response.json();
}

/**
 * Trigger generation for a saree
 * POST /api/generate
 * 
 * Modes:
 * - standard: Initial generation (poses 01-04, always 4 views)
 * - extend: Generate remaining views (poses 05-12)
 * - retry_failed: Re-run only failed outputs
 */
export async function generateViews(
    sareeId: string,
    mode: 'standard' | 'extend' | 'retry_failed' = 'standard'
): Promise<GenerateResponse> {
    const response = await fetch(`${API_BASE}/api/generate`, {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify({
            saree_id: sareeId,
            mode,
        }),
    });

    if (!response.ok) {
        const error = await response.text();
        throw new Error(`Generate failed: ${error}`);
    }

    return response.json();
}

/**
 * Poll job status
 * GET /api/status/:job_id
 */
export async function getJobStatus(jobId: string): Promise<JobStatus> {
    const response = await fetch(`${API_BASE}/api/status/${jobId}`);

    if (!response.ok) {
        const error = await response.text();
        throw new Error(`Failed to get job status: ${error}`);
    }

    return response.json();
}

/**
 * Get gallery list
 * GET /api/gallery
 */
export async function getGallery(): Promise<GalleryItem[]> {
    const response = await fetch(`${API_BASE}/api/gallery`, {
        cache: 'no-store', // Always fetch fresh data
    });

    if (!response.ok) {
        const error = await response.text();
        throw new Error(`Failed to get gallery: ${error}`);
    }

    return response.json();
}

/**
 * Get saree details including artifacts and generations
 * GET /api/gallery/:saree_id or constructed from artifacts
 */
export async function getSareeDetails(sareeId: string): Promise<SareeDetails> {
    const response = await fetch(`${API_BASE}/api/gallery/${sareeId}`, {
        cache: 'no-store',
    });

    if (!response.ok) {
        const error = await response.text();
        throw new Error(`Failed to get saree details: ${error}`);
    }

    return response.json();
}

/**
 * Get retry logs for a job
 * GET /api/logs/:job_id
 */
export async function getLogs(jobId: string): Promise<LogsResponse> {
    const response = await fetch(`${API_BASE}/api/logs/${jobId}`);

    if (!response.ok) {
        const error = await response.text();
        throw new Error(`Failed to get logs: ${error}`);
    }

    return response.json();
}

/**
 * Build artifact URL
 * The artifact path may include generation folder
 */
export function getArtifactUrl(sareeId: string, artifactPath: string): string {
    if (artifactPath.startsWith('http')) return artifactPath;
    if (artifactPath.startsWith('/')) return `${API_BASE}${artifactPath}`;
    return `${API_BASE}/api/artifacts/${sareeId}/${artifactPath}`;
}

/**
 * Build thumbnail URL for a saree
 */
export function getThumbnailUrl(sareeId: string): string {
    return getArtifactUrl(sareeId, 'original.jpg');
}

/**
 * Poll job status until completion
 * Returns final status or throws on timeout
 */
export async function pollJobStatus(
    jobId: string,
    intervalMs: number = 2000,
    maxAttempts: number = 150 // 5 minutes at 2s intervals
): Promise<JobStatus> {
    let attempts = 0;

    while (attempts < maxAttempts) {
        const status = await getJobStatus(jobId);

        if (status.status === 'success' || status.status === 'failed' || status.status === 'partial') {
            return status;
        }

        await new Promise(resolve => setTimeout(resolve, intervalMs));
        attempts++;
    }

    throw new Error('Job polling timeout');
}
