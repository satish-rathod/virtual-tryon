import '@testing-library/jest-dom';

// Mock next/navigation
jest.mock('next/navigation', () => ({
    useRouter: () => ({
        push: jest.fn(),
        replace: jest.fn(),
        refresh: jest.fn(),
        back: jest.fn(),
        forward: jest.fn(),
        prefetch: jest.fn(),
    }),
    usePathname: () => '/',
    useSearchParams: () => new URLSearchParams(),
    redirect: jest.fn(),
}));

// Mock next/image - use simple function syntax for SWC compatibility
jest.mock('next/image', () => ({
    __esModule: true,
    default: function MockImage(props: { src: string; alt: string;[key: string]: unknown }) {
        // Create a simple img element
        const imgProps: Record<string, unknown> = {};
        Object.keys(props).forEach((key) => {
            // Only include valid HTML img attributes
            if (['src', 'alt', 'width', 'height', 'className', 'style', 'id'].includes(key)) {
                imgProps[key] = props[key];
            }
        });
        return require('react').createElement('img', imgProps);
    },
}));
