# Frontend Documentation

## Technology Stack

- **Framework**: Next.js 16 (App Router)
- **Language**: TypeScript
- **Styling**: Tailwind CSS v4
- **Components**: shadcn/ui with neutral theme
- **Animation**: Framer Motion (subtle transitions)
- **Image Rendering**: Next.js `<Image />` component
- **Testing**: Jest + React Testing Library + MSW

## Environment Variables

```bash
NEXT_PUBLIC_API_BASE=http://localhost:8000  # Backend API base URL
```

## Routes

| Route | Description |
|-------|-------------|
| `/` | Redirects to `/gallery` |
| `/gallery` | Gallery view - grid of saree folders |
| `/gallery/[saree_id]` | Folder view - artifacts and generation timeline |

## Key Design Constraints (Non-Negotiable)

1. **Initial Generation**: Always uses 4 standard views. Frontend never allows pose selection during initial generation.

2. **View Labels**: Always use "View 1", "View 2", etc. Never expose pose IDs or internal filenames.

3. **Generate More Views**: Only two options available:
   - "Generate remaining views" → `mode: "extend"`
   - "Retry failed views" → `mode: "retry_failed"` (disabled if no failures)

4. **No Manual Editing**: No image editing, pose manipulation, or prompt tweaking.

## Component Structure

```
components/
├── ui/                     # shadcn/ui base components
│   ├── button.tsx
│   ├── dialog.tsx
│   ├── tabs.tsx
│   ├── badge.tsx
│   └── card.tsx
├── UploadButton.tsx        # Upload flow + auto-generate standard views
├── GalleryTile.tsx         # Gallery grid tile
├── GenerateMoreModal.tsx   # Controlled generation modal
├── ArtifactTabs.tsx        # Original/Cleaned/Flattened/Parts tabs
├── GenerationCard.tsx      # Generation snapshot with view grid
├── ImageLightbox.tsx       # Read-only image viewer
└── StatusBadge.tsx         # Status indicator badges
```

## API Client (`lib/api.ts`)

| Function | Endpoint | Description |
|----------|----------|-------------|
| `uploadSaree(file)` | `POST /api/upload` | Upload saree image |
| `generateViews(id, mode)` | `POST /api/generate` | Trigger generation |
| `getJobStatus(jobId)` | `GET /api/status/:id` | Poll job status |
| `getGallery()` | `GET /api/gallery` | List saree folders |
| `getSareeDetails(id)` | `GET /api/gallery/:id` | Get folder details |
| `getLogs(jobId)` | `GET /api/logs/:id` | Get retry logs |

## Running the Application

```bash
# Development
cd frontend && npm install && npm run dev

# Run tests
npm run test

# Build for production
npm run build

# Start production server
npm run start
```

## Verification Checklist

- [ ] `npm install` completes without errors
- [ ] `npm run dev` starts app on http://localhost:3000
- [ ] Upload triggers `POST /api/upload` then `POST /api/generate` with `mode=standard`
- [ ] Gallery shows tiles from `GET /api/gallery` with thumbnails and status badges
- [ ] Folder view shows artifacts tabs and generation cards with "View N" labels
- [ ] Generate More modal sends `mode=extend` or `mode=retry_failed` only
- [ ] No pose IDs or internal filenames visible in UI
- [ ] `npm run test` passes all tests