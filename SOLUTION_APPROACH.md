# Solution Approach: The "Constraint-First" Hybrid Pipeline

## 1. The Challenge
Virtual try-on (VTO) for Indian sarees is uniquely difficult compared to western clothing:
- **Complex Draping**: A saree is a single 6-9 yard piece of fabric draped in intricate folds, unlike a shirt or pant which has a fixed topology.
- **Identity Preservation**: Users demand that specific border motifs, pallu patterns, and zari work are preserved exactly. Generative AI models often "hallucinate" new patterns or simplify complex weaves.
- **Structural Integrity**: A saree must look like it is physically wrapped around the body, obeying gravity and tension, not just "painted on".

## 2. Our Philosophy: "Constraint-First" Generation
Instead of asking a Generative AI model to "draw a woman in a saree" (which yields high creativity but low fidelity), we use a **Hybrid Pipeline** that treats AI as a texturing engine, not a structural architect.

We impose strict **Determinism** on the geometry and use AI only for **Photorealism**.

### Core Principles
1.  **Geometry is Fixed**: We do not guess the drape. We use pre-computed, physically simulated 3D assets (Placement Maps) that define exactly where the fabric falls.
2.  **Texture is Extracted**: We do not generate motifs. We extract high-resolution textures from the user's uploaded photo using classical Computer Vision.
3.  **AI is Controlled**: We use Generative AI (Gemini) *only* to blend the extracted texture onto the fixed geometry, handling the complex lighting and micro-folds that are too hard to simulate manually.

## 3. The "Sandwich" Architecture

Our pipeline matches a "CV - AI - CV" sandwich pattern:

### Phase 1: Classical Computer Vision (The "Bun")
*Objective: Prepare deterministic inputs.*
1.  **Segmentation**: Use U2Net/IS-Net to cleanly separate the saree from the floor background.
2.  **Planar Flattening**: Use homography and warping to "iron" the saree into a flat texture map (`S_flat.png`).
3.  **Semantic Slicing**: Cut the flat saree into logical parts (Pallu, Body, Borders) using a strict coordinate layout. **This guarantees that the pallu is always the pallu, never accidental noise.**

### Phase 2: Constrained Generative AI (The "Meat")
*Objective: Realistic synthesis.*
1.  **Composition**: We feed the AI the *exact* pixel parts and a *precise* placement guide (grayscale depth/flow map).
2.  **Prompt Engineering**: We use "negative prompting" and imperative instructions ("Preserve border geometry exactly", "Do not invent motifs") to lock the AI's creativity.
3.  **In-painting/Blending**: The AI's job is limited to warping the extraction to match the folds in the placement map and correcting the lighting to match the model.

### Phase 3: Validation & Refinement (The "Top Bun")
*Objective: Quality Assurance.*
1.  **Metric Verification**: We compare the output against the input using SSIM (Structure) and Delta-E (Color).
2.  **Automated Reject/Retry**: If the AI hallucinates (e.g., changes the color from red to pink), the system detects the Delta-E spike, rejects the image, and retries with a stricter prompt and a different random seed.

## 4. Why This Works
| Feature | Pure Generative AI | Our Hybrid Approach |
| :--- | :--- | :--- |
| **Texture Fidelity** | Low (invents patterns) | **High** (uses original pixels) |
| **Drape Realism** | Variable (often defying physics) | **Perfect** (based on 3D sim) |
| **Consistency** | Low (random every time) | **High** (deterministic pipeline) |
| **Control** | "Prompt and Pray" | **Engineering Guarantees** |

## 5. Iterative Discovery
We acknowledge that AI has a "latent space" of variability. Our "Generate More" feature allows users to explore this space safely. By locking the pose and the structure, the only variable left is the "micro-style" (lighting nuances, fold crispness), allowing users to find the perfect render without risking the integrity of the saree design.
