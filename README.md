# Image to 3D

Transform any image into an interactive 3D scene using Apple's SHARP model for Gaussian splatting.

## Features

- **Drag-and-drop image upload** - PNG, JPG, WEBP support (up to 10MB)
- **Real-time 3D viewer** - Interactive Gaussian splat rendering with Three.js
- **Fast processing** - SHARP model runs in under 1 second on GPU (after warm-up)
- **Downloadable output** - Export generated PLY files

## Setup

### 1. Install Dependencies

```bash
npm install
```

### 2. Set Up Modal

1. Create a [Modal.com](https://modal.com) account
2. Install the Modal CLI:
   ```bash
   pip install modal
   ```
3. Authenticate with Modal:
   ```bash
   modal token new
   ```
4. Deploy the SHARP endpoint:
   ```bash
   modal deploy modal/sharp_api.py
   ```
5. Copy the endpoint URL from the Modal dashboard (looks like `https://your-username--image-to-3d-sharp-sharpmodel-generate.modal.run`)

### 3. Configure Environment

Create a `.env.local` file in the project root:

```bash
MODAL_ENDPOINT_URL=https://your-username--image-to-3d-sharp-sharpmodel-generate.modal.run
```

### 4. Run Development Server

```bash
npm run dev
```

Open [http://localhost:3000](http://localhost:3000) to use the app.

## Modal Endpoint

The Modal endpoint (`modal/sharp_api.py`) deploys Apple's SHARP model as a serverless GPU function:

- **GPU**: A10G (can be changed to A100 for faster processing)
- **Cold start**: ~30-60 seconds (first request after idle)
- **Warm inference**: <1 second
- **Idle timeout**: 5 minutes (configurable)

### Local Testing

You can test the Modal endpoint locally:

```bash
modal serve modal/sharp_api.py
```

Then use the local URL printed in the terminal.

### Deploying to Production

```bash
modal deploy modal/sharp_api.py
```

## Usage

1. Open the app in your browser
2. Drag and drop an image (or click to select)
3. Wait for processing (first request may take 30-60s due to cold start)
4. Explore the 3D scene:
   - **Drag** to rotate
   - **Scroll** to zoom/fly through
   - **Reset** button to return to default view
   - **Expand** for fullscreen mode
   - **Download** to save the PLY file

## Troubleshooting

### "Modal endpoint not configured"
Make sure you've set `MODAL_ENDPOINT_URL` in your `.env.local` file.

### Cold start taking too long
The first request after 5+ minutes of inactivity will trigger a cold start. You can:
- Increase `scaledown_window` in `sharp_api.py`
- Pre-warm the endpoint with a dummy request

### 3D viewer not loading
Check browser console for WebGL errors. The viewer requires WebGL 2.0 support.
