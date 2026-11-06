# Video Snapshot Harvester (`video-snapshot` / `video-harvest`)

High-precision forensic video keyframe and studio snapshot harvester designed to extract, filter, and curate training datasets for computer vision, biometric enrollment, and somatotype recognition.

## Features

- **Traceable Naming Protocol**: Every output snapshot embeds the original video filename and exact extracted timestamp: `<video_name>_snap_t<MM>m<SS>s_<millis>ms.jpg`.
- **Studio Aesthetic & Pose Quality Scoring**: Evaluates sharpness (Laplacian variance), studio lighting balance, dynamic contrast, pose stillness, and biometric presence to capture flattering, high-clarity moments.
- **Diverse Situation Discovery**: Segments videos into $N$ distinct situations/scenes and isolates the single peak studio moment for each situation.
- **Explicit Timestamp Extraction**: Direct capture at arbitrary millisecond-precise timestamps (`--timestamps "00:01:23.500, 00:04:12"`).
- **Perceptual Deduplication**: 64-bit difference hash (dHash) prevents saving duplicate stationary frames.
- **Anatomical Cropping**: Integrates with YOLOv8-Pose and YuNet to generate clean sub-region crops (`crops/face`, `crops/torso`, `crops/hips`).
- **Studio Contact Sheet**: Builds a high-resolution HUD grid visual summary (`*_STUDIO_CONTACT_SHEET.jpg`) for fast visual curation.
- **Desktop Thumbnail Pre-caching**: Synchronously populates GNOME/Nautilus 512px and 256px caches.
- **Seamless Dataset & Training Integration**: Direct integration with `model-media` for automated profile re-training.

## Installation & Symlinks

```bash
cd /home/fbetancourt/Gemini/video-snapshot-harvester
chmod +x video_snapshot_harvester.py
ln -sf $(pwd)/video_snapshot_harvester.py ~/.local/bin/video-snapshot
ln -sf $(pwd)/video_snapshot_harvester.py ~/.local/bin/video-harvest
```

## Running Tests

```bash
python3 -B -m unittest discover -s tests -v
```

## Usage Examples

```bash
# General sampling every 2 seconds with sub-anatomical crops
video-snapshot /path/to/stream.mp4 -m eva --fps 0.5 --crop-regions

# Studio mode discovering 8 diverse situations
video-snapshot /path/to/video.mp4 -m eva --studio --situations 8 --crop-regions

# Harvest specific timestamps chosen by user or AI assistant
video-snapshot /path/to/video.mp4 -m eva --timestamps "00:01:24.500,00:05:10,00:09:42.250"

# Generate studio contact sheet only
video-snapshot /path/to/video.mp4 --studio --situations 12 --contact-sheet
```

## License
MIT License.
