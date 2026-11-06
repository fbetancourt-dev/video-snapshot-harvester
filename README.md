# Video Snapshot Harvester (`video-snapshot` / `video-harvest`)

High-precision forensic video keyframe and studio snapshot harvester designed to extract, filter, morphometrically analyze, and curate training datasets for computer vision, biometric enrollment, anthropometrics, and somatotype recognition.

## Features

- **Traceable Naming Protocol**: Every output snapshot embeds the original video filename and exact extracted timestamp: `<video_name>_snap_t<MM>m<SS>s_<millis>ms.jpg`.
- **Studio Aesthetic & Pose Quality Scoring**: Evaluates sharpness (Laplacian variance), studio lighting balance, dynamic contrast, pose stillness, and biometric presence.
- **Anatomical Multi-Zone Breakdown**: Isolates 9 anatomical zones: `cara`, `cuerpo`, `breast`, `pompi`, `frente`, `aerola`, `piernas`, `brazos`, `torso`.
- **Comprehensive Morphometrics & Anthropometry**:
  - **Facial Features**: Eyebrows (style, arch, color), Eye color (iris RGB/Hex), Hair color, Face shape (Oval, Round, Heart, Square, Oblong, Diamond), Nose shape (Straight, Upturned, Broad), Ear shape (Lobe type, Proportions), Lips fullness.
  - **Areola Morphology**: Shape classification (Circular, Oval Horizontal/Vertical, Elliptical), aspect ratio, eccentricity.
  - **Skin Tone & Complexion**: RGB/Lab color, HEX, ITA angle, Fitzpatrick phototype (Type I–VI), texture smoothness.
  - **Silhouette & Somatotype**: Body archetype (Hourglass, Pear, Inverted Triangle, Athletic/Rectangle), WHR, SHR.
  - **Anthropometry**: Hand size (Small, Proportional, Long) and hand-to-face proportion, Estimated stature in cm via cephalic canon.
- **Structured JSON Catalog**: Generates `SNAPSHOTS_METADATA.json` with all morphological vectors.
- **Persistent SQLite Database (`--enable-db` / `--db`)**: Stores full metadata, indexable queries, and optional raw binary image BLOBs (`--store-blob`).
- **Antigravity Agent Supervision (`--agy-supervision`)**: Dispatches candidate snapshots to the `agy` CLI for cognitive supervision and caption enhancement.
- **Diverse Situation Discovery**: Segments videos into $N$ distinct situations/scenes and isolates the single peak studio moment for each.
- **Direct Timestamps Extraction**: Supports custom millisecond-precise timestamps (`--timestamps "00:01:23.500, 00:04:12"`).
- **Studio Contact Sheet**: Builds a high-resolution HUD grid visual summary (`*_STUDIO_CONTACT_SHEET.jpg`).
- **Desktop Thumbnail Pre-caching**: Synchronously populates GNOME/Nautilus 512px and 256px caches.

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
# Studio mode discovering 8 situations with full anatomical crops, SQLite DB, and AGY supervision
video-snapshot /path/to/stream.mp4 -m eva --studio --situations 8 --crop-regions --enable-db --agy-supervision

# Extract exact studio timestamps with image BLOBs inside SQLite
video-snapshot /path/to/video.mp4 -m eva --timestamps "00:01:24.500,00:05:10,00:09:42" --crop-regions --enable-db --store-blob

# Generate studio contact sheet only
video-snapshot /path/to/video.mp4 --studio --situations 12 --contact-sheet
```

## License
MIT License.
