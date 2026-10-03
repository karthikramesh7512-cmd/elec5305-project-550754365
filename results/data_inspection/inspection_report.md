# Ultrasonic data inspection

Run `python inspect_ultrasonic_data.py` from the project root. Only experimental
batches 013–019 are opened. All seven streams were decompressed to EOF; only 11
images per batch (77 total) were retained for statistics: the first occurrence of
each of seven categories, plus metadata rows 250, 500, 750, and 1000.
Seven representative figures use batch 013. Each contains the full image, the
README crop, and an unprocessed signed trace. No DSP, ML, or new splits were made.

## Directly documented facts

The [repository README](https://github.com/koomas/NDT_ML_Flaw) specifies XZ/LZMA
compression, 1,000 images per batch, dimensions 480 x 7168, and flaw area 1100–3100.
It identifies the metadata location as scan-axis location, but supplies no binary
dtype, endianness, serialization-order specification, or explicit sample IDs.

[Koskinen et al., Sections 2.1 and 2.3](https://doi.org/10.1007/s10921-021-00757-x)
describe 16-bit acquisition, 480 scan steps, and subsequent RF rectification,
sound-path cropping to 2,000 samples, max-pooling, and normalization.

## Data-supported decoding and axes

Each file decompresses to exactly 6,881,280,000 bytes, divisible into 1,000
records of 6,881,280 bytes. Interpret each record with
`np.frombuffer(payload, dtype='<i2').reshape(480, 7168)` in C order.
No NumPy/pickle/header wrapper was found. The proposed headerless int16 decoding
matches the documented dimensions/count, smooth oscillating signals, coherent
images, and metadata-aligned changes. Byte-swapped/unsigned comparisons are in
`decode_comparison.csv`; those alternatives introduce large discontinuities.

Axis 0: 480 scan positions. Axis 1: 7,168 samples along an ultrasonic A-scan
(time/sound-path direction). These are sample indices, not calibrated millimetres.
Figures transpose the array only for display: scan horizontal, sound path vertical.

`image[:, 1100:3100]` has shape (480, 2000). This is a supported interpretation
of the README flaw area, using a zero-based, upper-exclusive Python interval.
The README does not explicitly define endpoint convention. This region contains
all six representative difference rectangles.

The bipolar, oscillatory signals support an RF-form amplitude representation.
They are not a non-negative rectified signal or envelope, and are not zero-mean,
unit-variance normalized. 'Raw RF-form' does not mean untouched instrument data:
these images include virtual flaw augmentation. The complete processing history
and absolute amplitude calibration remain undocumented.

## Metadata alignment

For one-based metadata row i, read zero-based image index i-1. Its decompressed
byte offset is `(i-1) * 6881280`. For zero-based indices, metadata i maps to image i.
The code validates all batch counts and contiguous metadata row indices.

More strongly, each of the six representative flawed images differs from
batch_013:4 (no flaw) in a rectangle whose upper-left indices equal the integer
parts of that row's location and depth. `representative_alignment.csv` records
these checks. Thus, location/depth act like insertion origins in these examples,
not necessarily signal-peak coordinates or flaw centres. Green crosses show those
metadata coordinates, not detector outputs.

This verifies the positional interpretation for six examples and supports the
serialization layout. It is not an independent identity check of every one of
the 7,000 images; the files contain no embedded sample identifiers.

## Measured statistics and scaling

Every sampled image has positive and negative values. Across all 77 sampled
images: minimum -14,935; maximum 14,144; pooled mean -68.066939; pooled population
standard deviation 637.660884. Individual-image means span -68.333937 to
-67.771762; standard deviations span 626.623077 to 672.153455. NaN and Inf counts
are zero (integer storage cannot represent them). These are sample statistics,
not whole-batch amplitude extrema. Full-stream decompression checks integrity/counts.

For composition-matched comparisons, use one no-flaw and one of each P41 label
per batch (seven images each): mean image std ranges 640.635399–640.899174,
a max/min ratio of 1.000412. Mean crop std ranges 534.335163–535.552815,
a ratio of 1.002279. No substantial batch scaling difference is evident in this
small deterministic sample. Different source sizes/augmentation amounts and the
shared background limit what this comparison establishes; no population-wide
gain calibration is claimed. See `batch_scaling.csv` and `batch_scaling.png`.

All image panels share display limits +/-3616. Saturation is for display only;
no values are changed in the stored arrays or numerical statistics. Raster
figures necessarily display fewer pixels than the original full-resolution arrays.

Per-image statistics and sample references: `data/image_data_audit.csv`.
Unresolved: precise sampling frequency, time origin, sound speed/depth conversion,
absolute amplitude units, full acquisition/processing history, and an authoritative
binary serialization/row-order specification.
